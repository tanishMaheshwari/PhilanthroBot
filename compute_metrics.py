import os
import shutil
import uuid
import sys
from dotenv import load_dotenv
from typing import List, Annotated
import time
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- MODIFIED IMPORT ---
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from metrics import evaluate_retrieval, write_dict_csv
from metrics import evaluate_ragas

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset as HFDataset


# --- 1. SETUP AND CONFIGURATION ---

load_dotenv()  # Load environment variables from .env file
# Ensure you have set the GOOGLE_API_KEY environment variable
if 'GOOGLE_API_KEY' not in os.environ:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# Define directories
PROFILE_DIR = "./ngo_profiles"
DB_DIR = "./chroma_db"

# --- 2. PHASE 1: KNOWLEDGE BASE CONSTRUCTION (FOR PDFs) ---

doc_embed = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
    transport="rest",
    request_options={"timeout": 20},
)

query_embed = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    task_type="RETRIEVAL_QUERY",
    transport="rest",
    request_options={"timeout": 20},
)



def build_rag_pipeline():
    if not os.path.exists(PROFILE_DIR):
        print(f"Error: Profile directory '{PROFILE_DIR}' not found.")
        return None

    # Load documents and split
    loader = DirectoryLoader(PROFILE_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    for i, ch in enumerate(chunks):
        src_path = ch.metadata.get("source", "")
        src_name = os.path.basename(src_path)
        ch.metadata["doc_id"] = src_name          # parent document id
        ch.metadata["chunk_id"] = f"{src_name}-{i}"


    collection_name = "ngos_gemini001"  # keep per-model/dimension collections separate

    # Reuse existing DB if present; else build it once
    if os.path.exists(DB_DIR):
        # Load for serving with the QUERY encoder
        vector_store = Chroma(
            persist_directory=DB_DIR,
            collection_name=collection_name,
            embedding_function=query_embed,
        )
    else:
        # Build with the DOCUMENT encoder (first time)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=doc_embed,
            persist_directory=DB_DIR,
            collection_name=collection_name,
        )
        # Re-open with QUERY encoder for serving consistency in this run
        vector_store = Chroma(
            persist_directory=DB_DIR,
            collection_name=collection_name,
            embedding_function=query_embed,
        )

    # You can tune k or add metadata filters via search_kwargs later
    return vector_store.as_retriever( search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 60, "lambda_mult": 0.3},
    )


# --- 3. PHASE 2: STATEFUL AGENT WITH LANGGRAPH ---

class UserPreferences(TypedDict):
    """Structure to hold the user's learned preferences."""
    causes: List[str]
    locations: List[str]

class AgentState(TypedDict):
    """Defines the main state for the entire graph."""
    messages: Annotated[list, add_messages]
    preferences: UserPreferences
    retrieved_docs: List[Document]
    latest_intent: str

# Define the LLM and build the retriever
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    transport="rest",
)
retriever = None # Will be initialized in the main block

# --- Graph Nodes (No changes needed here) ---

def classify_intent_node(state: AgentState):
    """
    Analyzes the latest user message *in the context of the conversation* to determine its purpose.
    """
    
    # Get the full message history, filtering out any System confirmation messages
    messages = [msg for msg in state["messages"] if msg.name != "System"]
    last_human_message = messages[-1].content
    history_messages = messages[:-1]
    
    # Don't create history if it's the very first message
    chat_history = ""
    if history_messages:
        chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages])

    # Updated prompt that includes chat history and clearer instructions
    prompt = ChatPromptTemplate.from_template(
        """Given the chat history and the user's latest message, classify the user's intent.
Choose from one of the following single-word classifications: 
'preference_update' - User is stating a new preference (e.g., "I care about animals", "I am in Mumbai")
'recommendation_request' - User is asking for a new recommendation (e.g., "Find me an NGO")
'question' - User is asking a follow-up question about a previous topic, a general question, or affirming a bot's question (e.g., "Tell me more", "What is Pratham?", "yes", "what is its name?")
'greeting' - (e.g., "hi", "hello")
'goodbye' - (e.g., "bye", "thanks that's all")

Return only the single-word classification.

**Chat History:**
{chat_history}

**User Message:** {user_message}

Classification:"""
    )
    
    chain = prompt | llm
    
    # Handle the case of the very first message where history is empty
    if not chat_history:
        chain_input = {"chat_history": "No history yet.", "user_message": last_human_message}
    else:
        chain_input = {"chat_history": chat_history, "user_message": last_human_message}
        
    intent = chain.invoke(chain_input).content.strip()
    
    # I've added a print statement so you can see the classification in your terminal
    print(f"--- 0. CLASSIFIED INTENT: {intent} ---") 
    
    return {"latest_intent": intent}


def update_preferences_node(state: AgentState):
    """Parses the user's message to extract and store preferences."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at extracting user preferences for philanthropic causes and locations from a message.
Return a JSON object with two keys: 'causes' and 'locations', listing any extracted terms. If none, return empty lists."""),
        ("human", "{user_message}")
    ])
    parser = JsonOutputParser(pydantic_object=UserPreferences)
    chain = prompt | llm | parser

    user_message = state["messages"][-1].content
    extracted_prefs = chain.invoke({"user_message": user_message})

    current_prefs = state.get("preferences", {"causes": [], "locations": []})
    current_prefs["causes"] = list(set(current_prefs["causes"] + extracted_prefs.get("causes", [])))
    current_prefs["locations"] = list(set(current_prefs["locations"] + extracted_prefs.get("locations", [])))

    confirmation_message = "Thanks! I've updated your preferences."
    return {
        "preferences": current_prefs,
        "messages": [HumanMessage(content=confirmation_message, name="System")]
    }

def retrieve_documents_node(state: AgentState):
    """
    Constructs a standalone query from the chat history and retrieves relevant documents.
    """
    print("--- 1. RETRIEVING DOCUMENTS ---")
    
    # Get the full message history
    messages = state["messages"]
    # Get the last human message
    last_human_message = messages[-1].content
    
    # Get history *before* the last message
    history_messages = messages[:-1]
    chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages])

    # 1. Create a query-rewriting prompt
    # This prompt helps the LLM turn a follow-up question into a standalone query
    query_gen_prompt = ChatPromptTemplate.from_template(
        """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {input}

Standalone question:"""
    )
    
    # 2. Create the query-rewriting chain
    query_gen_chain = query_gen_prompt | llm
    
    # 3. Generate the standalone query
    generated_query = query_gen_chain.invoke({
        "chat_history": chat_history,
        "input": last_human_message
    }).content

    print(f"--- Generated Search Query: {generated_query} ---")
    
    # 4. Add preferences to the query (as you did before)
    prefs = state.get("preferences", {})
    final_query = generated_query
    if prefs.get("causes"): final_query += f" related to causes like {', '.join(prefs['causes'])}"
    if prefs.get("locations"): final_query += f" in locations like {', '.join(prefs['locations'])}"

    # 5. Retrieve documents with the new, standalone query
    docs = retriever.invoke(final_query)
    print(f"--- Retrieved {len(docs)} documents ---")
    return {"retrieved_docs": docs}


def generate_response_node(state: AgentState):
    """
    Generates a response, adapting whether documents were retrieved or not,
    and now includes chat history for context.
    """
    print("--- 2. GENERATING RESPONSE ---")
    
    # Get the full message history, filtering out any System confirmation messages
    messages = [msg for msg in state["messages"] if msg.name != "System"]
    last_human_message = messages[-1].content
    history_messages = messages[:-1]
    chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages])

    retrieved_docs = state.get("retrieved_docs")

    if not retrieved_docs:
        # If no documents are present, this is a general chat interaction
        print("--- Using General Chat Prompt ---")
        prompt = ChatPromptTemplate.from_template(
            """You are PhilanthroBot, a helpful and friendly AI assistant.
            Provide a simple, conversational response based on the chat history.

            **Chat History:**
            {chat_history}
            
            **Human:** {input}
            **AI:**"""
        )
        chain = prompt | llm
        response = chain.invoke({"chat_history": chat_history, "input": last_human_message})
    else:
        # If documents were retrieved, perform RAG to answer the question
        print("--- Using RAG Prompt ---")
        prompt = ChatPromptTemplate.from_template(
"""You are PhilanthroBot, a helpful AI assistant for discovering trustworthy NGOs.
Answer the user's latest question based on the **chat history** and the **provided context**.
Be conversational and helpful.
If the context doesn't contain the answer, state that you don't have enough information.

**Chat History:**
{chat_history}

**Context from Documents:**
{context}

**Human:** {input}
**AI:**"""
        )
        chain = prompt | llm
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        response = chain.invoke({
            "context": context, 
            "chat_history": chat_history, 
            "input": last_human_message
        })

    return {"messages": [response]}
# --- Conditional Edges (No changes needed here) ---

def route_after_classification(state: AgentState):
    """Decides the next step based on the classified intent."""
    intent = state["latest_intent"]
    if intent == "goodbye": return END
    if intent == "preference_update": return "update_preferences"
    if intent in ["question", "recommendation_request"]: return "retrieve_documents"
    return "generate_response"

# --- Build the Graph ---

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("update_preferences", update_preferences_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {"update_preferences": "update_preferences", "retrieve_documents": "retrieve_documents", "generate_response": "generate_response", END: END}
    )
    workflow.add_edge("update_preferences", END)
    workflow.add_edge("retrieve_documents", "generate_response")
    workflow.add_edge("generate_response", END)
    return workflow


# Keep this prompt near your globals
RAG_PROMPT = ChatPromptTemplate.from_template(
    "Use the context to answer the question concisely. If unknown, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def evaluate_ragas(llm, retriever, dataset, output_csv="ragas_results.csv", batch_sleep=0.0):
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embed = LangchainEmbeddingsWrapper(query_embed)

    records = {"question": [], "answer": [], "contexts": [], "reference": []}
    for ex in dataset:
        q = ex["question"]
        ctx_docs = retriever.invoke(q)
        ctx_texts = [d.page_content for d in ctx_docs]

        answer = (RAG_PROMPT | llm).invoke({"context": "\n\n".join(ctx_texts), "question": q}).content

        reference = ex.get("reference")
        if not reference:
            gts = ex.get("ground_truths", [])
            reference = gts[0] if isinstance(gts, list) and gts else ""

        records["question"].append(q)
        records["answer"].append(answer)
        records["contexts"].append(ctx_texts)
        records["reference"].append(reference)
        if batch_sleep > 0:
            time.sleep(batch_sleep)

    hfds = HFDataset.from_dict(records)
    result = evaluate(
        hfds,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embed,
    )
    df = result.to_pandas()
    df.to_csv(output_csv, index=False)
    return df



# --- Main Interaction Loop ---

# --- Main Interaction Loop ---

if __name__ == "__main__":
    print("Setting up PhilanthroBot...")
    retriever = build_rag_pipeline()
    
    if retriever:
        checkpointer = MemorySaver()
        builder = build_graph()
        app = builder.compile(checkpointer=checkpointer)
        print("\nPhilanthroBot is ready!")
        thread_id = str(uuid.uuid4())
        config = RunnableConfig(configurable={"thread_id": thread_id})

        # Quick Sanity Check
        sample_docs = retriever.invoke("education NGO in India")
        print([ (doc.metadata.get("doc_id"), os.path.basename(doc.metadata.get('source',''))) for doc in sample_docs ])


        # Document-level eval set: use PDF filenames as relevant_ids
        eval_set = [
            {
                "question": "Which NGO focuses on foundational literacy and numeracy in India?",
                "relevant_ids": ["6_pratham_education_foundation.pdf"],
                "reference": "Pratham improves foundational literacy and numeracy in India",
            },
            {
                "question": "Which NGO prioritizes girls' education initiatives?",
                "relevant_ids": ["6_pratham_education_foundation.pdf"],
                "reference": "Pratham runs programs for girls' education",
            },
        ]
        id_getter=lambda d: d.metadata.get("doc_id") or os.path.basename(d.metadata.get("source",""))


        # Retrieval metrics call (document-level)
        retr_results = evaluate_retrieval(
            retriever=retriever,
            dataset=[{"question": ex["question"], "relevant_ids": ex["relevant_ids"]} for ex in eval_set],
            k_values=(3, 6, 10),
            id_getter=lambda d: d.metadata.get("doc_id") or os.path.basename(d.metadata.get("source", "")),
        )
        print("Retrieval metrics (doc-level):", retr_results)
        write_dict_csv("retrieval_metrics.csv", [retr_results])

        # 2) RAGAS metrics (unchanged)
        df = evaluate_ragas(
            llm=llm,
            retriever=retriever,
            dataset=[{"question": ex["question"], "reference": ex["reference"]} for ex in eval_set],
            output_csv="ragas_results.csv",
        )

        print(df.head())
    print("Exiting PhilanthroBot.")
    os._exit(0)