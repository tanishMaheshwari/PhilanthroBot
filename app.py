import os
import shutil
import uuid
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

    for ch in chunks:
        src = os.path.basename(ch.metadata.get("source", ""))
        # Derive a readable NGO title from filename (strip numeric prefixes/underscores)
        title = os.path.splitext(src)[0]
        title = title.replace("_", " ")
        # Optional: special cases mapping if you want nicer names
        # name_map = {"6 pratham education foundation": "Pratham Education Foundation", ...}
        # title = name_map.get(title.lower(), title)
        ch.metadata["ngo_title"] = title

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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1) # Using gemini-2.0-flash as it's a newer model
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


def fmt(d):
    src = os.path.basename(d.metadata.get("source", ""))
    name = d.metadata.get("ngo_title", os.path.splitext(src)[0].replace("_", " "))
    return f"[NGO: {name} | File: {src}]\n{d.page_content}"

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
        prompt = ChatPromptTemplate.from_template("""
            You are **PhilanthroBot**, an intelligent, trustworthy assistant that helps users explore and compare NGOs.

            Use the **provided context from NGO documents** and the **chat history** to answer the user's question.

            Guidelines:
            - Keep responses concise (3-6 sentences) unless the user asks for detail.
            - If multiple NGOs are relevant, mention them by name and summarize each briefly.
            - Use natural language (avoid lists unless necessary).
            - If the answer isn't clearly in the context, say:
            "I'm not sure based on the available NGO profiles, but I can help you explore related organizations."

            **Chat History:**
            {chat_history}

            **Context:**
            {context}

            **User Message:**
            {input}

            PhilanthroBot:
            """)
        chain = prompt | llm
        # context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        context = "\n\n".join(fmt(d) for d in retrieved_docs)
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


# --- Main Interaction Loop ---

if __name__ == "__main__":
    print("Setting up PhilanthroBot...")
    retriever = build_rag_pipeline()
    
    if retriever:
        checkpointer = MemorySaver()
        builder = build_graph()
        app = builder.compile(checkpointer=checkpointer)
        # with open("graph.png", "wb") as f:
        #     f.write(app.get_graph().draw_mermaid_png())
        print("\nPhilanthroBot is ready! How can I help you find an NGO to support?")
        thread_id = str(uuid.uuid4())
        config = RunnableConfig(configurable={"thread_id": thread_id})
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("PhilanthroBot: Goodbye!")
                break
            events = app.stream({"messages": [HumanMessage(content=user_input)]}, config=config)
            final_message = None
            for event in events:
                if "generate_response" in event:
                    final_message = event["generate_response"]["messages"][-1]
                elif "update_preferences" in event:
                    final_message = event["update_preferences"]["messages"][-1]
            if final_message:
                print(f"PhilanthroBot: {final_message.content}")
