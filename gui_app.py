# streamlit_app.py
import os
import uuid
from dotenv import load_dotenv
from typing import List, Annotated
import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# --- Load environment variables ---
load_dotenv()
if 'GOOGLE_API_KEY' not in os.environ:
    st.error("⚠️ Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

PROFILE_DIR = "./ngo_profiles"
DB_DIR = "./chroma_db"

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

@st.cache_resource(show_spinner=False)
def build_rag_pipeline():
    if not os.path.exists(PROFILE_DIR):
        st.error(f"❌ Profile directory '{PROFILE_DIR}' not found.")
        return None

    loader = DirectoryLoader(PROFILE_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    for ch in chunks:
        src = os.path.basename(ch.metadata.get("source", ""))
        title = os.path.splitext(src)[0].replace("_", " ")
        ch.metadata["ngo_title"] = title

    collection_name = "ngos_gemini001"

    if os.path.exists(DB_DIR):
        vector_store = Chroma(
            persist_directory=DB_DIR,
            collection_name=collection_name,
            embedding_function=query_embed,
        )
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=doc_embed,
            persist_directory=DB_DIR,
            collection_name=collection_name,
        )
        vector_store = Chroma(
            persist_directory=DB_DIR,
            collection_name=collection_name,
            embedding_function=query_embed,
        )

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 60, "lambda_mult": 0.3},
    )

class UserPreferences(TypedDict):
    causes: List[str]
    locations: List[str]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    preferences: UserPreferences
    retrieved_docs: List[Document]
    latest_intent: str

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
retriever = None

def classify_intent_node(state: AgentState):
    messages = [msg for msg in state["messages"] if msg.name != "System"]
    last_human_message = messages[-1].content
    history_messages = messages[:-1]
    chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages]) if history_messages else "No history yet."

    prompt = ChatPromptTemplate.from_template("""
You are an intent classification assistant for a chatbot that helps users discover NGOs to donate to.

You will receive the chat history and the user's latest message.
Your job is to classify the **intent** of the user's latest message into ONE of the following categories:

- `greeting`: user greets or starts a conversation (e.g., "hi", "hello", "good morning")
- `goodbye`: user ends or thanks (e.g., "bye", "thank you", "thanks, that's all")
- `preference_update`: user expresses donation interests (e.g., "I want to help children", "I'm in Delhi", "I care about education in rural areas")
- `recommendation_request`: user asks for NGO suggestions (e.g., "Find me NGOs for climate change", "Recommend an organization in Bangalore")
- `question`: user asks a factual or follow-up question (e.g., "Tell me more about Pratham", "What does this NGO do?", "When was it founded?")

If you're unsure, choose the **closest** category.

Return ONLY one of the above words.

**Chat History:**
{chat_history}

**User Message:** {user_message}

Classification:
""")

    chain = prompt | llm
    intent = chain.invoke({"chat_history": chat_history, "user_message": last_human_message}).content.strip()
    return {"latest_intent": intent}

def update_preferences_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an expert at identifying a user's donation preferences from natural language.

Your task is to extract:
- causes (issues like education, healthcare, environment, animal welfare)
- locations (city, state, country names)

Return a JSON object with two lists:
{{ 
  "causes": [...],
  "locations": [...]
}}

Include only specific items. Avoid generic words like "help" or "charity".
If none are mentioned, return empty lists.

Examples:
Input: "I care about education and women empowerment in Delhi."
Output: {{ "causes": ["education", "women empowerment"], "locations": ["Delhi"] }}

Input: "I'm from Mumbai and I love working with animals."
Output: {{ "causes": ["animal welfare"], "locations": ["Mumbai"] }}
"""),
        ("human", "{user_message}")
    ])

    parser = JsonOutputParser(pydantic_object=UserPreferences)
    chain = prompt | llm | parser

    user_message = state["messages"][-1].content
    extracted_prefs = chain.invoke({"user_message": user_message})

    current_prefs = state.get("preferences", {"causes": [], "locations": []})
    current_prefs["causes"] = list(set(current_prefs["causes"] + extracted_prefs.get("causes", [])))
    current_prefs["locations"] = list(set(current_prefs["locations"] + extracted_prefs.get("locations", [])))

    confirmation_message = "Got it — I've updated your preferences! You can now ask for NGO recommendations based on your interests."
    return {
        "preferences": current_prefs,
        "messages": [HumanMessage(content=confirmation_message, name="System")]
    }

def retrieve_documents_node(state: AgentState):
    messages = state["messages"]
    last_human_message = messages[-1].content
    history_messages = messages[:-1]
    chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages])

    query_gen_prompt = ChatPromptTemplate.from_template("""
You are a smart query reformulator for an NGO recommendation chatbot.

Given the chat history and the user's latest message, rewrite the message into a **standalone question** that captures the full intent.

Preserve the language and tone, but make it contextually complete.
If the user refers to an NGO by "it" or "that", replace it with the actual NGO name mentioned earlier in the chat.

**Chat History:**
{chat_history}

**Follow-up Message:**
{input}

Rewritten standalone question:
""")

    query_gen_chain = query_gen_prompt | llm
    generated_query = query_gen_chain.invoke({
        "chat_history": chat_history,
        "input": last_human_message
    }).content.strip()

    prefs = state.get("preferences", {})
    final_query = generated_query
    if prefs.get("causes"):
        final_query += f" related to causes like {', '.join(prefs['causes'])}"
    if prefs.get("locations"):
        final_query += f" in locations like {', '.join(prefs['locations'])}"

    docs = retriever.invoke(final_query)
    return {"retrieved_docs": docs}

def fmt(d):
    src = os.path.basename(d.metadata.get("source", ""))
    name = d.metadata.get("ngo_title", os.path.splitext(src)[0].replace("_", " "))
    return f"[NGO: {name} | File: {src}]\n{d.page_content}"

def generate_response_node(state: AgentState):
    messages = [msg for msg in state["messages"] if msg.name != "System"]
    last_human_message = messages[-1].content
    history_messages = messages[:-1]
    chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages])
    retrieved_docs = state.get("retrieved_docs")

    if not retrieved_docs:
        prompt = ChatPromptTemplate.from_template("""
You are **PhilanthroBot**, a friendly and intelligent chatbot that helps users discover NGOs matching their values.

Use the chat history to maintain context and answer naturally.
If the user hasn't mentioned what causes or locations they care about, gently ask them.

**Chat History:**
{chat_history}

**Human:** {input}
**PhilanthroBot:** 
""")
        chain = prompt | llm
        response = chain.invoke({"chat_history": chat_history, "input": last_human_message})
    else:
        prompt = ChatPromptTemplate.from_template("""
You are **PhilanthroBot**, an intelligent assistant that helps users explore and compare NGOs.

Use the **provided context from NGO documents** and the **chat history** to answer the user's question.

Guidelines:
- Keep responses concise (3-6 sentences) unless the user explicitly asks for detail.
- If multiple NGOs are relevant, mention them by name and summarize each briefly.
- If asked about the contact information for Paws & Claws Rescue Alliance, provide it as: Email - info@pawsclaws.org
- if asked about contact information for an NGO not in the documents, provide it as:  Phone Number - +91 123456789
- If the answer isn't clearly in the context, say:
  "I'm not sure based on the available NGO profiles, but I can help you explore related organizations."

**Chat History:**
{chat_history}

**Context from NGO Documents:**
{context}

**User Message:**
{input}

PhilanthroBot:
""")
        context = "\n\n".join(fmt(d) for d in retrieved_docs)
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "input": last_human_message
        })

    return {"messages": [response]}

def route_after_classification(state: AgentState):
    intent = state["latest_intent"]
    if intent == "goodbye":
        return END
    if intent == "preference_update":
        return "update_preferences"
    if intent in ["question", "recommendation_request"]:
        return "retrieve_documents"
    return "generate_response"

@st.cache_resource(show_spinner=False)
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
        {
            "update_preferences": "update_preferences",
            "retrieve_documents": "retrieve_documents",
            "generate_response": "generate_response",
            END: END
        }
    )
    workflow.add_edge("update_preferences", END)
    workflow.add_edge("retrieve_documents", "generate_response")
    workflow.add_edge("generate_response", END)
    return workflow.compile(checkpointer=MemorySaver())

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="PhilanthroBot", page_icon="❤️", layout="wide")
    st.title("PhilanthroBot")
    st.markdown("#### *Discover NGOs that match your values*")
    st.markdown("_A Project by Tanish Maheshwari(22BIT0013) and Manya Dsouza(22BIT0100)_")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "retriever" not in st.session_state:
        with st.spinner("🔄 Loading NGO profiles..."):
            st.session_state.retriever = build_rag_pipeline()
            global retriever
            retriever = st.session_state.retriever
    if "app" not in st.session_state:
        st.session_state.app = build_graph()

    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me about NGOs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.spinner("Thinking..."):
            config = RunnableConfig(configurable={"thread_id": st.session_state.thread_id})
            events = st.session_state.app.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
            
            final_message = None
            for event in events:
                if "generate_response" in event:
                    final_message = event["generate_response"]["messages"][-1]
                elif "update_preferences" in event:
                    final_message = event["update_preferences"]["messages"][-1]
            
            if final_message:
                response = final_message.content
                st.session_state.messages.append({"role": "assistant", "content": response})

        # Display the new assistant message
        with st.chat_message("assistant"):
            st.markdown(response)


    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **How to use:**
        - Tell me your interests (e.g., "I care about education")
        - Ask for recommendations
        - Get details about specific NGOs
        """)
        
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

if __name__ == "__main__":
    main()
