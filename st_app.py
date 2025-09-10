import os
import shutil
from typing import List, Annotated

# --- FIX FOR ASYNCIO ERROR ---
import nest_asyncio
nest_asyncio.apply()
# -----------------------------

import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# --- 1. SETUP AND CONFIGURATION ---
# Set the page configuration for the Streamlit app
st.set_page_config(page_title="PhilanthroBot", layout="centered")

# Ensure you have set the GOOGLE_API_KEY as a Streamlit secret
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
if 'GOOGLE_API_KEY' not in os.environ:
    try:
        os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("Please set the GOOGLE_API_KEY environment variable or add it to your Streamlit secrets.")
        st.stop()


# Define directories
PROFILE_DIR = "./ngo_profiles"
DB_DIR = "./chroma_db"

# --- 2. PHASE 1: KNOWLEDGE BASE CONSTRUCTION (FOR PDFs) ---

@st.cache_resource
def build_rag_pipeline():
    """
    Builds the RAG pipeline. Using st.cache_resource ensures this heavy
    computation runs only once.
    Returns a retriever object.
    """
    if not os.path.exists(PROFILE_DIR) or not os.listdir(PROFILE_DIR):
        st.error(f"Profile directory '{PROFILE_DIR}' is empty or does not exist. Please add your NGO PDF files.")
        return None

    with st.spinner("Building knowledge base from NGO profiles... This may take a moment."):
        loader = DirectoryLoader(PROFILE_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        if not documents:
            st.error(f"No PDF documents were found in '{PROFILE_DIR}'.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)

        st.success(f"Knowledge base built successfully with {len(documents)} document(s)!")
        return vector_store.as_retriever()


# --- 3. PHASE 2: STATEFUL AGENT WITH LANGGRAPH ---

class UserPreferences(TypedDict):
    causes: List[str]
    locations: List[str]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    preferences: UserPreferences
    retrieved_docs: List[Document]
    latest_intent: str

# Define the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
retriever = build_rag_pipeline()


# Graph Nodes, Edges, and Compilation (No changes from the previous script)
def classify_intent_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        """Given the user's latest message, classify the intent into one of the following categories:
'preference_update', 'recommendation_request', 'question', 'greeting', 'goodbye'.
Return only the single-word classification. User Message: {user_message}"""
    )
    chain = prompt | llm
    intent = chain.invoke({"user_message": state["messages"][-1].content}).content.strip()
    return {"latest_intent": intent}

def update_preferences_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting philanthropic causes and locations from a message. Return a JSON object with 'causes' and 'locations' keys."),
        ("human", "{user_message}")
    ])
    parser = JsonOutputParser(pydantic_object=UserPreferences)
    chain = prompt | llm | parser
    extracted_prefs = chain.invoke({"user_message": state["messages"][-1].content})
    current_prefs = state.get("preferences", {"causes": [], "locations": []})
    current_prefs["causes"] = list(set(current_prefs["causes"] + extracted_prefs.get("causes", [])))
    current_prefs["locations"] = list(set(current_prefs["locations"] + extracted_prefs.get("locations", [])))
    return {
        "preferences": current_prefs,
        "messages": [AIMessage(content="Thanks! I've updated your preferences with this information.")]
    }

def retrieve_documents_node(state: AgentState):
    prefs = state.get("preferences", {})
    query = f"{state['messages'][-1].content}"
    if prefs.get("causes"): query += f" related to causes like {', '.join(prefs['causes'])}"
    if prefs.get("locations"): query += f" in locations like {', '.join(prefs['locations'])}"
    docs = retriever.invoke(query)
    return {"retrieved_docs": docs}

def generate_response_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        """You are PhilanthroBot, a helpful AI assistant for discovering trustworthy NGOs.
Answer the user's question based ONLY on the provided context. Be conversational and helpful.
If the context doesn't contain the answer, state that you don't have enough information.

**Context:**
{context}

**User Question:**
{question}"""
    )
    chain = prompt | llm
    context = "\n\n".join([doc.page_content for doc in state["retrieved_docs"]])
    response = chain.invoke({"context": context, "question": state["messages"][-1].content})
    return {"messages": [response]}

def route_after_classification(state: AgentState):
    intent = state["latest_intent"]
    if intent == "goodbye": return END
    if intent == "preference_update": return "update_preferences"
    if intent in ["question", "recommendation_request"]: return "retrieve_documents"
    return "generate_response"

@st.cache_resource
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("update_preferences", update_preferences_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent", route_after_classification,
        {"update_preferences": "update_preferences", "retrieve_documents": "retrieve_documents", "generate_response": "generate_response", END: END}
    )
    workflow.add_edge("update_preferences", END)
    workflow.add_edge("retrieve_documents", "generate_response")
    workflow.add_edge("generate_response", END)
    return workflow.compile()

# Build the LangGraph agent
if retriever:
    app = build_graph()
else:
    app = None

# --- 4. STREAMLIT UI ---

st.title("PhilanthroBot")
st.caption("Your AI-powered guide to discovering trustworthy Indian NGOs.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! How can I help you find an NGO to support today?")]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Accept user input
if prompt := st.chat_input("Ask about NGOs, causes, or locations..."):
    if not retriever or not app:
        st.error("The RAG pipeline is not available. Please check your PDF files and API key.")
        st.stop()

    # Add user message to session state and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)

    # Invoke the agent and stream the response
    config = RunnableConfig(configurable={"thread_id": "streamlit_session"})

    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            # The final response is the last event in the stream
            final_message = None
            events = app.stream({"messages": [HumanMessage(content=prompt)]}, config=config)
            for event in events:
                if "generate_response" in event:
                    final_message = event["generate_response"]["messages"][-1]
                elif "update_preferences" in event:
                    final_message = event["update_preferences"]["messages"][-1]

            if final_message:
                response_content = final_message.content
                st.markdown(response_content)
                st.session_state.messages.append(AIMessage(content=response_content))
            else:
                st.markdown("I'm not sure how to respond to that. Could you try rephrasing?")

