import os
import shutil
import uuid
from typing import List, Annotated, Dict
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# --- 1. SETUP AND CONFIGURATION ---
# Ensure you have set the GOOGLE_API_KEY environment variable
if 'GOOGLE_API_KEY' not in os.environ:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# Define directories
PROFILE_DIR = "./ngo_profiles"
DB_DIR = "./chroma_db"

# --- 2. PHASE 1: KNOWLEDGE BASE CONSTRUCTION ---

def create_dummy_ngo_profiles():
    """
    Creates dummy NGO profile files in Markdown format for the RAG pipeline.
    [cite_start]This simulates the data creation step[cite: 135].
    """
    if os.path.exists(PROFILE_DIR):
        shutil.rmtree(PROFILE_DIR)
    os.makedirs(PROFILE_DIR)

    ngo_1_content = """
# Global Reforestation Fund

## mission_statement
To combat deforestation by planting 1 million trees annually in critical ecosystems, empowering local communities through sustainable practices.

## cause_categories
- Environment
- Community Development
- Sustainability

## geographic_focus
- Amazon Rainforest, Brazil
- Southeast Asia
- California, USA

## impact_and_outcomes
Impact in 2023: Reforested 5,000 hectares, sequestering an estimated 10,000 tons of CO2. Supported 500 local jobs in sustainable agroforestry.

## financial_transparency_summary
We are committed to financial transparency. 85% of all donations go directly to our field programs. 10% is allocated to essential operational costs, and 5% is used for fundraising.

## vetting_and_accreditation
Accredited by Charity Navigator (4-Star Rating) and Guidestar (Platinum Seal of Transparency).
    """

    ngo_2_content = """
# Health for Horizons

## mission_statement
To provide critical medical supplies and healthcare training to remote communities in sub-Saharan Africa, focusing on maternal and child health.

## cause_categories
- Health
- Education
- Women and Children

## geographic_focus
- Sub-Saharan Africa

## impact_and_outcomes
Delivered 50,000 vaccination kits in the last year. Trained 1,000 local healthcare workers in basic pediatric care. Reduced maternal mortality by 15% in our target regions.

## financial_transparency_summary
90% of all funds are used for program services, including medical supply procurement and logistics. 10% covers administrative and fundraising costs.

## vetting_and_accreditation
Top-rated by GiveWell for our effective interventions and cost-efficiency.
    """

    with open(os.path.join(PROFILE_DIR, "ngo_1.md"), "w") as f:
        f.write(ngo_1_content)
    with open(os.path.join(PROFILE_DIR, "ngo_2.md"), "w") as f:
        f.write(ngo_2_content)
    print("Dummy NGO profiles created.")

def build_rag_pipeline():
    """
    [cite_start]Builds the RAG pipeline by loading, splitting, and indexing documents[cite: 132].
    Returns a retriever object.
    """
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    # [cite_start]Load documents from the directory [cite: 138]
    loader = DirectoryLoader(PROFILE_DIR, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()

    # [cite_start]Split documents into chunks using Markdown headers as separators [cite: 144, 153]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n## ", "\n###", "\n\n", "\n", " "]
    )
    chunks = text_splitter.split_documents(documents)

    # [cite_start]Create embeddings and store in Chroma vector store [cite: 158, 163, 165]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print("RAG pipeline built and indexed.")
    return vector_store.as_retriever()


# --- 3. PHASE 2: STATEFUL AGENT WITH LANGGRAPH ---

class UserPreferences(TypedDict):
    [cite_start]"""Structure to hold the user's learned preferences[cite: 179]."""
    causes: List[str]
    locations: List[str]

class AgentState(TypedDict):
    """
    [cite_start]Defines the main state for the entire graph[cite: 184].
    """
    messages: Annotated[list, add_messages]
    preferences: UserPreferences
    retrieved_docs: List[Document]
    latest_intent: str

# Define the LLM and the retriever
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
retriever = build_rag_pipeline()

# --- Graph Nodes ---

def classify_intent_node(state: AgentState):
    """
    [cite_start]Analyzes the latest user message to determine its purpose[cite: 193].
    """
    prompt = ChatPromptTemplate.from_template(
        """Given the user's latest message, classify the intent into one of the following categories:
'preference_update' - The user is stating their interests (e.g., "I care about the environment").
'recommendation_request' - The user is explicitly asking for NGO recommendations (e.g., "Can you suggest some charities?").
'question' - The user is asking a specific question about an NGO or a general question.
'greeting' - A simple greeting like 'hello' or 'hi'.
'goodbye' - The user wants to end the conversation.

Return only the single-word classification.

User Message: {user_message}"""
    )
    user_message = state["messages"][-1].content
    chain = prompt | llm
    intent = chain.invoke({"user_message": user_message}).content.strip()
    return {"latest_intent": intent}

def update_preferences_node(state: AgentState):
    """
    [cite_start]Parses the user's message to extract and store preferences[cite: 196].
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at extracting user preferences for philanthropic causes and locations.
Parse the user's message and extract any mentioned causes (e.g., 'environment', 'healthcare', 'education') and geographic locations (e.g., 'Africa', 'Brazil', 'Southeast Asia').

Return a JSON object with two keys: 'causes' and 'locations'. Each key should have a list of the extracted strings.
If no preferences are found, return empty lists."""),
        ("human", "{user_message}")
    ])
    parser = JsonOutputParser(pydantic_object=UserPreferences)
    chain = prompt | llm | parser

    user_message = state["messages"][-1].content
    extracted_prefs = chain.invoke({"user_message": user_message})

    # Update state with new preferences, keeping old ones
    current_prefs = state.get("preferences", {"causes": [], "locations": []})
    current_prefs["causes"].extend(extracted_prefs.get("causes", []))
    current_prefs["locations"].extend(extracted_prefs.get("locations", []))
    # Remove duplicates
    current_prefs["causes"] = list(set(current_prefs["causes"]))
    current_prefs["locations"] = list(set(current_prefs["locations"]))
    
    confirmation_message = "Thanks! I've updated your preferences."
    return {
        "preferences": current_prefs,
        "messages": [HumanMessage(content=confirmation_message, name="System")]
    }


def retrieve_documents_node(state: AgentState):
    """
    [cite_start]Constructs a query and retrieves relevant documents from the vector store[cite: 197].
    """
    user_message = state["messages"][-1].content
    prefs = state.get("preferences", {})
    
    # Construct a search query from the message and preferences
    query = f"{user_message}"
    if prefs.get("causes"):
        query += f" related to causes like {', '.join(prefs['causes'])}"
    if prefs.get("locations"):
        query += f" in locations like {', '.join(prefs['locations'])}"

    docs = retriever.invoke(query)
    return {"retrieved_docs": docs}


def generate_response_node(state: AgentState):
    """
    [cite_start]Generates a conversational, grounded answer based on retrieved documents[cite: 199].
    """
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
    question = state["messages"][-1].content

    response = chain.invoke({"context": context, "question": question})
    return {"messages": [response]}


# --- Conditional Edges ---

def route_after_classification(state: AgentState):
    """
    [cite_start]Decides the next step based on the classified intent[cite: 208].
    """
    intent = state["latest_intent"]
    if intent == "goodbye":
        return END
    if intent == "preference_update":
        return "update_preferences"
    if intent in ["question", "recommendation_request"]:
        return "retrieve_documents"
    # For greetings or other simple cases
    return "generate_response"

# --- Build the Graph ---

def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("update_preferences", update_preferences_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("generate_response", generate_response_node)

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # [cite_start]Add conditional edges [cite: 224]
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

    # Add normal edges
    workflow.add_edge("update_preferences", END) # Simple flow for now
    workflow.add_edge("retrieve_documents", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()


# --- Main Interaction Loop ---

if __name__ == "__main__":
    print("Setting up PhilanthroBot...")
    create_dummy_ngo_profiles()
    app = build_graph()

    print("\nPhilanthroBot is ready! How can I help you find an NGO to support?")
    
    # [cite_start]Use a unique ID for the conversation thread to maintain state [cite: 243, 244]
    thread_id = str(uuid.uuid4())
    config = RunnableConfig(configurable={"thread_id": thread_id})
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("PhilanthroBot: Goodbye!")
            break

        events = app.stream(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        
        # Stream the output
        final_message = None
        for event in events:
            if "generate_response" in event:
                final_message = event["generate_response"]["messages"][-1]
            elif "update_preferences" in event:
                 final_message = event["update_preferences"]["messages"][-1]


        if final_message:
             print(f"PhilanthroBot: {final_message.content}")