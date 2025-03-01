from dotenv import load_dotenv
import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from chains.document_relevance import document_relevance
from chains.evaluate import evaluate_docs
from chains.generate_answer import generate_chain
from chains.question_relevance import question_relevance
from state import GraphState
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv()

PAGE_TITLE = "Advanced RAG with Pinecone"
PAGE_ICON = "🔍"
FILE_UPLOAD_PROMPT = "Upload your Text file here"
FILE_UPLOAD_TYPE = ".txt"

# Read environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API keys
if not all([PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME, GROQ_API_KEY]):
    st.error("Missing API keys. Please check your .env file.")

# Setting up Pinecone Index
pinecone_db = Pinecone(index_name=INDEX_NAME, api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# MiniLLM for embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Groq Chat Model
chat_model = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)

def setup_ui():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")

def handle_file_upload(user_file):
    if user_file is None:
        return None
    documents = [user_file.read().decode()]
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=30)
    doc_splits = splitter.create_documents(documents)
    pinecone_db.from_documents(doc_splits, embedding_function)
    st.success("Embeddings successfully inserted into Pinecone!")
    return pinecone_db.as_retriever()

def retrieve(state: GraphState):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def evaluate(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for document in documents:
        response = evaluate_docs.invoke({"question": question, "document": document.page_content})
        if response.score.lower() == "yes":
            filtered_docs.append(document)
    return {"documents": filtered_docs, "question": question}

def generate_answer(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    solution = chat_model.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "solution": solution}

def create_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("Retrieve Documents", retrieve)
    workflow.add_node("Grade Documents", evaluate)
    workflow.add_node("Generate Answer", generate_answer)
    
    workflow.set_entry_point("Retrieve Documents")
    workflow.add_edge("Retrieve Documents", "Grade Documents")
    workflow.add_edge("Grade Documents", "Generate Answer")
    workflow.add_edge("Generate Answer", END)
    
    return workflow.compile()

def ask_question(user_file):
    if user_file is None:
        return
    st.divider()
    question = st.text_input('Enter your question:', placeholder="Example: What year was X event?", disabled=not user_file)
    if question:
        with st.spinner('Processing...'):
            graph = create_graph()
            result = graph.invoke(input={"question": question})
            st.info(result['solution'])
            st.divider()

def main():
    setup_ui()
    user_file = st.file_uploader(FILE_UPLOAD_PROMPT, type=FILE_UPLOAD_TYPE)
    global retriever
    retriever = handle_file_upload(user_file)
    ask_question(user_file)

if __name__ == "__main__":
    main()
