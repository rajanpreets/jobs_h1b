import streamlit as st
from dotenv import load_dotenv
import os
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from chains.document_relevance import document_relevance
from chains.evaluate import evaluate_docs
from chains.generate_answer import generate_chain
from chains.question_relevance import question_relevance
from state import GraphState
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer

# ✅ Ensure set_page_config is the first Streamlit command
st.set_page_config(page_title="Advanced RAG with Pinecone", page_icon="🔍")

# Load environment variables
load_dotenv()

# API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API keys
if not all([PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME, GROQ_API_KEY]):
    st.error("Missing API keys. Please check your .env file.")
    st.stop()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME in pc.list_indexes():
    index = pc.Index(INDEX_NAME)
else:
    st.error(f"Error: Index '{INDEX_NAME}' does not exist. Please create it in Pinecone.")
    st.stop()

# ✅ Using SentenceTransformer directly to avoid HuggingFaceEmbeddings issue
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# Chat model
chat_model = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)

def setup_ui():
    """Sets up the Streamlit UI components."""
    st.title("🔍 Advanced RAG with Pinecone")
    st.write("Upload a text file and ask questions to get intelligent answers.")

def handle_file_upload(user_file):
    """Handles file upload and processes embeddings."""
    if user_file is None:
        return None

    documents = [user_file.read().decode()]
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=30)
    doc_splits = splitter.create_documents(documents)

    vectors = []
    for i, doc in enumerate(doc_splits):
        vector = embedding_model.encode(doc.page_content).tolist()
        vectors.append({
            "id": f"doc_{i}",
            "values": vector,
            "metadata": {"text": doc.page_content}
        })
    
    index.upsert(vectors=vectors, namespace="documents")
    st.success("Embeddings successfully inserted into Pinecone!")
    return index

def retrieve(state: GraphState):
    """Retrieves the most relevant documents from Pinecone."""
    question = state["question"]
    question_embedding = embedding_model.encode(question).tolist()

    response = index.query(
        namespace="documents",
        vector=question_embedding,
        top_k=5,
        include_values=True,
        include_metadata=True
    )

    documents = [doc["metadata"]["text"] for doc in response["matches"]]
    return {"documents": documents, "question": question}

def evaluate(state: GraphState):
    """Evaluates the relevance of retrieved documents."""
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []

    for document in documents:
        response = evaluate_docs.invoke({"question": question, "document": document})
        if response.score.lower() == "yes":
            filtered_docs.append(document)

    return {"documents": filtered_docs, "question": question}

def generate_answer(state: GraphState):
    """Generates an answer based on retrieved documents."""
    question = state["question"]
    documents = state["documents"]
    solution = chat_model.invoke({"context": documents, "question": question})

    return {"documents": documents, "question": question, "solution": solution}

def create_graph():
    """Creates the RAG workflow."""
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
    """Handles user questions and runs the pipeline."""
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
    """Main function to run the Streamlit app."""
    setup_ui()
    user_file = st.file_uploader("Upload your Text file here", type=".txt")

    if user_file:
        handle_file_upload(user_file)
        ask_question(user_file)

if __name__ == "__main__":
    main()
