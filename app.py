
# Imports
from dotenv import load_dotenv
import streamlit as st

# Langchain imports
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

# Chains imports
from chains.document_relevance import document_relevance
from chains.evaluate import evaluate_docs
from chains.generate_answer import generate_chain
from chains.question_relevance import question_relevance

# Graph imports
from state import GraphState
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv()



# Constants for UI
PAGE_TITLE = "Advanced RAG"
PAGE_ICON = "🔎"
FILE_UPLOAD_PROMPT = "Upload your Text file here"
FILE_UPLOAD_TYPE = ".txt"

# Setup the UI configuration
def setup_ui():
    """
    Configures the Streamlit app page settings and displays the main title.
    """
    st.set_page_config(
        page_title=PAGE_TITLE, 
        page_icon=PAGE_ICON
    )
    st.header("",divider='blue')
    st.title(f"{PAGE_ICON} :blue[_{PAGE_TITLE}_] | Text File Search")
    st.header("",divider='blue')

def ask_question(user_file):
    """
    Allows the user to ask a question about the uploaded file and displays the result.
    """    
    if user_file is None:
        return
    st.divider()
    question = st.text_input('Please enter your question:', placeholder = "Which year was Marty transported to?", disabled=not user_file)

    if question:
        with st.spinner('Please wait...'):
            graph = create_graph()
            result = graph.invoke(input={"question": question})
            st.info(result['solution'])
            st.divider()


def handle_file_upload(user_file):
    """
    Handles the uploaded text file, splits it into chunks, and inserts embeddings into a vector database.
    """
    if user_file is None:
        return
    documents = [user_file.read().decode()]
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=30)
    doc_splits = splitter.create_documents(documents)

    # Create the embeddings for the text file
    embedding_function = OpenAIEmbeddings()
    chroma_db = Chroma.from_documents(
        documents=doc_splits, 
        collection_name= 'rag-chroma', 
        embedding = embedding_function,
        persist_directory="./.chroma"
    )
    

    st.success("Text file embeddings were successfully inserted into VectorDB")

    return chroma_db.as_retriever()


def create_graph(): 
    """
    Creates and configures the state graph for handling queries and generating answers.
    """    
    workflow = StateGraph(GraphState)
    
    # Adding nodes
    workflow.add_node("Retrieve Documents", retrieve)
    workflow.add_node("Grade Documents", evaluate)
    workflow.add_node("Generate Answer", generate_answer)
    workflow.add_node("Search Online", search_online)

    # Setting the entry point and edges
    workflow.set_entry_point("Retrieve Documents")
    workflow.add_edge("Retrieve Documents", "Grade Documents")
    workflow.add_conditional_edges(
        "Grade Documents",
        any_doc_irrelevant,
        {
            "Search Online": "Search Online",
            "Generate Answer": "Generate Answer",
        },
    )

    workflow.add_conditional_edges(
        "Generate Answer",
        hallucinations,
        {
            "Hallucinations detected": "Generate Answer",
            "Answers Question": END,
            "Question not addressed": "Search Online",
        },
    )
    workflow.add_edge("Search Online", "Generate Answer")
    workflow.add_edge("Generate Answer", END)

    graph = workflow.compile()
    
    # Generate a graphical representation of the workflow
    graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
    
    return graph

# Other utility functions for the graph nodes
def any_doc_irrelevant(state):
    """
    Determines whether any document is irrelevant, triggering online search.
    """   
    if state["online_search"]:
        return "Search Online"
    else:
        return "Generate Answer"


def hallucinations(state: GraphState):
    """
    Checks for hallucinations in the generated answers.
    """    
    question = state["question"]
    documents = state["documents"]
    solution = state["solution"]

    score = document_relevance.invoke(
        {"documents": documents, "solution": solution}
    )

    if score.binary_score:
        score = question_relevance.invoke({"question": question, "solution": solution})
        if score.binary_score:
            return "Answers Question"
        else:
            return "Question not addressed"
    else:
        return "Hallucinations detected"


def retrieve(state: GraphState):
    """
    Retrieves documents relevant to the user's question.
    """    
    question = state["question"]
    documents = documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def evaluate(state: GraphState):
    """
    Filters documents based on their relevance to the question.
    """
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    online_search = False
    for document in documents:
        response = evaluate_docs.invoke({"question": question, "document": document.page_content})
        result = response.score
        if result.lower() == "yes":
            filtered_docs.append(document)
        else:
            online_search = True
            
    return {"documents": filtered_docs, "question": question, "online_search": online_search}


def generate_answer(state: GraphState):
    """
    Generates an answer based on the retrieved documents.
    """    
    question = state["question"]
    documents = state["documents"]

    solution = generate_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "solution": solution}



def search_online(state: GraphState):
    """
    Searches online for additional context if the answer cannot be generated locally.
    """    
    question = state["question"]
    documents = state["documents"]
    tavily_client = TavilySearchResults(k=2)
    response = tavily_client.invoke({"query": question})
    results = "\n".join([element["content"] for element in response])
    results = Document(page_content=results)
    if documents is not None:
        documents.append(results)
    else:
        documents = [results]
    return {"documents": documents, "question": question}

# Main function to orchestrate the app
def main():
    # Setup the UI
    setup_ui()
    
    # File uploader
    user_file = st.file_uploader(FILE_UPLOAD_PROMPT, type=FILE_UPLOAD_TYPE)
    
    # Handle the file upload
    global retriever
    retriever = handle_file_upload(user_file)
    
    # Ask the question
    ask_question(user_file)

if __name__ == "__main__":
    main()
