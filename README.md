
---
# 🔎 Advanced RAG with LangGraph

**Advanced-RAG-LangGraph** is an advanced web application that implements a powerful Retrieval-Augmented Generation (RAG) pipeline using LangGraph. It combines the flexibility of **Streamlit** for an interactive user interface, **ChromaDB** as a vector database for efficient document retrieval, and **Tavily** for online search capabilities. This repository extends the traditional RAG framework with additional flows for robust error handling and reducing hallucinations in generated answers.

---

## Features

- **Document Upload and Embedding**:
  - Upload text files directly through the web interface.
  - Automatically converts the content into vector embeddings and stores them in **ChromaDB** for fast and accurate retrieval.

- **Interactive Q&A**:
  - Ask questions about the uploaded document's content.
  - If the answer exists in the document, it is retrieved from the vector store.
  - The application queries online sources via **Tavily**if the answer is not found.

- **Enhanced Retrieval-Augmented Generation (RAG)**:
  - Uses **LangGraph**, offering advanced features beyond traditional RAG implementations.
  - Employs conditional workflows to handle edge cases:
    - Fall back to alternative sources if the answers generated are inadequate.
    - Detect and address hallucinations to ensure reliable responses.

- **Tracing and Debugging**:
  - Integrates with **LangSmith** for tracing and debugging the LangGraph workflows.

---

## How It Works

1. **Upload a Document**:
   - Users can upload text files to the web application.
   - The content is embedded into vector representations and stored in **ChromaDB**.
   


2. **Ask Questions**:
   - Users type their questions in the query box.
   - The system retrieves the most relevant answers from the vector database or queries online using Tavily.
   


3. **Advanced Handling**:
   - Implements fallback logic for unanswered questions.
   - Detects and mitigates hallucinations in generated answers.
   


4. **Tracing with LangSmith**:
   - Visualize and debug LangGraph workflows using LangSmith.
   


---

## Prerequisites

- Python 3.9 or higher
- API key for Tavily (for online search functionality)
- Required Python libraries (see below)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/chitralputhran/Advanced-RAG-LangGraph.git
   cd Advanced-RAG-LangGraph

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the application:
   ```bash
   streamlit run app.py

---
## Usage 
1. Open the application in your browser (Streamlit URL provided in the terminal).
2. Upload your text document in the designated section.
3. Enter your query in the input box.
4. View the answer retrieved from the document or online sources.

---
## Technologies Used

- **Streamlit:** Web application framework for building interactive UIs.
- **LangGraph:** Advanced library for RAG workflows.
- **ChromaDB:** Vector database for embedding storage and retrieval.
- **Tavily:** Online search integration for fallback queries.
- **LangSmith:** Tracing and debugging tool for LangGraph workflows.
---
## Workflow Diagram

---
## Screenshots

1. Document Upload Section

2. Q&A Interaction

3. LangSmith Tracing

---
## Future Enhancements

- Support more document types (e.g., PDFs, Word documents).
- Extend vector database options for improved scalability.
- Further enhancements to UI design for accessibility and usability.
- Integration with additional online search APIs.
---
## Contribution

Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request if you'd like to contribute to this project.

--



