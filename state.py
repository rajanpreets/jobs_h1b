from typing import List, TypedDict

class GraphState(TypedDict):
    question: str
    solution: str
    documents: List[str]  # Stores retrieved context from Pinecone

    
