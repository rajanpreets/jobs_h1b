from typing import List, TypedDict

class GraphState(TypedDict):
    
    question: str
    solution: str
    online_search: bool
    documents: List[str]