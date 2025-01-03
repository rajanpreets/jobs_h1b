from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0)

class EvaluateDocs(BaseModel):

    score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_output = llm.with_structured_output(EvaluateDocs)

system = """You are an expert evaluator who professionally assesses whether the documents retrieved from the vector database can answer the user's query. \n 
    Your task is to determine whether the content of the documents provided below is sufficient to answer the user's query, grade it as relevant. \n
    Instructions: 
    1. Carefully review the content of the documents and evaluate whether they are appropriate for answering the user's query.
    2. When evaluating the sufficiency of the documents, consider the following factors:
    - a: Assess whether the main topics or aspects of the documents are relevant to answering the user's query.
    - b: The depth and specificity of the information provided in the documents to answer the user's query.
    - c: Complementary or overlapping information within the documents.
    - d: Compare the user's query directly with the main topics and key points of the documents to ensure they are closely aligned.
    3. Provide a binary assessment of whether the combined information from the documents is sufficient to answer the user's query.
        - yes: The documents are relevant to the user's query and provide enough information to answer it satisfactorily
        - no: The documents do not provide enough relevant information to adequately answer the user's query.
    4. Remember to assess the document's relevance strictly in the context of the user's specific query.
    
    Please provide your evaluation of whether the retrieved documents are sufficient to answer the user's query, using 'yes' or 'no'. """

evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

evaluate_docs = evaluate_prompt | structured_output