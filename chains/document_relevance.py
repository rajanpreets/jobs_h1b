from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq  # Replaced OpenAI with Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get Groq API key from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)

class DocumentRelevance(BaseModel):
    binary_score: bool = Field(
        description="Answer is grounded in the documents, 'yes' or 'no'"
    )

structured_output = llm.with_structured_output(DocumentRelevance)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

relevance_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {solution}"),
    ]
)

document_relevance: RunnableSequence = relevance_prompt | structured_output
