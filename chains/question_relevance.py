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

# Define the structured output model
class QuestionRelevance(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# Initialize Groq LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)

# Create structured output pipeline
structured_output = llm.with_structured_output(QuestionRelevance)

# Define system prompt
system = """You are a grader assessing whether an answer addresses / resolves a question. 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

# Create the prompt template
relevance_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {solution}"),
    ]
)

# Define the runnable sequence
question_relevance: RunnableSequence = relevance_prompt | structured_output
