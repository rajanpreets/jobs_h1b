from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq  # Replaced OpenAI with Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get Groq API key from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=GROQ_API_KEY)

# Pull prompt from LangChain hub
prompt = hub.pull("rlm/rag-prompt")

# Create the generation chain
generate_chain = prompt | llm | StrOutputParser()
