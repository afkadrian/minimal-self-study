import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import anthropic

from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM
gpt4 = ChatOpenAI(model="gpt-4o", api_key=os.getenv('OPENAI_API_KEY'))
claude = ChatAnthropic(model="claude-3-5-sonnet-latest", api_key=os.getenv('ANTHROPIC_API_KEY'))

llama = ChatOllama(model="llama3.1")
r1 = OllamaLLM(model="deepseek-r1:1.5b")
llava = OllamaLLM(model="llava")

gemini = ChatGoogleGenerativeAI(model="gemini-2.5-pro")