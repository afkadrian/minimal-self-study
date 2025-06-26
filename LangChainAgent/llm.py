import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaLLM
from langchain.llms import Ollama
# Initialize the LLM
gpt4 = ChatOpenAI(model="gpt-4o", api_key=os.getenv('OPENAI_API_KEY'))
llama = ChatOllama(model="llama3.1")
r1 = OllamaLLM(model="deepseek-r1:1.5b")