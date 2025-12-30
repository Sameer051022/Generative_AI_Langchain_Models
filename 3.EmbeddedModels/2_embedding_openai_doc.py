from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = [
    "Langchain is a framework for building applications with large language models.",
    "It provides tools for managing prompts, chains, agents, and memory.",
    "Langchain supports various LLM providers like OpenAI, HuggingFace, and more."
]

response = embedding.embed_documents("Explain the theory of langchain in simple terms.")

print(str(response))

