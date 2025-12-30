# 3.EmbeddedModels/1_embedding_openai_query.py
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Assign to a variable (was missing before)
embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

response = embedding.embed_query("Explain the theory of langchain in simple terms.")

print(str(response))