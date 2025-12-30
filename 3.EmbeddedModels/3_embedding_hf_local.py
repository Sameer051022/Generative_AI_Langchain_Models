from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Langchain is a framework for building applications with large language models.",
    "It provides tools for managing prompts, chains, agents, and memory.",
    "Langchain supports various LLM providers like OpenAI, HuggingFace, and more."
]

vector = embedding.embed_documents(documents)

print(str(vector))