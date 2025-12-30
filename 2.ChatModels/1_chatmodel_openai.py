from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model='gpt-4', temperature=0.7)

response = chat_model.invoke("Explain the concept of Langchain in simple terms.")

print(response.content) #.content to see the text response only