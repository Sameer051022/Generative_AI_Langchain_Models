import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Force CPU to avoid MPS memory issues
dtype = torch.float32

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={
        "dtype": dtype,
        "device_map": "cpu",  # force CPU instead of auto/MPS
        "load_in_8bit": False,
        "load_in_4bit": False
    }
)

model = ChatHuggingFace(llm=llm)

response = model.invoke(
    "Explain the theory of langchain in simple terms.",
    max_new_tokens=100,
    temperature=0.7
)

print(response.content)