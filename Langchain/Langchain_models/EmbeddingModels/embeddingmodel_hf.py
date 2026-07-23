from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

doc = [
    "what is the capital of pakistan?",
    "Is it Islamabad or Karachi?",
    "What is the largest city in pakistan?",
    "is it lahore or karachi?"
]

result = embeddings.embed_documents(doc)
print(result)
