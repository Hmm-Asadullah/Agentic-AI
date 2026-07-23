from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", dimension=35)

doc = [
    "what is the capital of pakistan?",
    "Is it Islamabad or Karachi?",
    "What is the largest city in pakistan?",
    "is it lahore or karachi?"
]

result = embedding.embed_documents(doc)
print(str(result))