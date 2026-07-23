from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", dimension=5)

result = embedding.embed_query("what is capital of pakistan?")
print(result)