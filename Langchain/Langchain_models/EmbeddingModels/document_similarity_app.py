from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", dimension=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

user_query = "who has aggressive batting style?"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(user_query)

similarity_scores = cosine_similarity([query_embedding], doc_embeddings)
index, score = sorted(list(enumerate(similarity_scores[0])),key=lambda x:x[1])[-1]

print(user_query)
print(documents[index])
print("similarity score is:", score)