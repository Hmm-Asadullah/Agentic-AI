from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("who is currently captain of pakistan cricket odi team?")
print(result.content)