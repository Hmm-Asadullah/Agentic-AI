from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template='Write a short summary on {text}',
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",    
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

loader = TextLoader("cricket.txt", encoding='utf-8')

docs = loader.load()

chain = prompt | model | parser
result = chain.invoke({"text": docs[0].page_content})
print(result)
