from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",    
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

parser = StrOutputParser()

sequential_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough(), 
    "summary" : RunnableSequence(prompt2,model,parser)
})
final_chain = RunnableSequence(sequential_chain, parallel_chain)
result = final_chain.invoke({'topic':'cricket'})
print("joke: ", result['joke'])
print("explanation: ", result['summary'])   