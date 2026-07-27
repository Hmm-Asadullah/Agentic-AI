from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",    
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = 'Write a joke about {topic}',
    input_variables=['topic']
)

def word_count(text):
    return len(text.split())

parser = StrOutputParser()

sequential_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough(), 
    "word_count" : RunnableLambda(word_count)
})

final_chain = RunnableSequence(sequential_chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])
print(final_result)