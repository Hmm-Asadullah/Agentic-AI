from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4-Pro",    
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative', 'mixed'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive, negative or mixed \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

prompt4 = PromptTemplate(
    template='Write an appropriate response to this mixed feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    (lambda x:x.sentiment == 'mixed', prompt4 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

text = """I've had a fantastic experience with CGModels. From the very beginning, the team has been friendly, supportive and professional. They take the time to answer my questions, provide helpful guidance and genuinely want their models to succeed. Communication has been excellent, and I've always felt encouraged and valued. As someone still growing in the industry, having an agency that is responsive and invested in my progress has made a huge difference. I would highly recommend CGModels to anyone looking for an agency that is trustworthy, supportive and committed to helping its talent achieve their goals. Thank you to the whole team for everything you do!
"""

print(chain.invoke({'feedback': text}))

chain.get_graph().print_ascii()