from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

from dotenv import load_dotenv

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

llm_with_tools = llm.bind_tools([multiply])

query = HumanMessage("please multiply 50 with 15")
messages = [query]

response = llm_with_tools.invoke(messages)
print(response)
messages.append(response)

# print(response.tool_calls[0])
# print(response.tool_calls[0]['args'])

# print(multiply.invoke(response.tool_calls[0]['args']))

tool_response = multiply.invoke(response.tool_calls[0])
messages.append(tool_response)

final_response = llm_with_tools.invoke(messages)
print(final_response)
print(final_response.content[0]['text'])