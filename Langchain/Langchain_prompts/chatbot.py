from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# while True:
#     user_input = input("You: ")
#     if user_input == "exit":
#         print("Exiting the chatbot. Goodbye!")
#         break
#     result = model.invoke(user_input)
#     print("Chatbot:", result.content[0]['text'])

chat_history = []
while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content[0]['text'])
    print("Chatbot:", result.content[0]['text'])

print(chat_history)