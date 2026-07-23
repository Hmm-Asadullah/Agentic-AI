from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

st.header("Langchain Google Gemini Chatbot")
user_input = st.text_input("Ask me anything!")

if st.button("Ask"):
    if user_input:
        result = model.invoke(user_input)
        st.write(result.content[0]['text'])    
    else:
        st.write("Please enter a question to ask the chatbot.")