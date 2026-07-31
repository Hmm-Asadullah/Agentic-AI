from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

video_id = "vif8NQcjVf0"
try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])   
    transcript_list = transcript_list[800:]  

    transcript = " ".join(chunk.text for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.create_documents([transcript])
# print(chunks)
# print(f"Number of chunks: {len(chunks)}")
# print(chunks[0].page_content)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.index_to_docstore_id

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# response = retriever.invoke('What is AGI?')
# print(response)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# question = "explain about AI data centers in space"
# retrieved_docs = retriever.invoke(question)
# for i, doc in enumerate(retrieved_docs):
#     print(f"\n--- Retrieved Document no {i + 1} ---")
#     print(doc.page_content)

# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# context_text
# final_prompt = prompt.invoke({"context": context_text, "question": question})

# response = llm.invoke(final_prompt)
# print(response.content[0]["text"])

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parser = StrOutputParser()

parallel_chain = RunnableParallel({
   "context" : retriever | RunnableLambda(format_docs),
   "question" : RunnablePassthrough()
})

final_chain = parallel_chain | prompt | llm | parser

response = final_chain.invoke("summarize the video")
print(response)