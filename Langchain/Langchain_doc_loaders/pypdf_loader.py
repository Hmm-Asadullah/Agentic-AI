from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('NOTES WEEK Five.pdf')

docs = loader.load()

print(len(docs))

print(docs[4].page_content)
print(docs[4].metadata)