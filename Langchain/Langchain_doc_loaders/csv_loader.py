from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='students_scores_dataset.csv')

docs = loader.load()

print(len(docs))
print(docs[1])