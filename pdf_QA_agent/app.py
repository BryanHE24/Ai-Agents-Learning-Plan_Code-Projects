import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Step 1: Load the PDF
pdf_path = "Document.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Step 3: Create vector DB
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory="db")
vectordb.persist()

# Step 4: Build retriever and agent
retriever = vectordb.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Step 5: Ask your question
query = input("Ask a question about the PDF: ")
result = qa_chain(query)

print("\nAnswer:\n", result['result'])
print("\nSource Chunks:\n")
for doc in result["source_documents"]:
    print(doc.page_content[:200], "\n---")
