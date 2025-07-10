import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import streamlit as st

# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Paths
DB_DIR = "db"
LOG_PATH = "data/logs.json"

# Setup Streamlit
st.set_page_config(page_title="PDF Q&A Agent", layout="wide")
tab1, tab2 = st.tabs(["📄 Chat with PDFs", "📊 Dashboard"])

with tab1:
    st.title("🧠 PDF Q&A Agent (Multi-PDF)")
    uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for file in uploaded_files:
            with open(file.name, "wb") as f:
                f.write(file.read())
            loader = PyPDFLoader(file.name)
            documents.extend(loader.load())

        # Split and embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(docs, embedding=embedding, persist_directory=DB_DIR)
        vectordb.persist()

        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        llm = ChatOpenAI(model="gpt-4", temperature=0)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        query = st.text_input("Ask something about the uploaded PDFs:")
        if query:
            result = qa_chain(query)
            st.subheader("Answer:")
            st.write(result["result"])

            with st.expander("📚 Source Chunks"):
                for doc in result["source_documents"]:
                    st.markdown(doc.page_content[:500] + "...---")

            # Log interaction
            log_entry = {
                "question": query,
                "answer": result["result"]
            }
            try:
                with open(LOG_PATH, "r") as f:
                    logs = json.load(f)
            except:
                logs = []

            logs.append(log_entry)
            with open(LOG_PATH, "w") as f:
                json.dump(logs, f, indent=2)

with tab2:
    st.title("📊 Q&A Log Dashboard")
    try:
        with open(LOG_PATH, "r") as f:
            logs = json.load(f)
        if logs:
            for i, entry in enumerate(reversed(logs[-20:]), 1):
                st.markdown(f"**Q{i}:** {entry['question']}")
                st.markdown(f"**A{i}:** {entry['answer']}")
                st.markdown("---")
        else:
            st.info("No logs yet. Ask some questions first!")
    except:
        st.error("Couldn't load logs.")