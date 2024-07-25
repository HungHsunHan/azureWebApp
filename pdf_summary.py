import os

import PyPDF2
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()

# Get the API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")


# 1. Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


# 2. Split text into chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# 3 & 4. Create embeddings and store in vector database
def create_vector_db(chunks):
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings)
    return vector_db


# 5. Implement RAG system
def setup_rag(vector_db):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_db.as_retriever()
    )
    return qa_chain


# 6. Simple application
def main():
    pdf_path = "/Users/hanhongxun/Desktop/2023.pdf"

    # Process the PDF
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    vector_db = create_vector_db(chunks)
    qa_chain = setup_rag(vector_db)

    # Interactive Q&A loop
    while True:
        query = input("Ask a question about the PDF (or type 'quit' to exit): ")
        if query.lower() == "quit":
            break

        response = qa_chain.run(query)
        print(f"Answer: {response}\n")


if __name__ == "__main__":
    main()
