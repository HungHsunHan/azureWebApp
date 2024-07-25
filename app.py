import os

import docx
import gradio as gr
import PyPDF2
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Set your OpenAI API key
load_dotenv()

# Get the API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")


def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_db(chunks):
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings)
    return vector_db


def setup_rag(vector_db):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_db.as_retriever()
    )
    return qa_chain


def process_file(file):
    if file.name.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        text = extract_text_from_docx(file)
    else:
        return None, "Unsupported file format. Please upload a PDF or DOCX file."

    chunks = split_text(text)
    vector_db = create_vector_db(chunks)
    qa_chain = setup_rag(vector_db)
    return qa_chain, f"Processed {file.name} successfully. You can now ask questions."


qa_chain = None


def answer_question(file, question):
    global qa_chain
    if qa_chain is None:
        qa_chain, message = process_file(file)
        if qa_chain is None:
            return message

    response = qa_chain.run(question)
    return response


def create_app():
    iface = gr.Interface(
        fn=answer_question,
        inputs=[
            gr.File(label="Upload PDF or DOCX file"),
            gr.Textbox(label="Ask a question about the document"),
        ],
        outputs=gr.Textbox(label="Answer"),
        title="Document Q&A System",
        description="Upload a PDF or DOCX file and ask questions about its content.",
    )
    return iface


app = create_app()

if __name__ == "__main__":
    app.launch()
