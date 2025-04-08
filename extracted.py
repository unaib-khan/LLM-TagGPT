import os
import io
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from PyPDF2 import PdfReader
from PIL import Image, ImageStat
import pytesseract
import pdfplumber
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_ollama import OllamaLLM
import google.generativeai as genai

# ==========================
# Setup & Configuration
# ==========================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("GOOGLE_API_KEY is missing! Add it to your .env file.")
    st.stop()

logging.basicConfig(level=logging.INFO)

os.makedirs("uploaded_files", exist_ok=True)
processed_files = set()


# ==========================
# Utility Functions
# ==========================
def compute_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(8192):
            file_hash.update(chunk)
        return file_hash.hexdigest()


def is_blank_image(image):
    stat = ImageStat.Stat(image)
    return sum(stat.stddev) < 10


def extract_text_with_pdfplumber(pdf_path):
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            extracted_text += f"\n--- Page {page_number} ---\n{text}"
            for img in page.images:
                image_stream = img.get("stream")
                if image_stream:
                    image_bytes = io.BytesIO(image_stream.read())
                    try:
                        with Image.open(image_bytes) as image:
                            if not is_blank_image(image):
                                ocr_text = pytesseract.image_to_string(image).strip()
                                extracted_text += f"\n[Image OCR - Page {page_number}]\n{ocr_text}"
                    except Exception as e:
                        logging.error(f"Error processing image on Page {page_number}: {e}")
    return extracted_text.strip()


def extract_text_from_pdf(pdf_path, enable_ocr=False, ocr_tool="fitz"):
    combined_text = ""
    
    if ocr_tool == "pdfplumber":
        return extract_text_with_pdfplumber(pdf_path)
    
    elif ocr_tool == "fitz":
        # Using PyMuPDF (fitz) for text and image OCR
        doc = fitz.open(pdf_path)
        for page_number in range(len(doc)):
            page = doc[page_number]
            text = page.get_text()  # Extract text from the page
            combined_text += f"\n--- Page {page_number + 1} ---\n{text}"
            
            if enable_ocr:
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]  # Image XREF number
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    if not is_blank_image(image):
                        ocr_text = pytesseract.image_to_string(image).strip()
                        combined_text += f"\n[Image OCR - Page {page_number + 1}, Image {img_index + 1}]\n{ocr_text}"
        return combined_text.strip()
    
    else:
        # Default to PyPDF2
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            combined_text += f"\n--- Page {page_number} ---\n{text}"
        return combined_text.strip()


def process_pdf(uploaded_file, tag, enable_ocr, ocr_tool):
    tag_folder = os.path.join("uploaded_files", tag)
    os.makedirs(tag_folder, exist_ok=True)

    file_path = os.path.join(tag_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    file_hash = compute_hash(file_path)
    if file_hash in processed_files:
        st.warning(f"Duplicate file detected: {uploaded_file.name}")
        return None
    processed_files.add(file_hash)

    text = extract_text_from_pdf(file_path, enable_ocr=enable_ocr, ocr_tool=ocr_tool)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    return {"name": uploaded_file.name, "path": file_path, "vector_store": vector_store, "text": text}


def classify_question(question):
    quantitative_keywords = ["how many", "amount", "percentage", "number", "calculate", "ratio", "figure"]
    return "quantitative" if any(kw in question.lower() for kw in quantitative_keywords) else "qualitative"


def ask_question_with_model(question, context, model_choice, vector_store):
    if model_choice == "Mistral":
        llm = OllamaLLM(model="mistral:latest")
        prompt = f"Context:\n{context}\nQuestion:\n{question}\nAnswer:"
        return llm.invoke(prompt)
    else:
        docs = vector_store.similarity_search(question)
        query_type = classify_question(question)
        prompt_template = {
            "quantitative": """
            You are a quantitative expert. Provide clear numeric-based answers.
            Context:\n{context}\nQuestion:\n{question}\nAnswer:
            """,
            "qualitative": """
            You are a qualitative insights assistant. Provide thorough answers.
            Context:\n{context}\nQuestion:\n{question}\nAnswer:
            """,
        }[query_type]
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return response["output_text"]


# ==========================
# Streamlit Application
# ==========================
def main():
    st.set_page_config(page_title="RAG Application")
    st.header("Tag RAG Application ")

    if "processed_pdfs" not in st.session_state:
        st.session_state.processed_pdfs = {}

    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    tag = st.sidebar.text_input("Enter a tag:")
    enable_ocr = st.sidebar.checkbox("Enable OCR")
    ocr_tool = st.sidebar.radio("Choose OCR Tool", ["fitz", "pdfplumber"])
    model_choice = st.sidebar.selectbox("Choose LLM Model", ["Gemini", "Mistral"])
    process_button = st.sidebar.button("Process Files")

    if process_button and uploaded_files and tag:
        with st.spinner("Processing files..."):
            with ThreadPoolExecutor() as executor:
                for uploaded_file in uploaded_files:
                    result = executor.submit(process_pdf, uploaded_file, tag, enable_ocr, ocr_tool).result()
                    if result:
                        st.session_state.processed_pdfs.setdefault(tag, []).append(result)
        st.success("Files processed successfully!")

    if st.session_state.processed_pdfs:
        tag_choice = st.selectbox("Select a Tag", list(st.session_state.processed_pdfs.keys()))
        file_choice = st.selectbox("Select a File", [f["name"] for f in st.session_state.processed_pdfs[tag_choice]])
        question = st.text_input("Ask a question:")

        if question:
            file = next(f for f in st.session_state.processed_pdfs[tag_choice] if f["name"] == file_choice)
            response = ask_question_with_model(question, file["text"], model_choice, file["vector_store"])
            st.subheader("Answer:")
            st.write(response)
    st.markdown("*Made with love by unaib*")


if __name__ == "__main__":
    main()
