# Imports
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# --- PDF Text Extraction ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# --- Split text into chunks ---
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

# --- Create FAISS vector store ---
def get_vector_store(text_chunks, api_key):
    # CORRECTED: Changed the embedding model name to a valid one
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# --- Conversational chain ---
def get_conversational_chain(api_key):
    prompt_template = """
        Answer the question as detailed as possible from the provided context.
        If the answer is not in the context, just say "answer is not available in the context".

        Context: {context}
        Question: {question}

        Answer:
    """
    # Note: "gemini-2.5-flash" might not be a standard model name.
    # If it causes issues, try using "gemini-1.5-pro-latest" or another
    # valid model from the Google AI Studio list.
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- Handle user input ---
def user_input(user_question, api_key, pdf_docs, conversation_history):
    if not api_key or not pdf_docs:
        st.warning("Please upload a PDF and provide your API Key")
        return

    # Extract and chunk PDF text
    text_chunks = get_text_chunks(get_pdf_text(pdf_docs))
    # Note: The traceback mentioned `get_vectore_store` and `model_name` but your
    # code uses `get_vector_store` and `api_key`. The code below aligns with your provided snippet.
    vector_store = get_vector_store(text_chunks, api_key)
    
    # CORRECTED: Changed the embedding model name when loading the index
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response['output_text']

    # Update conversation history
    pdf_names = [pdf.name for pdf in pdf_docs]
    conversation_history.append((user_question, answer, "Google AI", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))

    # Display chat
    st.markdown(f"""
        <div style="background:#2b313e;padding:10px;border-radius:5px;color:white;"><b>You:</b> {user_question}</div>
        <div style="background:#475063;padding:10px;border-radius:5px;color:white;"><b>Bot:</b> {answer}</div>
    """, unsafe_allow_html=True)

    # Sidebar CSV download
    if conversation_history:
        df = pd.DataFrame(conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    api_key = st.sidebar.text_input("Enter your Google API Key")
    pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)

    user_question = st.text_input("Ask a question from your PDFs")
    if user_question:
        user_input(user_question, api_key, pdf_docs, st.session_state.conversation_history)

if __name__ == "__main__":
    main()