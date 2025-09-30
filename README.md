# 📄 Contextual Query Engine

A Retrieval-Augmented Generation (RAG) powered chatbot that allows users to upload a PDF file and ask questions about its content. The chatbot responds only from the PDF to ensure accuracy and avoid hallucinations.  

---

## 🚀 Features
- Upload any PDF and query its content in natural language.
- Provides contextual answers grounded in the uploaded document.
- Simple Streamlit-based web interface for easy interaction.
- Uses ChromaDB for vector storage and retrieval.
- Powered by LangChain for pipeline orchestration and Gemini API for response generation.

---

## 🛠️ Tech Stack
- Python
- Streamlit – Web app interface
- LangChain – Framework for RAG pipeline
- ChromaDB – Vector database for embeddings
- Gemini API – Language model for generating responses

---

## ⚙️ How It Works
1. PDF Upload – User uploads a PDF file.  
2. Text Chunking – The PDF is split into smaller text chunks.  
3. Vectorization – Each chunk is converted into embeddings.  
4. Storage – Embeddings are stored in **ChromaDB**.  
5. Query Handling – User query is converted into a vector.  
6. Retrieval – Relevant chunks are fetched from the database.  
7. Response Generation – Retrieved chunks + query are passed to Gemini API, which generates an accurate answer.  

---

## 📂 Project Structure
rag-pdf-chatbot/
│── app.py # Main Streamlit app
│── requirements.txt # Dependencies
│── utils.py # Helper functions (chunking, embeddings, etc.)
│── README.md # Project documentation

---

## ▶️ Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/shahu285/contextualqengine.git
   cd contextualqengine
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Run the Streamlit app:
   ```bash 
   streamlit run app.py
   
4. Upload a PDF and start chatting with it!

