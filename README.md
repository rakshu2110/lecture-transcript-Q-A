# repo1

# 📚 Enhanced RAG-based Lecture Transcript Q&A System
This project is an AI-powered Question & Answer system that helps users extract knowledge from lecture transcripts in PDF format. Using advanced Retrieval-Augmented Generation (RAG) techniques, the system retrieves relevant content from the transcripts and generates accurate answers with proper citations.

## 🚀 Features
🔍 Automatic PDF Text Extraction

🧼 Transcript Cleaning & Preprocessing

🔁 Context Retrieval using Sentence Transformers + FAISS

✍️ Answer Generation using T5 Transformer

📝 Answer Formatting with Proper Capitalization and Citations

🌐 User Interface with Streamlit

## 🧠 Technologies Used
Python

PyPDF2 (PDF text extraction)

FAISS (Similarity search)

SentenceTransformers (Embedding)

T5 (Text-to-Text Transfer Transformer for answer generation)

Streamlit (Web UI)

## 📂 Project Structure
bash
Copy
Edit

## 📦 project-root/
├── transcripts/                 # Raw lecture PDFs
├── data/                        # Cleaned JSON output
├── app.py                       # Streamlit app
├── rag_module.py                # Core RAG logic
├── utils/
│   ├── cleaner.py               # Transcript cleaning logic
│   ├── extractor.py             # PDF text extractor
│   └── preprocessor.py          # Preprocessing to JSON
└── requirements.txt             # Required Python packages

## ⚙️ Setup Instructions
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/rag-transcript-qa.git
cd rag-transcript-qa
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Add your lecture PDFs Place them in the transcripts/ folder.

Run preprocessing

bash
Copy
Edit
python preprocess_all.py
Launch the web app

bash
Copy
Edit
streamlit run app.py

## 💡 How It Works
PDFs are scanned, and raw text is extracted.

Text is cleaned to remove artifacts and unnecessary formatting.

Text is converted to a searchable format using embeddings and indexed with FAISS.

User submits a query — the most relevant context is retrieved.

T5 model generates the answer, and citations are attached.

## 📌 Example Output
Question: What is a symmetric matrix?
Answer: Symmetric matrices are one of the most important classes of matrices. A Toeplitz matrix has constant diagonals. While it may have constant diagonals, a Toeplitz matrix does not necessarily need to be symmetric.
Sources:

“Gets on the tape. So tell me its properties. Symmetric.”

“That’s the way to visualize it. And let me use a capital T for transpose.”

“We’ll use MATLAB language because that’s a good common language...”

## 👩‍💻 Author
Rakshitha S
Intern @ Rubixe
Passionate about Data Analytics and Building AI-Powered Tools

