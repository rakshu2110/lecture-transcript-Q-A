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



## 📦 project-root
📁 Desktop/<br>
└── 📁 rubixe/<br>
    ├── transcripts/              # Raw lecture PDFs<br>
    ├── app.py                    # Streamlit app<br>
    ├── new.py                    # Core RAG logic<br>
    ├── embeddings/              <br>
    │   ├── faiss_index.bin       # FAISS index file<br>
    │   └── passages.json         # JSON with split passages<br>
    ├── transcript.txt            # Cleaned transcript file<br>
    ├── requirements.txt          # Required Python packages<br>
    └── utils/<br>
        ├── cleaner.py            # Transcript cleaning logic<br>
        ├── extractor.py          # PDF text extractor<br>
        └── preprocessor.py       # Preprocessing to JSON



## ⚙️ Setup Instructions

Open Terminal or PowerShell
(Make sure you're in the virtual environment.)

Navigate to the project folder:

bash
Copy
Edit
cd %USERPROFILE%\Desktop\rubixe
Install the required libraries:

bash
Copy
Edit
pip install streamlit sentence-transformers faiss-cpu torch transformers
Run the Streamlit App:
bash
Copy
Edit
streamlit run streamlit_app.py
This will open a browser tab with your app. You can now ask questions based on your lecture transcript.



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

