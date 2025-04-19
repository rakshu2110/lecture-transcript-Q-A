import os
import json
import torch
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

import requests
from urllib.parse import urlparse
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

def clean_transcript(text):
    text = re.sub(r"http\S+", "", text)

    fillers = ['OK', 'Okay', 'So', 'Right', 'Well', 'Just', 'Let me', 'I guess', 'I want to', 'Shall I']
    for f in fillers:
        text = re.sub(rf'\b{f}\b', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()

# --- Download transcript ---

def download_file(url, save_dir="lecture_pdfs"):
    """Download a PDF or HTML file from a URL and save it locally."""
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, os.path.basename(urlparse(url).path))
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {filename}")
        return filename
    else:
        print(f"Failed to download: {url}")
        return None

# --- Data Ingestion ---

def extract_text_from_pdf(pdf_path):
    """Extract plain text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def preprocess_text_to_json(clean_text, output_path):
    """Split text into passages and save as JSON."""

    # Split into paragraphs, then further split long paragraphs into smaller chunks
    paragraphs = clean_text.split("\n\n")
    passages = []
    
    for i, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue
            
        # Chunk long paragraphs (more than 500 chars) into smaller pieces
        if len(paragraph) > 500:
            words = paragraph.split()
            chunk_size = 100  # approximate number of words per chunk
            for j in range(0, len(words), chunk_size):
                chunk = " ".join(words[j:j+chunk_size])
                if chunk.strip():
                    passages.append({"id": f"{i}-{j//chunk_size}", "text": chunk.strip()})
        else:
            passages.append({"id": str(i), "text": paragraph.strip()})
    
    with open(output_path, 'w') as f:
        json.dump(passages, f, indent=4)
    return passages
         

def process_transcripts(input_dir="transcripts", output_dir="data"):
    """Process all PDFs in the input directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_count = 0
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            
            # Step 1: Extract text from PDF
            text = extract_text_from_pdf(pdf_path)
            
            if text:  # If text extraction is successful
                # Step 2: Clean the extracted text
                cleaned_text = clean_transcript(text)  # Clean it here
                
                # Step 3: Process and save cleaned text into JSON
                output_path = os.path.join(output_dir, f"{filename[:-4]}.json")
                preprocess_text_to_json(cleaned_text, output_path)
                
                print(f"Processed {filename} -> {output_path}")
                processed_count += 1

    print(f"Processed {processed_count} PDF files")
    return processed_count > 0

# --- Knowledge Base ---

def load_passages(data_dir="data"):
    """Load all passages from JSON files."""
    passages = []
    passage_sources = {}  
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for entry in data:
                        passage_id = f"{filename}:{entry['id']}"
                        passages.append(entry["text"])
                        passage_sources[len(passages)-1] = passage_id
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return passages, passage_sources

def build_vector_database(passages, output_dir="embeddings"):
    """Generate embeddings and build FAISS index."""
    if not passages:
        print("No passages to embed!")
        return None, None, None
        
    print(f"Building embeddings for {len(passages)} passages...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(passages, show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    with open(os.path.join(output_dir, "passages.json"), 'w') as f:
        json.dump(passages, f)
        
    print(f"Index built with {index.ntotal} vectors of dimension {dimension}")
    return model, index, passages

def retrieve_passages(query, model, index, passages, k=3):
    """Embed query and retrieve top-k passages."""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(passages):
            results.append((passages[idx], distances[0][i]))
    
    return results

# --- Improved RAG QA System ---

class EnhancedRAGQA:
    def __init__(self, faiss_index_path, passages_path):
        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)
        
        # Load passages
        with open(passages_path, 'r') as f:
            self.passages = json.load(f)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize T5 for answer generation
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")

    def retrieve(self, query, k=3):
        """Retrieve relevant passages using vector similarity."""
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        
        retrieved_passages = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.passages):
                # Add both passage and its relevance score
                retrieved_passages.append({
                    "text": self.passages[idx],
                    "score": float(distances[0][i])
                })
        
        return retrieved_passages

    def generate_answer(self, query):
        """Generate a concise answer using retrieved context and T5."""
        # Get top relevant passages
        retrieved_docs = self.retrieve(query)
        
        if not retrieved_docs:
            return "I couldn't find any relevant information to answer your question.", []
        
        # Combine retrieved passages into context
        contexts = [doc["text"] for doc in retrieved_docs]
        
        # Format input for T5 (which expects a text-to-text format)
        #input_text = f"question: {query} context: {' '.join(contexts)}"

        input_text = f"Answer briefly: {query} Context: {' '.join(contexts)}"

        
        # Tokenize and generate
        input_ids = self.t5_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).input_ids
        
        with torch.no_grad():
            output = self.t5_model.generate(
                input_ids=input_ids,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )
        
        answer = self.t5_tokenizer.decode(output[0], skip_special_tokens=True)
        
        return answer, contexts

    def answer_with_citations(self, query):
        """Generate an answer with citations to the source passages."""
        answer, contexts = self.generate_answer(query)
        
      
    # Capitalize the first letter of the answer
        if answer:
            answer = answer[0].upper() + answer[1:]

    # Format sources (top 3)
        formatted_sources = [src.strip() for src in contexts[:3]]


    # Final structured output
        result = {
            "question": query,
            "answer": answer,
            "sources": formatted_sources
        }

        return result


# --- Usage example ---
def main():
    # Configuration parameters
    config = {
        "lecture_url": "https://ocw.mit.edu/courses/18-085-computational-science-and-engineering-i-fall-2008/48dc80405bc69c3dec2a2250efb79c55_18-085F08-L01.pdf",
        "input_dir": "lecture_pdfs",
        "output_dir": "structured_output",
        "embeddings_dir": "embeddings"
    }
    
    # Ensure directories exist
    for dir_path in [config["input_dir"], config["output_dir"], config["embeddings_dir"]]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Download file if needed
    file_path = download_file(config["lecture_url"], config["input_dir"])
    
    # Process transcripts
    process_transcripts(config["input_dir"], config["output_dir"])
    
    # Load passages
    passages, _ = load_passages(config["output_dir"])
    
    # Build vector database if needed
    index_path = os.path.join(config["embeddings_dir"], "faiss_index.bin")
    passages_path = os.path.join(config["embeddings_dir"], "passages.json")
    
    if not os.path.exists(index_path) or not os.path.exists(passages_path):
        build_vector_database(passages, config["embeddings_dir"])
    
    # Initialize QA system
    qa_system = EnhancedRAGQA(
        faiss_index_path=index_path,
        passages_path=passages_path
    )
    
    # Example query
    question = "What is a symmetric matrix?"
    result = qa_system.answer_with_citations(question)
    
    # Print results
    print("\n" + "="*50)
    print(f"📘 Question: {result['question']}")
    print(f"💬 Answer: {result['answer']}")
    print("\n🔍 Sources:")
    for i, source in enumerate(result['sources']):
        print(f"  [{i+1}] {source[:200]}...")
    print("="*50)

if __name__ == "__main__":
    main()