# streamlit_app.py

import streamlit as st
from new import EnhancedRAGQA  # Replace 'your_script_name' with the name of the file above (without .py)

# Load QA system
qa_system = EnhancedRAGQA("embeddings/faiss_index.bin", "embeddings/passages.json")

# UI
st.title("🎓 Lecture Transcript Q&A System")
st.caption(" Internship Project by Rakshitha S ✨")

st.markdown("Ask questions from the lecture transcript to get concise answers!")

query = st.text_input("🔎 Enter your question:")

if query:
    with st.spinner("Generating answer..."):
        result = qa_system.answer_with_citations(query)


if query:
    result = qa_system.answer_with_citations(query)

    st.subheader("💬 Answer:")
    st.write(result["answer"])


    st.subheader("📄 Source Passages:")
    for i, src in enumerate(result["sources"]):
        with st.expander(f"Source {i+1}"):
            st.write(src)




