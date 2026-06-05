import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Semantic Search Engine")
st.write("Search using meaning instead of exact keywords.")

# Load Model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Knowledge Base
knowledge_base = [
    "The Semantic Search Engine uses AI to understand the meaning of a user's query instead of matching exact keywords.",
    "It converts documents and user queries into vector embeddings using Sentence Transformers.",
    "Cosine similarity is used to find the most relevant document or answer based on semantic meaning.",
    "The system can retrieve accurate results even when the user's wording differs from the stored content.",
    "This project is useful for FAQ bots, document search, customer support systems, and RAG-based applications.",
    "Transformers power modern Large Language Models.",
    "Embeddings capture semantic meaning.",
    "Vector databases store embeddings.",
    "Machine Learning learns patterns from data.",
    "Python is widely used in AI."
]

# Generate Embeddings
knowledge_embeddings = model.encode(knowledge_base)

# User Input
query = st.text_input(
    "Ask your question:",
    placeholder="Example: Tell me about AI"
)

if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a question.")
    else:

        # Query Embedding
        query_embedding = model.encode([query])

        # Similarity Scores
        scores = cosine_similarity(
            query_embedding,
            knowledge_embeddings
        )[0]

        # Best Match
        best_index = np.argmax(scores)

        st.subheader("🎯 Best Match")

        st.success(
            knowledge_base[best_index]
        )

        st.subheader("📊 Similarity Scores")

        for sentence, score in sorted(
            zip(knowledge_base, scores),
            key=lambda x: x[1],
            reverse=True
        ):
            st.write(
                f"**Score:** {score:.4f}"
            )
            st.write(sentence)
            st.divider()

# Sidebar
st.sidebar.title("About Project")

st.sidebar.info(
    """
    This Semantic Search Engine uses:
    
    • Sentence Transformers
    
    • Embeddings
    
    • Cosine Similarity
    
    • Streamlit
    
    The system understands meaning rather than exact keywords.
    """
)