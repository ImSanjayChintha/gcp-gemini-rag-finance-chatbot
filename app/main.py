
import streamlit as st
from utils.embedding_utils import get_embedding_model, embed_text_chunks
from utils.search_utils import search_similar_chunks
from google.cloud import aiplatform_v1beta1 as aiplatform
from pathlib import Path
import fitz 
# GCP Configuration
PROJECT_ID = "<GCP PROJECT ID>"
LOCATION = "us-central1"
PUBLISHER_MODEL = "projects/{0}/locations/{1}/publishers/google/models/gemini-1.5-pro-preview".format(PROJECT_ID, LOCATION)

def load_pdf_text_chunks(pdf_file, max_chunk_size=500):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    doc.close()
    # Simple split
    return [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

def query_gemini(prompt, context):
    client = aiplatform.PredictionServiceClient()
    instance = {
        "context": context,
        "messages": [{"role": "user", "content": prompt}],
    }
    parameters = {
        "temperature": 0.4,
        "maxOutputTokens": 1024,
        "topP": 1,
        "topK": 32,
    }
    response = client.predict(
        endpoint=f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/{PUBLISHER_MODEL}:predict",
        instances=[instance],
        parameters=parameters,
    )
    return response.predictions[0]["candidates"][0]["content"]

st.title("📄 GCP Gemini PDF Finance Assistant")

pdf_file = st.file_uploader("Upload a financial PDF", type=["pdf"])
query = st.text_input("Ask a question from the document:")

if pdf_file and query:
    with st.spinner("Processing..."):
        chunks = load_pdf_text_chunks(pdf_file)
        model = get_embedding_model()
        chunks, vectors = embed_text_chunks(chunks, model)
        top_chunks = search_similar_chunks(query, chunks, vectors, model)
        context = "\n".join(top_chunks)
        answer = query_gemini(query, context)
        st.markdown("### 💬 Gemini Response:")
        st.write(answer)
