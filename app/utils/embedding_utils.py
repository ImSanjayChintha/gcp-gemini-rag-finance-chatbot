from google.cloud import aiplatform_v1beta1 as aiplatform
import numpy as np

PROJECT_ID = "<GCP PROJECT ID>"
LOCATION = "us-central1"
MODEL_ID = "textembedding-gecko@001"
ENDPOINT = f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}"

def get_embedding_model():
    return None  # Placeholder, not used directly

def embed_text_chunks(chunks, _model=None, batch_size=10):
    client = aiplatform.PredictionServiceClient()
    embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        instances = [{"content": chunk} for chunk in batch]

        response = client.predict(
            endpoint=ENDPOINT,
            instances=instances,
            parameters={},
            timeout=60.0 
        )

        embeddings += [pred["embeddings"]["values"] for pred in response.predictions]

    return chunks, np.array(embeddings)
