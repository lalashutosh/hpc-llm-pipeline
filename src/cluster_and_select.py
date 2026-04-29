import os
import fitz  # PyMuPDF
import shutil
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# --- CONFIGURATION ---
INPUT_DIR = "qml_papers/papers"          # The folder where your 1500 PDFs are unpacking
OUTPUT_DIR = "selected_500"       # Where the chosen PDFs will go
NUM_CLUSTERS = 10
SAMPLES_PER_CLUSTER = 50
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5" # Excellent open-source embedding model

def extract_abstract(pdf_path):
    """Extracts the first 3000 characters of a PDF (usually captures Title + Abstract + Intro)."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        # Only read the first 2 pages to save memory and time
        for page_num in range(min(2, len(doc))):
            text += doc[page_num].get_text()
        
        # Clean up and truncate
        text = text.replace("\n", " ").strip()
        return text[:3000] if len(text) > 100 else None
    except Exception as e:
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pdf')]
    print(f"📄 Found {len(pdf_files)} PDFs. Extracting text...")

    data = []
    for pdf in pdf_files:
        text = extract_abstract(os.path.join(INPUT_DIR, pdf))
        if text:
            data.append({"filename": pdf, "text": text})

    df = pd.DataFrame(data)
    print(f"✅ Successfully extracted text from {len(df)} papers.")

    # --- EMBEDDING ---
    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("⏳ Embedding papers (this will take a minute on GPU)...")
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True, device="cuda")

    # --- CLUSTERING ---
    print(f"📊 Clustering into {NUM_CLUSTERS} semantic groups...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(embeddings)

    # --- STRATIFIED SAMPLING ---
    print("🎯 Selecting papers from each cluster...")
    selected_papers = []
    
    for cluster_id in range(NUM_CLUSTERS):
        cluster_df = df[df['cluster'] == cluster_id]
        # Sample 50, or all of them if the cluster has fewer than 50
        sample_size = min(SAMPLES_PER_CLUSTER, len(cluster_df))
        sampled = cluster_df.sample(n=sample_size, random_state=42)
        selected_papers.append(sampled)
        print(f"  - Cluster {cluster_id}: Selected {sample_size} / {len(cluster_df)} papers")

    final_df = pd.concat(selected_papers)
    
    # --- SPLIT AND COPY FILES ---
    HOLDOUT_DIR = "holdout_test_set"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HOLDOUT_DIR, exist_ok=True)

    # Identify the unselected papers
    selected_filenames = set(final_df['filename'])
    holdout_df = df[~df['filename'].isin(selected_filenames)]

    print(f"📁 Copying {len(final_df)} training papers to {OUTPUT_DIR}/...")
    for filename in final_df['filename']:
        shutil.copy2(os.path.join(INPUT_DIR, filename), os.path.join(OUTPUT_DIR, filename))

    print(f"🔒 Copying {len(holdout_df)} blind-test papers to {HOLDOUT_DIR}/...")
    for filename in holdout_df['filename']:
        shutil.copy2(os.path.join(INPUT_DIR, filename), os.path.join(HOLDOUT_DIR, filename))

    # Save tracking reports
    final_df.to_csv("training_selection_report.csv", index=False)
    holdout_df.to_csv("blind_test_holdout_report.csv", index=False)
    
    print("🚀 Done! Training and Holdout sets are completely isolated.")

if __name__ == "__main__":
    main()
