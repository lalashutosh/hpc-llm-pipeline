import os
import sys

print("--- 1. Testing Imports ---", flush=True)

try:
    print("Loading PyMuPDF (fitz)...", end=" ", flush=True)
    import fitz
    print("✅", flush=True)

    print("Loading Pandas...", end=" ", flush=True)
    import pandas as pd
    print("✅", flush=True)

    print("Loading PyTorch...", end=" ", flush=True)
    import torch
    print("✅", flush=True)

    print("Loading SentenceTransformers...", end=" ", flush=True)
    from sentence_transformers import SentenceTransformer
    print("✅", flush=True)

    print("All modules imported successfully!\n", flush=True)
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    sys.exit(1)

# ... (Keep the rest of your script exactly the same starting from "--- 2. Testing File Paths ---")
print("\n--- 2. Testing File Paths ---")
INPUT_DIR = "qml_papers/papers"
if not os.path.exists(INPUT_DIR):
    print(f"❌ Directory '{INPUT_DIR}' does not exist. Are you in the right folder?")
    sys.exit(1)

pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pdf')]
if len(pdf_files) == 0:
    print(f"❌ Directory exists, but found 0 PDFs inside it.")
    sys.exit(1)
else:
    print(f"✅ Found {len(pdf_files)} PDFs in '{INPUT_DIR}'")

print("\n--- 3. Testing PyMuPDF Extraction ---")
try:
    test_file = os.path.join(INPUT_DIR, pdf_files[0])
    doc = fitz.open(test_file)
    text = doc[0].get_text().strip()[:100]
    print(f"✅ Successfully read first PDF.")
    print(f"   Preview: '{text}...'")
except Exception as e:
    print(f"❌ PDF extraction failed: {e}")
    sys.exit(1)

print("\n--- 4. Testing PyTorch Environment ---")
print(f"✅ PyTorch Version: {torch.__version__}")
# Note: It's perfectly normal for this to be False on a login node!
print(f"✅ CUDA Available: {torch.cuda.is_available()}")

if "cu118" not in torch.__version__ and "cu121" not in torch.__version__:
    print("\n⚠️ WARNING: PyTorch might not be the GPU version. Check your installation.")
else:
    print("\n🚀 ALL SYSTEMS GO! SMOKE TEST PASSED.")
