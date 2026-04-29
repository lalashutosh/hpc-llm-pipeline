import os
import sys

print("--- 1. Testing Core Imports & Pydantic ---", flush=True)
try:
    from pydantic import BaseModel, Field
    import json
    
    # Test our exact schema
    class QMLSchema(BaseModel):
        analysis: str
        paper: str
    
    # Verify Pydantic can generate the schema JSON that vLLM needs
    schema_json = QMLSchema.schema_json()
    print("✅ Pydantic schema compiled successfully.")
except Exception as e:
    print(f"❌ Pydantic failed: {e}")
    sys.exit(1)

print("\n--- 2. Testing Data Directories ---", flush=True)
INPUT_DIR = "selected_500"
if not os.path.exists(INPUT_DIR):
    print(f"❌ '{INPUT_DIR}' directory is missing!")
    sys.exit(1)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pdf')]
print(f"✅ Found {len(files)} PDFs in '{INPUT_DIR}'.")
if len(files) != 500:
    print(f"⚠️ Warning: Expected 500 files, found {len(files)}. Proceeding anyway.")

print("\n--- 3. Testing Anti-Hallucination Split Logic ---", flush=True)
import random
def test_split(text):
    split_point = int(len(text) * 0.25)
    roll = random.random()
    if roll < 0.2: return "Contextless (Top 25% deleted)"
    elif roll < 0.4: return "Premature Cutoff (Bottom 25% deleted)"
    else: return "Golden (Full text)"

dummy_text = "A" * 2000
print(f"✅ Split logic functional. Example roll: {test_split(dummy_text)}")

print("\n--- 4. Testing vLLM Installation (Expect CUDA Warnings Here!) ---", flush=True)
try:
    from vllm import LLM, SamplingParams
    print("\n✅ vLLM imported successfully!")
except Exception as e:
    print(f"\n❌ vLLM import failed: {e}")
    print("Did you run 'pip install vllm' inside the active environment?")
    sys.exit(1)

print("\n--- 5. Checking Hugging Face Cache Path ---", flush=True)
hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
print(f"ℹ️ Current HF_HOME is set to: {hf_home}")
if "scratch" not in hf_home:
    print("⚠️ WARNING: HF_HOME is not pointing to scratch. Make sure your run_teacher.sh exports it!")
else:
    print("✅ HF Cache path is safe.")

print("\n🚀 TEACHER PIPELINE SMOKE TEST PASSED. Ready for sbatch!")
