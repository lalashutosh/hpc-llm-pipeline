import os
import json
import random
import fitz # PyMuPDF
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams

# --- CONFIGURATION ---
INPUT_DIR = "selected_500"
OUTPUT_FILE = "qml_finetune_dataset_v4.jsonl"
# We use a 32B model for high reasoning. It requires multiple V100s.
TEACHER_MODEL = "Qwen/Qwen2.5-32B-Instruct"

# --- PYDANTIC SCHEMA ---
class QMLSchema(BaseModel):
    analysis: str = Field(description="Brief logic outlining what information is present or missing.")
    paper: str = Field(description="[Title, Authors, Year] or '[Not explicitly detailed]'")
    problem: str = Field(description="[1-2 sentences gap addressed] or '[Not explicitly detailed]'")
    approach: str = Field(description="[2-3 sentences technical idea] or '[Not explicitly detailed]'")
    result: str = Field(description="[1-2 sentences demonstrated] or '[Not explicitly detailed]'")
    implication: str = Field(description="[1-2 sentences meaning for field] or '[Not explicitly detailed]'")
    open_questions: str = Field(description="[1 sentence unresolved] or '[Not explicitly detailed]'")

def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc[:4]]) # First 4 pages
        return text.strip()
    except:
        return ""

def apply_curriculum_split(text):
    """Enforces the Anti-Hallucination Split: 60% Golden, 20% Premature Cutoff, 20% Contextless"""
    if len(text) < 1000:
        return text, "golden"
        
    split_point = int(len(text) * 0.25)
    roll = random.random()
    
    if roll < 0.2:
        # Contextless: Delete the top 25% (Removes Abstract/Intro)
        return text[split_point:], "contextless"
    elif roll < 0.4:
        # Premature Cutoff: Delete the bottom 25% (Removes Conclusion/Results)
        return text[:-split_point], "premature"
    else:
        # Golden: Keep the whole excerpt
        return text, "golden"

def build_chatml_record(paper_filename, raw_excerpt, json_output):
    """Formats the final training record for Gemma 3 / Qwen."""
    try:
        data = json.loads(json_output)
    except:
        return None # Skip if JSON parsing fails
        
    target_output = (
        f"<analysis>\n{data.get('analysis', '')}\n</analysis>\n\n"
        f"Paper: {data.get('paper', '')}\n"
        f"Problem: {data.get('problem', '')}\n"
        f"Approach: {data.get('approach', '')}\n"
        f"Result: {data.get('result', '')}\n"
        f"Implication: {data.get('implication', '')}\n"
        f"Open questions: {data.get('open_questions', '')}"
    )

    instruction = "Extract and format the following research summary into the exact 6-field schema."
    return {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Paper Name: {paper_filename}\n\n{raw_excerpt}"},
            {"role": "assistant", "content": target_output}
        ]
    }

def main():
    print("📄 Loading PDFs and applying Curriculum Split...")
    prompts = []
    metadata = []
    
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".pdf"): continue
        
        text = extract_text(os.path.join(INPUT_DIR, filename))
        if not text: continue
            
        excerpt, split_type = apply_curriculum_split(text)
        
        # Build the prompt for the Teacher Model
        system_prompt = "You are an expert QML data annotator. Read the excerpt and extract the 6 fields. If information is missing, you MUST write '[Not explicitly detailed]'. Output STRICT JSON."
        user_prompt = f"Paper File: {filename}\n\nExcerpt:\n{excerpt}"
        
        # Qwen Chat Template formatting
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        prompts.append(formatted_prompt)
        metadata.append((filename, excerpt))

    print(f"🚀 Booting vLLM with {TEACHER_MODEL} across multiple GPUs...")
    # tensor_parallel_size=4 splits the massive 32B model across 4 V100 GPUs
    llm = LLM(
        model=TEACHER_MODEL, 
        tensor_parallel_size=4, 
        dtype="half", 
        trust_remote_code=True,
        max_model_len=8192,             # Cap the memory allocation to 8k tokens
        gpu_memory_utilization=0.85     # Leave 15% of VRAM free for safety
    )
    
    # guided_json forces the LLM to output valid JSON matching our Pydantic schema
    sampling_params = SamplingParams(
        temperature=0.1, 
        max_tokens=800,
        guided_json=QMLSchema.schema_json() 
    )
    
    print("🧠 Generating JSON annotations (This is blazing fast)...")
    outputs = llm.generate(prompts, sampling_params)
    
    print(f"💾 Saving formatted ChatML dataset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for i, output in enumerate(outputs):
            json_str = output.outputs[0].text
            filename, excerpt = metadata[i]
            
            chatml_record = build_chatml_record(filename, excerpt, json_str)
            if chatml_record:
                f.write(json.dumps(chatml_record) + "\n")

    print("✅ Dataset generation complete!")

if __name__ == "__main__":
    main()
