# HPC-Based Continual Fine-Tuning Pipeline

## What Is It?

Modern researchers repeatedly teach AI the same things: their domain context, preferred reasoning style, and desired output format—only to lose that progress at the next interaction.

This project was built to solve that problem: **a model that learns and grows with the researcher**.

Instead of starting over in every session, the pipeline enables a foundation model to continuously evolve alongside a growing knowledge base—learning new domain information, adapting to preferred output styles, and preserving previously acquired behavior through iterative fine-tuning.

The system is domain-agnostic by design. It can ingest any specialized corpus (research papers, technical reports, internal documentation, or enterprise knowledge bases), identify the most informative new knowledge, convert that knowledge into training supervision, and incrementally update a target model without full retraining.

Its goal is not only to teach facts, but to teach **retrieval-aware reasoning**—training the model to answer accurately when grounded in external evidence, while explicitly recognizing when insufficient context exists instead of hallucinating.

---

## Technical Engine

Under the hood, the system performs parameter-efficient **Supervised Fine-Tuning (SFT)** using **LoRA adapters** on a **Gemma-3-12B** student model inside a **SLURM-managed HPC environment**.

To fit a 12B-parameter model within single-GPU memory limits, the training stack combines:

- **bitsandbytes 4-bit NF4 quantization** for compressed model loading
- **Paged AdamW 8-bit optimization** for memory-efficient gradient updates
- mixed precision execution on **32GB V100 hardware**
- distributed batch orchestration through SLURM

Rather than directly fine-tuning on raw documents, the system uses a **teacher–student distillation framework**:
a larger **Qwen-14B-Instruct** model first transforms curated source material into synthetic instruction-response pairs, which are then used to supervise the smaller deployable student model.

This approach improves sample efficiency, reduces catastrophic forgetting, and enables continual adaptation with significantly lower compute cost than full retraining.

---

## Architecture
<img width="1530" height="589" alt="finetuning" src="https://github.com/user-attachments/assets/20565903-1f07-4478-8b22-2121473e4463" />


The pipeline is designed as a continual learning loop that incrementally updates the student model with only high-value knowledge additions.

### 1. Corpus Ingestion
A growing research corpus is continuously ingested from raw PDFs and converted into machine-readable text for downstream processing.

### 2. Semantic Diversity Sampling
Documents are embedded into vector space and clustered using **K-Means centroid selection** to identify the most representative and diverse subset of the corpus, removing redundancy while preserving maximum knowledge coverage.

### 3. Synthetic Data Generation (Teacher)
A larger **Qwen-14B-Instruct** teacher model converts the selected documents into structured instruction-response examples (`JSONL`), creating a supervised curriculum tailored for downstream adaptation.

### 4. Behavioral Alignment (Anti-Hallucination)
The synthetic dataset is mixed with a curriculum designed to shape model behavior:
- **Golden Path** samples reinforce grounded answers from relevant context
- **Contextless Traps** teach refusal when evidence is missing
- **Premature Cutoffs** train robustness against incomplete retrieval

This explicitly teaches retrieval-aware reasoning and reduces hallucination.

### 5. Incremental Fine-Tuning (Student)
The curated curriculum is used to LoRA fine-tune **Gemma-12B** under 4-bit quantization, enabling efficient updates on single-GPU HPC hardware without full retraining.

---

### System Outcome
The final model behaves as a **continually evolving research assistant**:
it absorbs new knowledge, preserves prior learning, maintains domain relevance, and adapts incrementally as the researcher’s corpus grows.

## Design Decisions

### 1. Semantic Diversity Filtering (K-Means Centroids)

Specialized fine-tuning risks **knowledge collapse**—overfitting to heavily represented topics while ignoring niche but important concepts.

To prevent this, the full document corpus is embedded into vector space and clustered using **K-Means**. Rather than sampling randomly, the pipeline selects documents nearest to each cluster centroid, preserving semantic diversity while removing redundancy.

**Impact:** maximizes knowledge coverage, improves sample efficiency, and stabilizes downstream fine-tuning.

---

### 2. V100 Volta Hardware Compatibility

The target HPC environment used legacy **NVIDIA V100 (Volta)** GPUs, which do not support native **bfloat16** operations.

Modern fine-tuning libraries often assume bf16 availability, causing AMP failures and unstable gradients on Volta hardware. To resolve this, the training stack explicitly overrides default precision behavior by:

- forcing model weights to **float16** immediately after load,
- manually casting trainable **LoRA adapters** to **float32**,
- preserving mixed precision only where numerically safe.

**Impact:** stable training on older hardware without sacrificing model scale.

---

### 3. Hybrid Environment Management (Modules + Virtualenv)

HPC systems expose highly optimized native libraries through environment modules, but Python ML stacks often require tightly pinned dependencies.

To balance both, the pipeline uses a **hybrid environment strategy**:

- system-level **PyTorch/CUDA** loaded via HPC modules for maximum hardware performance,
- application-layer libraries (`transformers`, `trl`, `peft`) isolated in a `--system-site-packages` virtual environment,
- bounded package versions to avoid conflicts with cluster-native containers (e.g., `vLLM`).

**Impact:** reproducibility without losing vendor-level HPC optimizations.

---

### 4. Anti-Hallucination Curriculum (Behavioral Alignment)

A major weakness in standard RAG systems is assuming retrieved context is always correct.

To explicitly teach uncertainty handling, the synthetic dataset is behaviorally balanced:

- **60% Golden Path** — grounded reasoning from relevant context
- **20% Contextless Traps** — irrelevant or empty retrieval; model must refuse confidently
- **20% Premature Cutoffs** — truncated evidence; model learns to detect incomplete context

**Impact:** stronger refusal behavior, lower hallucination rates, and better retrieval robustness.

---

### 5. Resource-Aware Smoke Testing

Submitting failed SLURM jobs wastes expensive shared compute resources and slows iteration.

To minimize cluster expenditure, every pipeline stage was designed with a **local smoke-test mode** executable directly on the login node:

- dataset path validation,
- dependency verification,
- dry-run model loading,
- batch shape checks,
- memory sanity tests.

Only validated jobs are promoted to full GPU execution.

**Impact:** faster debugging, lower queue waste, and significantly reduced HPC resource burn.

## How to Reproduce

This pipeline is designed for execution on a SLURM-managed HPC cluster.

### 1. Clone the Repository
Move to your scratch storage and clone the project:

```bash
git clone <repository_url>
cd hpc-llm-pipeline
```

---

### 2. Set Up the Environment
Load the cluster PyTorch module, create a virtual environment, and install dependencies:

```bash
module load pytorch
python3 -m venv --system-site-packages v100_compute_env
source v100_compute_env/bin/activate
pip install -r requirements.txt
```

---

### 3. Add Your Dataset
Place all source PDFs in the default input directory:

```bash
mkdir -p qml_papers/papers
# copy your PDF files into qml_papers/papers/
```

> Optional: To use a custom input path, update the configuration in `src/cluster_and_select.py`.

---

### 4. Run Semantic Clustering
Generate embeddings, perform diversity-aware clustering, and select the representative subset:

```bash
sbatch cluster.sh
```

---

### 5. Generate Synthetic Training Data
Launch the Qwen-14B teacher model to convert selected papers into JSONL instruction-response pairs:

```bash
sbatch run_teacher.sh
```

---

### 6. Fine-Tune the Student Model
Run supervised fine-tuning on Gemma-12B using the generated dataset:

```bash
sbatch finetune.sh
```

---

### Output
Final outputs include:

- clustered corpus subset
- synthetic instruction dataset (`.jsonl`)
- fine-tuned model checkpoints
- training logs and evaluation metrics
