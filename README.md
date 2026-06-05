# Multimodal Medical Vision-Language Graph-Vector RAG

MedVLM HybridRAG is an advanced, clinical-grade Retrieval-Augmented Generation (RAG) pipeline designed to process raw medical scans (e.g., X-rays, CTs, MRIs) and complex PDF medical reports. It leverages state-of-the-art Multimodal Vision-Language Models (VLMs) for parsing, stores structured and unstructured facts in a Hybrid Storage Layer (Neo4j + Qdrant), and employs an agentic search workflow via LangGraph to produce zero-hallucination, medically verified answers.

---

## System Architecture

```
         [Raw Medical Scans / PDFs]
                     │
                     ▼
┌───────────────────────────────────────────┐
│        Multimodal Vision Parser           │ ──► Local VLM: [Qwen2.5-VL / Llama-3.2-Vision]
└───────────────────────────────────────────┘
                     │
                     │ (Strictly Validated JSON Output via Pydantic)
                     ▼
┌───────────────────────────────────────────┐
│          Hybrid Storage Layer             │
│  ├─ Graph Database (Neo4j)                │ ──► Tracks Entities (Patient, Drug, Disease)
│  └─ Vector Database (Qdrant)              │ ──► Stores Dense Context & Clinical Descriptions
└───────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│       Agentic Inference (LangGraph)       │ ──► Loop: Query Rewriter ➔ Evaluator ➔ BGE-Reranker
└───────────────────────────────────────────┘
                     │
                     ▼
    [Zero-Hallucination Verified Answer]
```

### Architectural Breakdown

1. **Multimodal Vision Parser**: Ingests raw medical scans, X-rays, MRIs, or PDF reports and processes them using advanced vision-language models like **Qwen2.5-VL** or **Llama-3.2-Vision**. Raw extractions are mapped into strictly validated JSON schemas using **Pydantic** to prevent parsing hallucinations.
2. **Hybrid Storage Layer**:
   - **Graph Database (Neo4j)**: Maps structural relationships between primary entities (e.g., `Patient` → `DIAGNOSED_WITH` → `Disease` → `TREATED_WITH` → `Drug`).
   - **Vector Database (Qdrant)**: Indexes dense semantic representations of clinical reports, narrative summaries, and dense visual-text descriptions.
3. **Agentic Inference Layer (LangGraph)**: An autonomous orchestration loop that dynamically:
   - **Rewrites** queries to improve search specificity.
   - **Retrieves** information in parallel from Neo4j and Qdrant.
   - **Evaluates** the retrieved medical data to check for gaps or contradictions.
   - **Reranks** context candidates using a BGE-Reranker cross-encoder.
   - Generates a **verified, hallucination-free answer** grounded in empirical clinical data.

---

## Tech Stack

* **Orchestration:** `llama-index-core` / `langgraph`
* **Databases:** `neo4j` (Property Graphs) / `qdrant-client` (Vector Indexing)
* **Models:** `transformers` (Local VLM execution) / `sentence-transformers` (BGE-Reranker & Embeddings)
* **Data Validation:** `pydantic` / `pydantic-settings`
* **Interface:** `streamlit`

---

## Getting Started

### 1. Prerequisites & Infrastructure
Launch the required graph and vector microservices via Docker Compose:

```bash
docker-compose up -d
```

### 2. Environment Setup
Configure your credentials by copying the example environment file:
```bash
cp .env_example .env
```

### 3. Installation

Install the package and all dependencies locally in your Python virtual environment:

```bash
pip install .
pip install -e .
```


## Execution Flow

### Step 1: Parsing and Hybrid Ingestion

Ingest a directory of raw scans or PDFs to parse them with the VLM, populate the Neo4j schema, and build vector embeddings in Qdrant:

```bash
python -m medvlm_hybrid_rag.ingest --data-dir ./data/raw_scans/
```

### Step 2: Running the Agentic Query Engine

Execute semantic queries via the LangGraph interactive CLI agent:

```bash
python -m medvlm_hybrid_rag.query "Analyze patient X's recent chest X-ray and reconcile with prescribed Beta Blockers"
```

### Step 3: Streamlit Interactive UI

Launch the visual dashboard to upload scans, view Neo4j graph nodes, and run agent queries:

```bash
streamlit run app.py
```

---

## Data & Graph Schema

The pipeline uses strict **Pydantic** validation models to convert unstructured VLM outputs into a typed structure.

### Entity-Relation JSON Definition
```json
{
  "patient_id": "PT-9942",
  "entities": [
    {"id": "E1", "type": "Disease", "name": "Pneumonia", "attributes": {"severity": "Moderate"}},
    {"id": "E2", "type": "Drug", "name": "Amoxicillin", "attributes": {"dosage": "500mg"}}
  ],
  "relationships": [
    {"source": "PT-9942", "target": "E1", "type": "DIAGNOSED_WITH"},
    {"source": "E1", "target": "E2", "type": "TREATED_WITH"}
  ],
  "narrative_summary": "Patient presented with a consolidation in the right lower lobe consistent with moderate bacterial pneumonia. Prescribed amoxicillin."
}
```

---

## Zero-Hallucination Agent Flow

The agent utilizes **LangGraph** to guarantee clinical accuracy. The query evaluation sequence is as follows:

```
[User Query] ➔ [Query Rewriter] ➔ [Parallel Graph & Vector Retrieval] ➔ [BGE-Reranking] ➔ [Evaluator Critique] ── (Pass) ──► [Grounding Verification] ➔ [Final Output]
                                                                                                    │
                                                                                                 (Fail)
                                                                                                    ▼
                                                                                            [Query Refinement]
```

1. **Dynamic Query Rewriter**: Reformulates patient-specific history and clinical jargon into pristine database queries.
2. **Parallel Hybrid Retriever**: Traverses the **Neo4j Cypher** graph paths while executing vector similarity search against **Qdrant**.
3. **Cross-Encoder Reranking**: Utilizes local `BGE-Reranker-Large` to prioritize precise medical contexts.
4. **Evaluator Guardrails**: A strict LLM/VLM evaluator judges if the retrieved context is fully sufficient to answer. If insufficient, it triggers loop-back refinement.
5. **Zero-Hallucination Filter**: Prior to final streaming output, the agent validates that every claim in the response is strictly supported by active source citations, preventing any medical hallucinations.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
