# Evaluating Graph-Based and Hybrid RAG Architectures Using a Serverless Vector Database

**Master's Thesis — Nazim Alperen Akcakaya**
Universitat Rovira i Virgili, 2026
Directed by Dr. Pedro Garcia Lopez

---

## Overview

This repository contains the code, benchmark questions, and result data for a controlled comparison of five Retrieval-Augmented Generation (RAG) architectures evaluated on an academic question-answering task. The central question is whether a serverless vector database (Blocks-DB) can match the answer quality of graph-based systems (GraphRAG, LightRAG) at lower latency and cost.

All five systems were evaluated on the same 20-question benchmark, using the same document corpus and GPT-4o-mini as both the generation and evaluation model.

---

## Repository Structure

```
.
├── questions/                    # 20-question benchmark (shared across all systems)
├── baseline_rag/                 # Baseline RAG system files
├── lightrag_rag/                 # LightRAG system files
├── microsoft_graphrag/           # GraphRAG system files
├── results/                      # Per-system raw answer and latency outputs
│
├── test_baseline_all.py          # Baseline RAG runner (LangChain + ChromaDB)
├── lightrag_system.py            # LightRAG runner (HKUDS LightRAG library)
├── blocks_rag.py                 # Blocks-DB runner (AWS Lambda + S3)
├── hybrid_rag.py                 # Hybrid RAG runner (GraphRAG + Blocks-DB fusion)
├── evaluate_all.py               # Evaluation scorer (LLM-as-judge, all systems)
├── visualize_results.py          # Charts and figures used in the thesis
│
├── baseline_answers.json         # Raw answers + latency: Baseline RAG (n=20)
├── graphrag_answers.json         # Raw answers + latency: GraphRAG (n=20)
├── lightrag_answers.json         # Raw answers + latency: LightRAG (n=20)
├── blocks_rag_results.json       # Raw answers + latency + cost: Blocks-DB (n=20)
│
└── all_evaluations.json          # Final scored results for all 5 systems (n=20 each)
```

---

## Systems

| System | Description | Infrastructure |
|---|---|---|
| **Baseline RAG** | Standard dense vector retrieval | LangChain, ChromaDB, `text-embedding-ada-002` |
| **GraphRAG** | Knowledge graph + community summaries, local search mode | Microsoft GraphRAG library |
| **LightRAG** | Dual-level graph index, incremental updates | HKUDS LightRAG library |
| **Blocks-DB** | Serverless vector DB with block-based partitioning | AWS Lambda + S3 + EC2, Lithops, FAISS IVF |
| **Hybrid RAG** | Pre-computed GraphRAG answer + real-time Blocks-DB retrieval, fused by GPT-4o-mini | AWS (same as Blocks-DB) |

All systems use **GPT-4o-mini** (`gpt-4o-mini-2024-07-18`, temperature 0) for answer generation.

---

## Benchmark

### Document Corpus

Four academic papers in the RAG research domain (85 pages total):

| Paper | Authors | Year | Pages |
|---|---|---|---|
| From Local to Global: A Graph RAG Approach to Query-Focused Summarization | Edge et al. | 2024 | 21 |
| LightRAG: Simple and Fast Retrieval-Augmented Generation | Guo et al. | 2024 | 15 |
| TagRAG: Tag-guided Hierarchical Knowledge Graph RAG | Tao et al. | 2026 | 9 |
| Retrieval-Augmented Generation for LLMs: A Survey | Gao et al. | 2024 | 40 |

### Question Set

20 questions across four types and three difficulty levels (see `questions/`):

| Type | Count |
|---|---|
| Factual | 5 |
| Comparison | 5 |
| Explanation | 4 |
| Reasoning | 6 |

| Difficulty | Count |
|---|---|
| Easy | 3 |
| Medium | 8 |
| Hard | 9 |

---

## Results Summary

Final results from the four-document corpus run. GraphRAG used local search mode throughout.

| System | Overall (1-5) | Latency (mean) | vs Baseline |
|---|---|---|---|
| Baseline RAG | 3.78 | 3.40 s | n/a |
| GraphRAG | 4.43 | 18.84 s | Yes (p = 0.016, d = 0.593) |
| LightRAG | 4.60 | 12.63 s | Yes (p = 0.002, d = 0.780) |
| **Hybrid RAG** | **4.70** | 8.62 s | Yes (p = 0.002, d = 0.800) |
| Blocks-DB | 4.11 | **4.19 s** | No (p = 0.174, d = 0.316) |

Pairwise comparisons among Hybrid RAG, LightRAG, and GraphRAG are not statistically significant (all p > 0.14). Hybrid RAG vs Blocks-DB reaches significance (p = 0.045, d = 0.480).

Blocks-DB scored 2.00 on four graph-specific questions (Q6, Q9, Q11, Q20) where pre-built knowledge graph summaries were the only effective retrieval path. On the remaining 16 questions, Blocks-DB averaged 4.64, statistically indistinguishable from LightRAG and GraphRAG.

> **Note on corpus versions:** A preliminary run used three documents (GraphRAG, LightRAG, Survey papers) with GraphRAG in global search mode. That run produced GraphRAG timeouts (mean latency 237 s, two failures) and a higher Blocks-DB score (4.69) because the four graph-specific questions above were answerable without TagRAG-related context. The results in this repository reflect the final four-document run with GraphRAG in local search mode.

---

## Evaluation Methodology

Each answer is scored 1-5 on four dimensions by GPT-4o-mini acting as judge:

- **Accuracy** — factual correctness against the corpus
- **Relevance** — directly addresses the question
- **Completeness** — covers all key aspects
- **Conciseness** — appropriately scoped, no padding

Overall score = arithmetic mean of the four dimensions. All pairwise comparisons use two-tailed paired t-tests (alpha = 0.05) with Cohen's d effect sizes.

Full scoring rubrics are in the thesis Appendix.

---

## Reproducing the Results

### Prerequisites

```bash
pip install -r requirements.txt
```

Set your OpenAI API key in a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

### Running Each System

**Baseline RAG** (requires corpus as `.txt` files in `microsoft_graphrag/input/`):
```bash
python test_baseline_all.py
```

**LightRAG:**
```bash
python lightrag_system.py
```

**GraphRAG:** Uses [Microsoft GraphRAG](https://github.com/microsoft/graphrag). The index must be pre-built before querying:
```bash
graphrag index --root ./microsoft_graphrag
graphrag query --root ./microsoft_graphrag --method local --query "your question"
```
Results were saved manually to `graphrag_answers.json`.

**Blocks-DB:** Runs on AWS infrastructure (EC2 c7i.xlarge coordinator + Lambda 8 GB + S3, `eu-west-1`) using the [Lithops](https://lithops-cloud.github.io/) serverless framework. Cloud credentials and deployment configuration are not included in this repository. See Barcelona-Pons et al. (2025) for the full system specification and contact the URV CloudLab group for access.
```bash
python blocks_rag.py
```

**Hybrid RAG** (requires `graphrag_answers.json` to exist first):
```bash
python hybrid_rag.py
```

### Scoring All Systems

```bash
python evaluate_all.py
```

This reads each system's answer file and writes scored results to `all_evaluations.json`.

### Generating Figures

```bash
python visualize_results.py
```

---

## Data Format

All answer files share a common structure:

```json
{
  "id": "Q1_diff",
  "question": "What are the main differences between...",
  "type": "comparison",
  "difficulty": "medium",
  "answer": "...",
  "latency": 12.31,
  "tokens_input": 850,
  "tokens_output": 420,
  "cost_usd": 0.000378
}
```

`all_evaluations.json` additionally contains `scores` (per dimension) and `avg_score` fields, organized by system name at the top level.

---

## Citation

If you use this benchmark or code, please cite:

```
Akcakaya, N. A. (2026). Evaluating Graph-Based and Hybrid RAG Architectures
Using a Serverless Vector Database. Master's Thesis, Universitat Rovira i Virgili.
```

---

## License

This repository is made available for academic use. The benchmark questions and evaluation scripts may be reused with attribution. The Blocks-DB system is the property of the URV CloudLab group; see [Barcelona-Pons et al. (2025)](https://dl.acm.org/doi/10.1145/3698820) for details.
