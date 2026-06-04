"""
Blocks-DB RAG - Benchmark Runner v2
=====================================
Run from: ~/rag_benchmark/
Command : python blocks_rag.py

Reads txt files from: ~/rag_benchmark/pdfs/
Saves to: ~/rag_benchmark/blocks_rag_results.json

Thesis settings (Section 3.1.4):
  - text-embedding-3-small
  - chunk_size=1000, overlap=200
  - top-5 retrieval
  - LLM: gpt-4o-mini, temperature=0
"""

import os, json, time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/rag_benchmark/.env"))
client = OpenAI()

PRICE_INPUT_PER_1K  = 0.000150
PRICE_OUTPUT_PER_1K = 0.000600

QUESTIONS = [
    {"id": "Q1_diff",        "text": "What are the main differences between Naive RAG, GraphRAG, and TagRAG?",                                              "type": "comparison",  "difficulty": "medium"},
    {"id": "Q2_limits",      "text": "What are the key limitations of Naive RAG systems?",                                                                  "type": "factual",     "difficulty": "easy"},
    {"id": "Q3_improve",     "text": "How does GraphRAG improve upon standard RAG for complex reasoning tasks?",                                             "type": "explanation", "difficulty": "medium"},
    {"id": "Q4_efficiency",  "text": "Compare the efficiency of GraphRAG and TagRAG in terms of construction and retrieval time.",                           "type": "comparison",  "difficulty": "hard"},
    {"id": "Q5_usecases",    "text": "In which cases is standard RAG still useful compared to graph-based approaches?",                                      "type": "reasoning",   "difficulty": "hard"},
    {"id": "Q6_community",   "text": "What is the role of community detection in GraphRAG's indexing process?",                                             "type": "factual",     "difficulty": "medium"},
    {"id": "Q7_multihop",    "text": "How do graph-based RAG systems handle multi-hop reasoning better than vector-based RAG?",                             "type": "explanation", "difficulty": "hard"},
    {"id": "Q8_tagchain",    "text": "What are tag chains in TagRAG and how do they enable hierarchical knowledge retrieval?",                               "type": "factual",     "difficulty": "medium"},
    {"id": "Q9_cost",        "text": "What are the computational costs of building a GraphRAG index compared to a simple vector store?",                    "type": "comparison",  "difficulty": "medium"},
    {"id": "Q10_hybrid",     "text": "How could combining vector databases and graph databases create a more effective hybrid RAG system?",                  "type": "reasoning",   "difficulty": "hard"},
    {"id": "Q11_entities",   "text": "What types of entities and relationships does GraphRAG extract during the indexing phase?",                            "type": "factual",     "difficulty": "easy"},
    {"id": "Q12_embedding",  "text": "What role do text embeddings play in vector-based RAG retrieval?",                                                    "type": "factual",     "difficulty": "easy"},
    {"id": "Q13_global",     "text": "How does GraphRAG's global search differ from local search in terms of answer generation?",                           "type": "explanation", "difficulty": "medium"},
    {"id": "Q14_lightrag",   "text": "How does LightRAG reduce the computational overhead of graph-based retrieval compared to GraphRAG?",                  "type": "explanation", "difficulty": "medium"},
    {"id": "Q15_quality",    "text": "Compare the answer quality of graph-based RAG systems versus naive RAG on complex, multi-document queries.",          "type": "comparison",  "difficulty": "hard"},
    {"id": "Q16_scalability","text": "How do GraphRAG and TagRAG differ in their ability to scale to large document corpora?",                              "type": "comparison",  "difficulty": "hard"},
    {"id": "Q17_hallucination","text":"Why might graph-based RAG systems produce fewer hallucinations than naive RAG systems?",                             "type": "reasoning",   "difficulty": "hard"},
    {"id": "Q18_tradeoff",   "text": "When would a practitioner choose LightRAG over GraphRAG despite GraphRAG's higher answer quality?",                  "type": "reasoning",   "difficulty": "hard"},
    {"id": "Q19_chunking",   "text": "How does chunk size selection affect retrieval precision and recall in naive RAG systems?",                            "type": "reasoning",   "difficulty": "medium"},
    {"id": "Q20_future",     "text": "What are the most promising future research directions for improving RAG systems beyond graph-based approaches?",      "type": "reasoning",   "difficulty": "hard"},
]

# ── Load txt files ────────────────────────────────────────────────────────────
def load_texts(txt_dir):
    text = ""
    for fname in sorted(os.listdir(txt_dir)):
        if fname.endswith(".txt"):
            with open(os.path.join(txt_dir, fname), encoding="utf-8") as f:
                content = f.read()
            text += f"\n\n=== {fname} ===\n\n" + content
            print(f"  Loaded: {fname} ({len(content)//1000}K chars)")
    return text

# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# ── Embeddings ────────────────────────────────────────────────────────────────
def get_embedding(text):
    resp = client.embeddings.create(
        input=text[:8000],
        model="text-embedding-3-small"
    )
    return np.array(resp.data[0].embedding)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_top_chunks(q_emb, chunk_embeddings, chunks, top_k=5):
    sims = [cosine_sim(q_emb, ce) for ce in chunk_embeddings]
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return "\n\n".join([chunks[i] for i in top_idx])

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Blocks-DB RAG Benchmark ===\n")

    txt_dir = os.path.expanduser("~/rag_benchmark/pdfs")
    print("Loading documents...")
    text   = load_texts(txt_dir)
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    print(f"Total chunks: {len(chunks)}")

    print("\nEmbedding chunks (text-embedding-3-small)...")
    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        chunk_embeddings.append(get_embedding(chunk))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(chunks)} chunks embedded")
    print(f"All {len(chunks)} chunks embedded.")

    n = len(QUESTIONS)
    print(f"\n{'='*60}")
    print(f"BLOCKS-DB RAG - {n} QUESTIONS")
    print(f"{'='*60}")

    items      = []
    total_cost = 0.0

    for i, q in enumerate(QUESTIONS):
        print(f"\n[{i+1}/{n}] {q['id']}: {q['text'][:70]}...")

        start = time.time()

        q_emb   = get_embedding(q['text'])
        context = retrieve_top_chunks(q_emb, chunk_embeddings, chunks, top_k=5)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are an expert assistant. Answer the question based on the following context. If the context does not contain enough information, say so clearly."},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {q['text']}\n\nAnswer:"}
            ]
        )

        latency = round(time.time() - start, 2)
        answer  = response.choices[0].message.content

        tokens_in  = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        cost = round(
            (tokens_in  / 1000 * PRICE_INPUT_PER_1K) +
            (tokens_out / 1000 * PRICE_OUTPUT_PER_1K), 6
        )
        total_cost += cost

        print(f"  Latency : {latency}s | Tokens: {tokens_in}/{tokens_out} | Cost: ${cost:.6f}")
        print(f"  Preview : {answer[:150]}...")

        items.append({
            "id":         q['id'],
            "question":   q['text'],
            "type":       q['type'],
            "difficulty": q['difficulty'],
            "system":     "Blocks-DB",
            "latency":    latency,
            "answer":     answer,
            "tokens_in":  tokens_in,
            "tokens_out": tokens_out,
            "cost_usd":   cost,
        })

        # Save after every question
        with open(os.path.expanduser("~/rag_benchmark/blocks_rag_results.json"), "w") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)

    avg_lat = sum(r['latency'] for r in items) / len(items)
    print(f"\n{'='*60}")
    print(f"BLOCKS-DB RAG - DONE")
    print(f"  Questions    : {n}")
    print(f"  Avg latency  : {avg_lat:.2f}s")
    print(f"  Total cost   : ${total_cost:.4f}")
    print(f"  Saved to     : ~/rag_benchmark/blocks_rag_results.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
