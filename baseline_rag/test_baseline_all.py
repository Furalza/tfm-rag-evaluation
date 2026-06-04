"""
Baseline RAG - Benchmark Runner v2
Folder  : baseline_rag/
Run from: TFM_GraphRAG/ root folder
Command : .venv/Scripts/python.exe baseline_rag/test_baseline_all.py
"""

import json, sys, time, importlib.util
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).resolve().parent
ROOT       = THIS_DIR.parent
DATA_PATH  = ROOT / "microsoft_graphrag" / "input"
LOCAL_OUT  = THIS_DIR
SHARED_OUT = ROOT / "results"

load_dotenv(ROOT / ".env")
SHARED_OUT.mkdir(exist_ok=True)

# ── Load QUESTIONS directly from file (avoids circular import) ────────────────
_spec = importlib.util.spec_from_file_location("_questions", ROOT / "questions" / "questions.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
QUESTIONS = _mod.QUESTIONS
print(f"Loaded {len(QUESTIONS)} questions.")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import get_openai_callback

PRICE_INPUT_PER_1K  = 0.000150
PRICE_OUTPUT_PER_1K = 0.000600

# ── Load documents ────────────────────────────────────────────────────────────
print("Loading documents...")
documents = []
for file in sorted(DATA_PATH.glob("*.txt")):
    loader = TextLoader(str(file), encoding="utf-8")
    documents.extend(loader.load())
    print(f"  Loaded: {file.name} ({file.stat().st_size // 1024} KB)")

print(f"Total: {len(documents)} documents")

if len(documents) != 4:
    print(f"WARNING: Expected 4 documents, got {len(documents)}!")
    print(f"Check: {DATA_PATH}")

# ── Build vector store ────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks")

print("Building vector store (text-embedding-ada-002)...")
embeddings  = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = Chroma.from_documents(texts, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 5})
print("Vector store ready.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "You are an expert assistant. "
    "Answer the question based on the following context. "
    "If the context does not contain enough information, say so clearly.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# ── Run benchmark ─────────────────────────────────────────────────────────────
n = len(QUESTIONS)
print(f"\n{'='*60}")
print(f"BASELINE RAG - {n} QUESTIONS")
print(f"{'='*60}")

items = []
total_tokens_in = total_tokens_out = 0
total_cost = 0.0

for i, q in enumerate(QUESTIONS):
    print(f"\n[{i+1}/{n}] {q['id']}: {q['text'][:70]}...")

    start = time.time()
    with get_openai_callback() as cb:
        answer = chain.invoke(q['text'])
    latency = round(time.time() - start, 2)

    tokens_in  = cb.prompt_tokens
    tokens_out = cb.completion_tokens
    cost = round(
        (tokens_in  / 1000 * PRICE_INPUT_PER_1K) +
        (tokens_out / 1000 * PRICE_OUTPUT_PER_1K), 6
    )
    total_tokens_in  += tokens_in
    total_tokens_out += tokens_out
    total_cost       += cost

    print(f"  Latency : {latency}s")
    print(f"  Tokens  : {tokens_in} in / {tokens_out} out | Cost: ${cost:.6f}")
    print(f"  Preview : {answer[:150]}...")

    items.append({
        "id":         q['id'],
        "question":   q['text'],
        "type":       q['type'],
        "difficulty": q['difficulty'],
        "system":     "Baseline RAG",
        "latency":    latency,
        "answer":     answer,
        "tokens_in":  tokens_in,
        "tokens_out": tokens_out,
        "cost_usd":   cost,
    })

# ── Save to both locations ────────────────────────────────────────────────────
for save_path in [LOCAL_OUT / "baseline_answers.json",
                  SHARED_OUT / "baseline_answers.json"]:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"Saved: {save_path}")

avg_lat = sum(r['latency'] for r in items) / len(items)
print(f"\n{'='*60}")
print(f"BASELINE RAG - DONE")
print(f"  Questions       : {n}")
print(f"  Avg latency     : {avg_lat:.2f}s")
print(f"  Total tokens in : {total_tokens_in}")
print(f"  Total tokens out: {total_tokens_out}")
print(f"  Total cost      : ${total_cost:.4f}")
print(f"{'='*60}")
print("Next: .venv/Scripts/python.exe lightrag_rag/lightrag_system.py")