"""
LightRAG - Benchmark Runner v2
Folder  : lightrag_rag/
Run from: TFM_GraphRAG/ root folder
Command : .venv/Scripts/python.exe lightrag_rag/lightrag_system.py

WARNING: Deletes old lightrag_data index and rebuilds with 4 documents.
         Index build takes ~15-30 minutes.
"""

import os, json, sys, time, asyncio, shutil, importlib.util
from pathlib import Path
from dotenv import load_dotenv

THIS_DIR    = Path(__file__).resolve().parent
ROOT        = THIS_DIR.parent
DATA_PATH   = ROOT / "microsoft_graphrag" / "input"
WORKING_DIR = THIS_DIR / "lightrag_data"
LOCAL_OUT   = THIS_DIR
SHARED_OUT  = ROOT / "results"

load_dotenv(ROOT / ".env")
SHARED_OUT.mkdir(exist_ok=True)

# Load QUESTIONS directly from file path — avoids circular import
_spec = importlib.util.spec_from_file_location("_q", ROOT / "questions" / "questions.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
QUESTIONS = _mod.QUESTIONS
print(f"Loaded {len(QUESTIONS)} questions.")

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
except ImportError:
    print("ERROR: LightRAG not installed. Run: pip install lightrag-hku")
    exit(1)

PRICE_INPUT_PER_1K  = 0.000150
PRICE_OUTPUT_PER_1K = 0.000600


async def main():
    # Delete old index, rebuild with 4 documents
    if WORKING_DIR.exists():
        print(f"Deleting old index: {WORKING_DIR}")
        shutil.rmtree(WORKING_DIR)
    WORKING_DIR.mkdir(parents=True)

    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
        tiktoken_model_name="gpt-3.5-turbo",
    )
    await rag.initialize_storages()

    # Load 4 txt files
    print("\nLoading documents...")
    all_text = ""
    for file in sorted(DATA_PATH.glob("*.txt")):
        with open(file, encoding="utf-8") as f:
            content = f.read()
        all_text += f"\n\n=== {file.name} ===\n\n" + content
        print(f"  Loaded: {file.name} ({len(content)//1000}K chars)")

    print(f"\nTotal: {len(all_text)//1000}K chars")
    print("Building LightRAG index (~15-30 min)...")
    index_start = time.time()
    await rag.ainsert(all_text)
    index_time = round(time.time() - index_start, 1)
    print(f"Index built in {index_time}s ({index_time/60:.1f} min)")

    # Run benchmark
    n = len(QUESTIONS)
    print(f"\n{'='*60}")
    print(f"LIGHTRAG - {n} QUESTIONS (hybrid mode)")
    print(f"{'='*60}")

    items      = []
    total_cost = 0.0

    for i, q in enumerate(QUESTIONS):
        print(f"\n[{i+1}/{n}] {q['id']}: {q['text'][:70]}...")

        try:
            start      = time.time()
            answer     = await rag.aquery(q['text'], param=QueryParam(mode="hybrid"))
            latency    = round(time.time() - start, 2)
            answer_str = str(answer)
        except Exception as e:
            print(f"  ERROR: {e}")
            answer_str = f"ERROR: {str(e)}"
            latency    = 0.0

        tokens_in  = len(q['text'].split()) * 4 // 3
        tokens_out = len(answer_str.split()) * 4 // 3
        cost = round(
            (tokens_in  / 1000 * PRICE_INPUT_PER_1K) +
            (tokens_out / 1000 * PRICE_OUTPUT_PER_1K), 6
        )
        total_cost += cost

        print(f"  Latency : {latency}s | Cost: ${cost:.6f} (est)")
        print(f"  Preview : {answer_str[:150]}...")

        items.append({
            "id":             q['id'],
            "question":       q['text'],
            "type":           q['type'],
            "difficulty":     q['difficulty'],
            "system":         "LightRAG",
            "latency":        latency,
            "answer":         answer_str,
            "tokens_in_est":  tokens_in,
            "tokens_out_est": tokens_out,
            "cost_usd_est":   cost,
        })

    # Save to both locations
    for save_path in [LOCAL_OUT / "lightrag_answers.json",
                      SHARED_OUT / "lightrag_answers.json"]:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"Saved: {save_path}")

    avg_lat = sum(r['latency'] for r in items) / len(items)
    print(f"\n{'='*60}")
    print(f"LIGHTRAG - DONE")
    print(f"  Questions    : {n}")
    print(f"  Index time   : {index_time}s ({index_time/60:.1f} min)")
    print(f"  Avg latency  : {avg_lat:.2f}s")
    print(f"  Total cost   : ${total_cost:.4f} (estimated)")
    print(f"{'='*60}")

    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())