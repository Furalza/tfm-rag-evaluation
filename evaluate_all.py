"""
Evaluate RAG Systems - Windows
================================
Folder  : TFM_GraphRAG/
Command : .venv/Scripts/python.exe evaluate_windows.py

Evaluates: Baseline RAG, LightRAG, GraphRAG
Output  : results/windows_evaluations.json
"""

import json, sys, time, importlib.util
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from openai import OpenAI
client = OpenAI()

EVAL_SYSTEM = """You are an expert evaluator of RAG (Retrieval-Augmented Generation) system answers.
Score the given answer on a 1-5 integer scale for each of the four criteria below.
Return ONLY valid JSON with no markdown, no explanation, no extra text.
Format: {"accuracy": X, "relevance": X, "completeness": X, "conciseness": X}

Scoring rubrics:

ACCURACY (Is factual content correct and consistent with the source material?):
  5 = Fully correct. All factual claims are accurate and grounded in the corpus.
  4 = Mostly correct. One minor factual error or slight imprecision.
  3 = Partially correct. Some accurate claims alongside notable errors.
  2 = Mostly incorrect. Major factual errors or significant unsupported claims.
  1 = Incorrect or fabricated. Answer contains predominantly false content.

RELEVANCE (Does the answer directly address the question?):
  5 = Fully on-topic. Directly and completely addresses what was asked.
  4 = Mostly relevant. Minor tangential content that does not impede the answer.
  3 = Partially relevant. Addresses the question but includes significant off-topic content.
  2 = Mostly off-topic. Only loosely related to the question.
  1 = Not relevant. Does not address the question asked.

COMPLETENESS (Does the answer cover all key aspects?):
  5 = Thorough. All key aspects addressed with appropriate depth.
  4 = Mostly complete. One minor aspect missing or insufficiently developed.
  3 = Partially complete. Several relevant aspects missing or superficial.
  2 = Incomplete. Significant portions of the expected answer absent.
  1 = Superficial. Answer barely scratches the surface.

CONCISENESS (Is the answer appropriately scoped?):
  5 = Well-scoped. Focused with no redundant or tangential content.
  4 = Mostly concise. Minor repetition that does not impede clarity.
  3 = Moderately verbose. Noticeable repetition or padding that could be removed.
  2 = Verbose. Substantially longer than necessary with significant redundancy.
  1 = Excessively verbose. Padded to the point of obscuring relevant content."""


def evaluate(question, answer, retries=3):
    if not answer or answer in ("TIMEOUT", "") or str(answer).startswith(("ERROR:", "EXCEPTION:")):
        return {"accuracy": 1, "relevance": 1, "completeness": 1, "conciseness": 1}

    prompt = (
        f"Question: {question}\n\n"
        f"Answer:\n{str(answer)[:3000]}\n\n"
        "Score this answer using the rubric. Return only JSON."
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=100,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            scores = json.loads(raw)
            assert all(k in scores for k in ["accuracy","relevance","completeness","conciseness"])
            assert all(1 <= int(v) <= 5 for v in scores.values())
            return {k: int(v) for k, v in scores.items()}
        except Exception as e:
            print(f"    [attempt {attempt+1}] {e}")
            time.sleep(2)
    return {"accuracy": 1, "relevance": 1, "completeness": 1, "conciseness": 1}


def get_field(item, *keys):
    for k in keys:
        if k in item:
            return item[k]
    return ""


FILES = {
    "Baseline RAG": ROOT / "baseline_rag"      / "baseline_answers.json",
    "LightRAG":     ROOT / "lightrag_rag"       / "lightrag_answers.json",
    "GraphRAG":     ROOT / "microsoft_graphrag" / "graphrag_answers.json",
}

OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

all_results = {}

for system, path in FILES.items():
    if not path.exists():
        print(f"[SKIP] {system} — not found: {path}")
        continue

    print(f"\n{'='*55}")
    print(f"Evaluating: {system}")
    print(f"{'='*55}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    evaluated = []
    for item in data:
        qid      = get_field(item, 'id', 'question_id')
        question = get_field(item, 'question')
        answer   = get_field(item, 'answer')
        latency  = get_field(item, 'latency', 'latency_seconds')

        print(f"  [{qid}] scoring...", end=" ", flush=True)
        scores = evaluate(question, answer)
        avg    = round(sum(scores.values()) / 4, 2)
        print(f"avg={avg} | {scores}")

        evaluated.append({
            "id":         qid,
            "question":   question,
            "type":       get_field(item, 'type'),
            "difficulty": get_field(item, 'difficulty'),
            "system":     system,
            "latency":    float(latency) if latency else 0.0,
            "answer":     answer,
            "scores":     scores,
            "avg_score":  avg,
        })
        time.sleep(0.3)

    overall = round(sum(r['avg_score'] for r in evaluated) / len(evaluated), 4)
    avg_lat = round(sum(r['latency']   for r in evaluated) / len(evaluated), 2)

    all_results[system] = {
        "items":       evaluated,
        "overall":     overall,
        "avg_latency": avg_lat,
    }
    print(f"  -> Overall: {overall}/5 | Avg latency: {avg_lat}s")

out = OUTPUT_DIR / "windows_evaluations.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*55}")
print(f"DONE — Windows Systems")
print(f"{'='*55}")
print(f"{'System':<15} {'Score':>8} {'Latency':>10} {'n':>4}")
print("-"*55)
for sys_name, res in all_results.items():
    print(f"{sys_name:<15} {res['overall']:>8.4f} {res['avg_latency']:>9.2f}s {len(res['items']):>4}")
print(f"{'='*55}")
print(f"\nSaved: {out}")