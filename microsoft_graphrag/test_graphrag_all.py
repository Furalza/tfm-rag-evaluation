"""
GraphRAG - Benchmark Runner v2
Folder  : microsoft_graphrag/
Run from: TFM_GraphRAG/ root folder
Command : .venv/Scripts/python.exe microsoft_graphrag/test_graphrag_all.py

NOTE: Index already built. Only runs queries.
      Uses LOCAL search (thesis Section 3.1.2).
      Each query ~2-4 min. Total ~80 min.

Output:
  - microsoft_graphrag/graphrag_answers.json
  - results/graphrag_answers.json
"""

import subprocess, json, sys, time, importlib.util
from pathlib import Path

THIS_DIR   = Path(__file__).resolve().parent
ROOT       = THIS_DIR.parent
LOCAL_OUT  = THIS_DIR
SHARED_OUT = ROOT / "results"
SHARED_OUT.mkdir(exist_ok=True)

# Load QUESTIONS directly from file path
_spec = importlib.util.spec_from_file_location("_q", ROOT / "questions" / "questions.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
QUESTIONS = _mod.QUESTIONS
print(f"Loaded {len(QUESTIONS)} questions.")

n = len(QUESTIONS)
print("=" * 60)
print(f"GRAPHRAG - {n} QUESTIONS (local search)")
print("=" * 60)
print("Estimated time: ~80 minutes. Do not close terminal.\n")

items       = []
start_total = time.time()

for i, q in enumerate(QUESTIONS):
    print(f"\n[{i+1}/{n}] {q['id']}: {q['text'][:70]}...")

    cmd = [
        sys.executable, "-m", "graphrag", "query",
        "--root", str(THIS_DIR),
        "--method", "local",
        q['text']
    ]

    try:
        start = time.time()
        proc  = subprocess.run(cmd, capture_output=True, timeout=300)
        latency = round(time.time() - start, 2)

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")

        lines = stdout.split('\n')
        answer_lines = []
        capture = False
        for line in lines:
            if 'SUCCESS:' in line or 'Local Search Response:' in line:
                capture = True
                if 'SUCCESS:' in line:
                    after = line.split('SUCCESS:', 1)[1].strip()
                    if after:
                        answer_lines.append(after)
                continue
            if capture:
                answer_lines.append(line)

        answer = '\n'.join(answer_lines).strip() if answer_lines else stdout.strip()

        if proc.returncode != 0:
            print(f"  ERROR (rc={proc.returncode}): {stderr[:200]}")
            answer = f"ERROR: {stderr[:500]}"
        else:
            elapsed = round((time.time() - start_total) / 60, 1)
            print(f"  Latency: {latency}s | Elapsed: {elapsed} min")
            print(f"  Preview: {answer[:150]}...")

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (300s)!")
        answer  = "TIMEOUT"
        latency = 300.0
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        answer  = f"EXCEPTION: {str(e)}"
        latency = 0.0

    items.append({
        "id":         q['id'],
        "question":   q['text'],
        "type":       q['type'],
        "difficulty": q['difficulty'],
        "system":     "GraphRAG",
        "latency":    latency,
        "answer":     answer,
    })

    # Save after every question in case of crash
    for save_path in [LOCAL_OUT / "graphrag_answers.json",
                      SHARED_OUT / "graphrag_answers.json"]:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)

avg_lat  = sum(r['latency'] for r in items) / len(items)
timeouts = sum(1 for r in items if r['answer'] == "TIMEOUT")
errors   = sum(1 for r in items if str(r['answer']).startswith(("ERROR", "EXCEPTION")))
total_min = round((time.time() - start_total) / 60, 1)

print(f"\n{'='*60}")
print(f"GRAPHRAG - DONE")
print(f"  Questions    : {n}")
print(f"  Avg latency  : {avg_lat:.2f}s")
print(f"  Timeouts     : {timeouts}")
print(f"  Errors       : {errors}")
print(f"  Total time   : {total_min} min")
print(f"{'='*60}")
print("Next: .venv/Scripts/python.exe evaluate_all.py")