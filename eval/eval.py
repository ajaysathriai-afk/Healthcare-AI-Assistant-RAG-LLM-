import json
import time
import requests
from pathlib import Path
def post_with_retries(url, payload, timeout, retries=3):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return requests.post(url, json=payload, timeout=timeout)
        except Exception as e:
            last_err = e
            print(f"   ⏳ Retry {attempt}/{retries} due to error: {e}")
            time.sleep(2)
    raise last_err

# ========== CONFIG ==========
LOCAL_API = "http://127.0.0.1:8000/ask_llm"
RENDER_API = "https://healthcare-ai-assistant-rag-llm.onrender.com/ask_llm"

# change this:
API_URL = RENDER_API   # <-- start with LOCAL first

TOP_K = 3
TIMEOUT = 180

QUESTIONS_FILE = Path("eval/eval_questions.json")

def normalize_list(x):
    if not x:
        return []
    return [str(i).strip() for i in x if str(i).strip()]

def run_one(question_obj):
    q = question_obj["question"]
    expected_sources = normalize_list(question_obj.get("expected_sources", []))

    payload = {"question": q, "top_k": TOP_K}
    r = post_with_retries(API_URL, payload, TIMEOUT, retries=3)
    r.raise_for_status()

    data = r.json()
    answer = (data.get("answer") or "").strip()
    sources = normalize_list(data.get("sources", []))

    # checks
    has_answer = len(answer) > 0
    has_sources = len(sources) > 0

    matched_expected = 0
    for exp in expected_sources:
        if exp in sources:
            matched_expected += 1

    strict_refusal = "i don't know" in answer.lower() or "i dont know" in answer.lower()

    return {
        "question": q,
        "answer_preview": answer[:120].replace("\n", " "),
        "sources": sources,
        "expected_sources": expected_sources,
        "has_answer": has_answer,
        "has_sources": has_sources,
        "matched_expected": matched_expected,
        "strict_refusal": strict_refusal
    }


def main():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    print(f"\n✅ Running evaluation on API: {API_URL}")
    print(f"✅ Total questions: {len(questions)}")
    print(f"✅ top_k={TOP_K}\n")
    # 🔥 WARM UP Render API (prevents first-question timeout)
    print("🔥 Warming up Render API...")
    try:
        requests.get(API_URL.replace("/ask_llm", "/"), timeout=30)
    except:
        pass
    time.sleep(2)

    results = []
    for i, qobj in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {qobj['question']}")
        try:
            res = run_one(qobj)
            results.append(res)

            ok_source = res["matched_expected"] > 0
            status = "✅ PASS" if ok_source else "❌ FAIL"

            print(f"   {status} | sources={res['sources']}")
            time.sleep(0.5)

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({"question": qobj["question"], "error": str(e)})

    # summary
    total = len(results)
    pass_count = sum(1 for r in results if r.get("matched_expected", 0) > 0)
    fail_count = total - pass_count

    print("\n================ SUMMARY ================")
    print(f"TOTAL: {total}")
    print(f"PASS (expected source found): {pass_count}")
    print(f"FAIL: {fail_count}")
    print("=========================================\n")

    # save detailed results
    out_file = Path("eval/eval_results.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"📄 Saved detailed results to: {out_file}")


if __name__ == "__main__":
    main()

