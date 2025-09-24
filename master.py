#!/usr/bin/env python3
import os
import sys
import re
import json
import time
import random
import argparse
from typing import Optional, Dict, Any, List, Tuple
from contextlib import suppress
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError, APIError

# =========================================
# Config
# =========================================
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-chat-latest")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL")  # optional (proxy / Azure-style endpoint)

# ---------- System Prompts (Questioner: Plan -> Questions) ----------
QUESTIONER_PLANNER_SYSTEM = (
    "You are a precise planning assistant. "
    "Given a Job Description (JD) and a candidate Resume, produce a compact plan in STRICT JSON "
    "for generating a ranked list of the 15 most likely interview questions.\n\n"
    "The JSON MUST have EXACTLY these keys and types:\n"
    "{\n"
    '  "steps": ["...", "..."],\n'
    '  "assumptions": ["...", "..."],\n'
    '  "success_criteria": ["...", "..."]\n'
    "}\n"
    "Rules:\n"
    "- JSON only. No markdown, no comments, no trailing commas.\n"
    "- 3–6 short, actionable steps to derive 15 ranked questions tailored to the JD & Resume.\n"
    "- Keep assumptions and success_criteria concise and specific."
)

QUESTIONER_SYSTEM = (
    "You are an interview-prep assistant. Using the user's JD and Resume plus the provided PLAN, "
    "generate EXACTLY 15 interview questions ranked by likelihood (#1 most likely ... #15). "
    "Output in Markdown with a heading 'Top 15 Likely Interview Questions' and a single numbered list. "
    "Each item MUST be on one line formatted as 'N. <question>' with NO sub-bullets or extra commentary. "
    "Mix role/technical, behavioral, and resume-based follow-ups. Avoid duplicates."
)

# ---------- System Prompts (Answerer: Plan -> Final Answer) ----------
ANSWERER_PLANNER_SYSTEM = (
    "You are a precise planning assistant. "
    "Given a Job Description (JD), a candidate Resume, and ONE selected interview question, "
    "produce a compact plan in STRICT JSON to craft a strong Markdown answer.\n\n"
    "The JSON MUST have EXACTLY these keys and types:\n"
    "{\n"
    '  "steps": ["...", "..."],\n'
    '  "assumptions": ["...", "..."],\n'
    '  "success_criteria": ["...", "..."]\n'
    "}\n"
    "Rules:\n"
    "- JSON only. No markdown, no comments, no trailing commas.\n"
    "- 3–6 short, actionable steps.\n"
    "- Align with the JD & Resume and the selected question."
)

ANSWERER_SYSTEM = (
    "You are an expert interview coach. Using the PLAN plus the JD, Resume, and the selected QUESTION, "
    "write a concise, specific Markdown answer.\n"
    "Structure:\n"
    "## Quick Context (1–3 lines)\n"
    "## Answer (bullet points or short paragraphs)\n"
    "## Evidence from Resume (3–5 bullets that validate the answer)\n"
    "## Next Steps (3–5 bullets tailored to the role)\n"
    "Only answer the selected question; do not include unrelated content."
)

# =========================================
# Utilities
# =========================================
def make_client(api_key: str, base_url: Optional[str]) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

def call_with_retries(fn, retries=3, backoff=1.5):
    attempt = 0
    while True:
        try:
            return fn()
        except (APIConnectionError, APIStatusError, RateLimitError, APIError) as e:
            attempt += 1
            if attempt > retries:
                raise
            sleep_s = (backoff ** attempt) + random.uniform(0, 0.75)
            print(f"Transient error ({e.__class__.__name__}). Retrying in {sleep_s:.1f}s...", file=sys.stderr)
            time.sleep(sleep_s)

def read_file_or_stdin(path: str, label: str) -> str:
    """
    If path == '-', read from stdin (paste). Terminate with a blank line or Ctrl-D (macOS/Linux) / Ctrl-Z+Enter (Windows).
    Otherwise read from file.
    """
    if path == "-":
        print(f"Paste {label} below. Finish with a blank line or Ctrl-D (macOS/Linux) / Ctrl-Z then Enter (Windows):", file=sys.stderr)
        lines: List[str] = []
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                if line.strip() == "" and lines:
                    break
                lines.append(line)
        except KeyboardInterrupt:
            sys.exit("\nAborted.")
        text = "".join(lines).strip()
        if not text:
            sys.exit(f"{label} was empty.")
        return text
    if not os.path.exists(path):
        sys.exit(f"{label} file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        sys.exit(f"{label} file is empty: {path}")
    return text

def parse_json_lenient(txt: str) -> Dict[str, Any]:
    """
    Try to parse possibly malformed JSON:
    - Strip code fences
    - Keep substring from first '{' to last '}'
    """
    s = (txt or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    return json.loads(s)

def repair_json_with_model(client: OpenAI, model: str, broken_text: str, temperature: float, top_p: float, max_tokens: int) -> Dict[str, Any]:
    """
    One-time repair: ask the model to output valid JSON matching the schema.
    """
    repair_system = "You repair malformed JSON. Output VALID JSON only. No comments, no markdown."
    repair_user = f"""Broken JSON-like content:

{broken_text}

Schema keys required:
- steps: array of 3-6 short strings
- assumptions: array of strings
- success_criteria: array of strings

Return ONLY valid JSON conforming to that schema.
"""
    resp = call_with_retries(lambda: client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": repair_system},
            {"role": "user", "content": repair_user},
        ],
        temperature=max(0.0, min(temperature, 0.3)),
        top_p=top_p,
        max_tokens=max(400, min(max_tokens, 800)),
    ))
    fixed = (resp.choices[0].message.content or "").strip()
    return parse_json_lenient(fixed)

def run_planner(client: OpenAI, model: str, system_prompt: str, user_prompt: str,
                temperature: float, top_p: float, max_tokens: int, save_raw_as: Optional[str] = None) -> Dict[str, Any]:
    """
    Generic planner: returns dict with required keys. Attempts one-time repair if JSON breaks.
    """
    resp = call_with_retries(lambda: client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    ))
    raw = (resp.choices[0].message.content or "").strip()

    # Save raw (useful for debugging)
    if save_raw_as:
        with suppress(Exception):
            with open(save_raw_as, "w", encoding="utf-8") as f:
                f.write(raw)

    try:
        plan = parse_json_lenient(raw)
        if not all(k in plan for k in ("steps", "assumptions", "success_criteria")):
            raise ValueError("Missing required keys in plan JSON.")
        return plan
    except Exception as e:
        print("Planner JSON invalid; attempting one-time repair...", file=sys.stderr)
        plan = repair_json_with_model(client, model, raw, temperature, top_p, max_tokens)
        if not all(k in plan for k in ("steps", "assumptions", "success_criteria")):
            raise RuntimeError("Repaired JSON missing required keys.") from e
        return plan

def run_generation_md(client: OpenAI, model: str, system_prompt: str, user_prompt: str,
                      temperature: float, top_p: float, max_tokens: int, stream: bool = False) -> str:
    if stream:
        content_parts: List[str] = []
        stream_resp = call_with_retries(lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            stream=True,
            max_tokens=max_tokens,
        ))
        for chunk in stream_resp:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                content_parts.append(delta)
                print(delta, end="", flush=True)
        print()
        return "".join(content_parts).strip()
    else:
        resp = call_with_retries(lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ))
        return (resp.choices[0].message.content or "").strip()

def extract_ranked_questions(md: str, expected: int = 15) -> List[str]:
    """
    Parse lines like 'N. question' from the Markdown.
    Returns a list of questions indexed 0..N-1.
    """
    qs: List[Tuple[int, str]] = []
    for line in md.splitlines():
        m = re.match(r"\s*(\d{1,2})\.\s+(.*\S)\s*$", line)
        if m:
            idx = int(m.group(1))
            text = m.group(2).strip()
            qs.append((idx, text))
    # Sort by numeric index and filter valid range
    qs = sorted(qs, key=lambda x: x[0])
    ordered = [q for i, q in qs if 1 <= i <= expected]
    # If list shorter than expected, just return what we found
    return ordered

# =========================================
# Prompts
# =========================================
def questioner_planner_user_prompt(jd: str, resume: str) -> str:
    jd = jd[:60_000]
    resume = resume[:60_000]
    return f"""[JOB DESCRIPTION]
{jd}

[RESUME]
{resume}

Your task: Produce ONLY the strict JSON plan with the required keys (steps, assumptions, success_criteria) to create 15 ranked questions.
"""

def questioner_user_prompt(jd: str, resume: str, plan: Dict[str, Any]) -> str:
    jd = jd[:60_000]
    resume = resume[:60_000]
    return f"""[JOB DESCRIPTION]
{jd}

[RESUME]
{resume}

[PLAN]
{json.dumps(plan, ensure_ascii=False, indent=2)}

Instructions:
- Generate EXACTLY 15 interview questions ranked by likelihood (#1 most likely ... #15).
- Output in Markdown:
  # Top 15 Likely Interview Questions
  1. First question
  2. Second question
  ...
  15. Fifteenth question
- Do not add sub-bullets or explanations.
"""

def answerer_planner_user_prompt(jd: str, resume: str, selected_question: str) -> str:
    jd = jd[:60_000]
    resume = resume[:60_000]
    return f"""[JOB DESCRIPTION]
{jd}

[RESUME]
{resume}

[QUESTION]
{selected_question}

Your task: Produce ONLY the strict JSON plan with the required keys (steps, assumptions, success_criteria) to answer the single selected question.
"""

def answerer_user_prompt(jd: str, resume: str, plan: Dict[str, Any], selected_question: str) -> str:
    jd = jd[:60_000]
    resume = resume[:60_000]
    return f"""[JOB DESCRIPTION]
{jd}

[RESUME]
{resume}

[PLAN]
{json.dumps(plan, ensure_ascii=False, indent=2)}

[SELECTED QUESTION]
{selected_question}

Instructions:
- Answer ONLY the selected question.
- Follow the required Markdown structure.
"""

# =========================================
# Main
# =========================================
def main():
    parser = argparse.ArgumentParser(
        description="Interview Helper (Combined): generates 15 ranked questions and answers the one at --rank via two-stage chains."
    )
    parser.add_argument("--jd", required=True, help="Path to Job Description .txt, or '-' to paste")
    parser.add_argument("--resume", required=True, help="Path to Resume .txt, or '-' to paste")
    parser.add_argument("--rank", type=int, required=True, help="Rank of the question to answer (1=most likely ... 15)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature (default: 0.4)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top_p (default: 1.0)")
    parser.add_argument("--max_tokens", type=int, default=1600, help="Max tokens for generation calls (default: 1600)")
    parser.add_argument("--stream", action="store_true", help="Stream tokens for Markdown generations")
    parser.add_argument("--out", default="output", help="Output base name (writes multiple files with this prefix)")
    args = parser.parse_args()

    if not (1 <= args.rank <= 15):
        sys.exit("--rank must be between 1 and 15.")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Missing OPENAI_API_KEY. Add it to .env or export it.")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)

    # Read inputs
    jd_text = read_file_or_stdin(args.jd, "JOB DESCRIPTION")
    resume_text = read_file_or_stdin(args.resume, "RESUME")

    client = make_client(api_key, base_url)

    # -------- Questioner: Chain 1 (Plan) --------
    print("\n[Questioner: Chain 1] Creating plan for generating ranked questions...")
    q_plan = run_planner(
        client=client,
        model=args.model,
        system_prompt=QUESTIONER_PLANNER_SYSTEM,
        user_prompt=questioner_planner_user_prompt(jd_text, resume_text),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max(600, min(args.max_tokens, 1200)),
        save_raw_as=f"{args.out}.questioner_plan_raw.txt",
    )
    with open(f"{args.out}.questioner_plan.json", "w", encoding="utf-8") as f:
        json.dump(q_plan, f, indent=2, ensure_ascii=False)
    print(f"[Questioner: Chain 1] Saved plan → {args.out}.questioner_plan.json")

    # -------- Questioner: Chain 2 (Generate 15 Ranked Questions) --------
    print("[Questioner: Chain 2] Generating 15 ranked interview questions...")
    questions_md = run_generation_md(
        client=client,
        model=args.model,
        system_prompt=QUESTIONER_SYSTEM,
        user_prompt=questioner_user_prompt(jd_text, resume_text, q_plan),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stream=args.stream,
    )
    if not questions_md:
        sys.exit("Model returned empty content for the ranked questions.")
    with open(f"{args.out}.questions.md", "w", encoding="utf-8") as f:
        f.write(questions_md)
    print(f"[Questioner: Chain 2] Saved ranked questions → {args.out}.questions.md")

    # Extract the selected question by rank
    ranked_questions = extract_ranked_questions(questions_md, expected=15)
    if len(ranked_questions) < args.rank:
        print("Could not reliably parse 15 questions from the model output.", file=sys.stderr)
        # Fallback: best-effort naive extraction (take non-empty numbered lines)
        # If still not enough, abort with guidance.
        if len(ranked_questions) == 0:
            sys.exit("Failed to parse any questions from the output. Open the questions MD and select manually.")
    selected_question = ranked_questions[args.rank - 1]
    print(f"\n[Selection] Question #{args.rank}: {selected_question}\n")

    # -------- Answerer: Chain 1 (Plan) --------
    print("[Answerer: Chain 1] Creating plan to answer the selected question...")
    a_plan = run_planner(
        client=client,
        model=args.model,
        system_prompt=ANSWERER_PLANNER_SYSTEM,
        user_prompt=answerer_planner_user_prompt(jd_text, resume_text, selected_question),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max(600, min(args.max_tokens, 1200)),
        save_raw_as=f"{args.out}.answerer_plan_raw.txt",
    )
    with open(f"{args.out}.answerer_plan.json", "w", encoding="utf-8") as f:
        json.dump(a_plan, f, indent=2, ensure_ascii=False)
    print(f"[Answerer: Chain 1] Saved plan → {args.out}.answerer_plan.json")

    # -------- Answerer: Chain 2 (Final Markdown Answer) --------
    print("[Answerer: Chain 2] Generating final Markdown answer...")
    final_md = run_generation_md(
        client=client,
        model=args.model,
        system_prompt=ANSWERER_SYSTEM,
        user_prompt=answerer_user_prompt(jd_text, resume_text, a_plan, selected_question),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stream=args.stream,
    )
    if not final_md:
        sys.exit("Model returned empty content for the final answer.")

    # Display and save
    print("\n========== SELECTED QUESTION ==========\n")
    print(f"{args.rank}. {selected_question}\n")
    print("========== FINAL ANSWER (Markdown) ==========\n")
    print(final_md)
    print("\n=============================================\n")

    with open(f"{args.out}.answer.md", "w", encoding="utf-8") as f:
        f.write(f"# Selected Question\n\n{args.rank}. {selected_question}\n\n# Answer\n\n{final_md}\n")
    print(f"Saved final answer → {args.out}.answer.md")

if __name__ == "__main__":
    main()