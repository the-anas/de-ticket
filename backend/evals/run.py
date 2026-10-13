"""Evaluate document search and the assistant, and write an HTML report.

Run from backend/:

    uv run python -m evals.run retrieval
    uv run python -m evals.run agent
    uv run python -m evals.run all

Compare models by repeating a flag:

    uv run python -m evals.run retrieval --search embedding --search hybrid
    uv run python -m evals.run retrieval \\
        --embedding sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \\
        --embedding intfloat/multilingual-e5-large
    uv run python -m evals.run agent --llm gemini:gemini-flash-lite-latest --llm gemini:gemini-flash-latest

The retrieval eval is local and free. The agent eval calls the LLM (roughly 40
requests per model for the full set) and the live timetable. Defaults come
from .env. Results go to evals/results/.
"""

import argparse
import asyncio
import json
import time
from functools import lru_cache
from datetime import date
from pathlib import Path

from app.config import get_settings
from app.deps import get_transit
from app.llm.agent import run_agent
from app.llm.tools import ToolRunner
from app.rag.embedder import FastEmbedEmbedder
from app.rag.ingest import make_store
from app.rag.reranker import FastEmbedReranker

from .checks import ToolUse, Trace, dumps, grade, now_stamp
from .report import write_report

HERE = Path(__file__).parent
RESULTS = HERE / "results"
TOP_K = 5  # how many sections the assistant is given per search


# ---------- retrieval ----------


SEARCH_LABELS = {
    "rerank": "meaning + keywords, reranked",
    "hybrid": "meaning + keywords",
    "embedding": "meaning only",
    "keyword": "keywords only",
}


@lru_cache
def reranker(name: str) -> FastEmbedReranker:
    return FastEmbedReranker(name)


def eval_retrieval(model: str, mode: str, reranker_model: str = "") -> dict:
    # indexed exactly as the app does
    store = make_store(FastEmbedEmbedder(model), reranker=reranker(reranker_model) if mode == "rerank" else None)
    cases = json.loads((HERE / "retrieval_cases.json").read_text(encoding="utf-8"))
    rows = []
    seconds: list[float] = []
    for case in cases:
        started = time.monotonic()
        ranked = [(h.chunk.section, h.score) for h in store.rank(case["query"], mode)]
        seconds.append(time.monotonic() - started)
        # a section no keyword matched is absent from a keyword-only ranking
        found = [n for n, (section, _) in enumerate(ranked, 1) if section in case["expect"]]
        rank = found[0] if found else len({c.key for c in store.chunks}) + 1
        rows.append(
            {
                "group": case["group"],
                "query": case["query"],
                "expect": case["expect"],
                "rank": rank,
                "top": [{"section": sec, "score": s} for sec, s in ranked[:TOP_K]],
                "note": case.get("note", ""),
            }
        )

    def summarise(subset: list[dict]) -> dict:
        n = len(subset)
        return {
            "n": n,
            "top1": sum(r["rank"] == 1 for r in subset) / n,
            "top5": sum(r["rank"] <= TOP_K for r in subset) / n,
            "mrr": sum(1 / r["rank"] for r in subset) / n,
        }

    groups = sorted({r["group"] for r in rows}, key=[r["group"] for r in rows].index)
    return {
        "model": f"{model.split('/')[-1]} ({SEARCH_LABELS[mode]}"
        + (f" by {reranker_model.split('/')[-1]})" if mode == "rerank" else ")"),
        # the first query is left out: it includes loading the model
        "ms_per_query": round(1000 * sum(seconds[1:]) / max(1, len(seconds) - 1)),
        "sections": len({c.key for c in store.chunks}),
        "overall": summarise(rows),
        "groups": {g: summarise([r for r in rows if r["group"] == g]) for g in groups},
        "rows": rows,
    }


# ---------- agent ----------


class RecordingTools(ToolRunner):
    """Runs the real tools and keeps their results for grading."""

    def __init__(self, transit):
        super().__init__(transit)
        self.used: list[ToolUse] = []
        self.retrieved: list[str] = []

    async def run(self, name: str, args: dict):
        outcome = await super().run(name, args)
        self.used.append(ToolUse(name, args, outcome.content, outcome.is_error))
        for event in outcome.events:
            if event["type"] == "sources":
                self.retrieved += [s["section"] for s in event["sources"]]
        return outcome


def make_llm(spec: str):
    provider, _, model = spec.partition(":")
    key = get_settings().llm_api_key
    if provider == "gemini":
        from app.llm.gemini_provider import GeminiProvider

        return GeminiProvider(model, key)
    if provider == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model, key)
    if provider == "openai":
        from app.llm.openai_compat_provider import OpenAICompatProvider

        return OpenAICompatProvider(model, key, get_settings().llm_base_url)
    raise SystemExit(f"Unknown provider in --llm {spec!r}; use provider:model, e.g. gemini:gemini-flash-lite-latest")


async def run_case(llm, case: dict) -> Trace:
    tools = RecordingTools(get_transit())
    trace = Trace(prompt=case["prompt"])
    started = time.monotonic()
    history = [*case.get("history", []), {"role": "user", "content": case["prompt"]}]
    try:
        async for event in run_agent(llm, tools, history, case.get("origin")):
            if event["type"] == "text":
                trace.answer += event["text"]
            elif event["type"] == "sources":
                trace.cited += [s["section"] for s in event["sources"]]
            elif event["type"] == "error":
                trace.errors.append(event["message"])
    except Exception as e:  # a crash is a result too
        trace.errors.append(f"{type(e).__name__}: {e}")
    trace.seconds = time.monotonic() - started
    trace.tools = tools.used
    trace.sections = tools.retrieved
    trace.answer = trace.answer.strip()
    return trace


async def eval_agent(spec: str, only: list[str] | None) -> dict:
    llm = make_llm(spec)
    cases = json.loads((HERE / "agent_cases.json").read_text(encoding="utf-8"))
    if only:
        cases = [c for c in cases if c["id"] in only]
    today = date.today()
    rows = []
    for case in cases:
        print(f"  {spec}  {case['id']} …", flush=True)
        trace = await run_case(llm, case)
        results = grade(trace, case["checks"], today)
        rows.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "origin": (case.get("origin") or {}).get("name"),
                "answer": trace.answer,
                "seconds": round(trace.seconds, 1),
                "tools": [{"name": t.name, "args": t.args, "is_error": t.is_error} for t in trace.tools],
                "sections": trace.sections,
                "cited": trace.cited,
                "checks": {name: {"passed": ok, "detail": detail} for name, (ok, detail) in results.items()},
            }
        )
    per_check: dict[str, list[bool]] = {}
    for row in rows:
        for name, result in row["checks"].items():
            per_check.setdefault(name, []).append(result["passed"])
    total = [ok for oks in per_check.values() for ok in oks]
    return {
        "model": spec,
        "cases": len(rows),
        "checks_passed": sum(total),
        "checks_total": len(total),
        "cases_fully_passed": sum(all(c["passed"] for c in r["checks"].values()) for r in rows),
        "per_check": {name: {"passed": sum(oks), "total": len(oks)} for name, oks in per_check.items()},
        "rows": rows,
    }


# ---------- entry point ----------


def print_retrieval(result: dict) -> None:
    o = result["overall"]
    print(f"\n{result['model']}\n  {o['n']} queries: right section first {o['top1']:.0%}, in top {TOP_K} {o['top5']:.0%}, "
          f"MRR {o['mrr']:.2f}, {result['ms_per_query']} ms per query")
    for group, g in result["groups"].items():
        print(f"    {group:14} n={g['n']:<2} first {g['top1']:>4.0%}   top {TOP_K} {g['top5']:>4.0%}")


def print_agent(result: dict) -> None:
    print(f"\n{result['model']}\n  checks passed {result['checks_passed']}/{result['checks_total']}, "
          f"cases with everything right {result['cases_fully_passed']}/{result['cases']}")
    for name, c in result["per_check"].items():
        print(f"    {name:18} {c['passed']}/{c['total']}")
    for row in result["rows"]:
        failed = [f"{n}: {c['detail']}" for n, c in row["checks"].items() if not c["passed"]]
        if failed:
            print(f"    FAIL {row['id']}: " + " | ".join(failed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["retrieval", "agent", "all"])
    parser.add_argument("--embedding", action="append", help="embedding model to test (repeatable)")
    parser.add_argument("--search", action="append", choices=list(SEARCH_LABELS),
                        help="search method to test (repeatable; default: what the app uses, which is "
                             "rerank when RERANKER_MODEL is set and hybrid otherwise)")
    parser.add_argument("--reranker", action="append", help="reranker model to test with --search rerank (repeatable)")
    parser.add_argument("--llm", action="append", help="provider:model to test (repeatable)")
    parser.add_argument("--case", action="append", help="only run these agent case ids (repeatable)")
    args = parser.parse_args()
    settings = get_settings()

    output: dict = {"date": date.today().isoformat(), "retrieval": [], "agent": []}
    if args.what in ("retrieval", "all"):
        for model in args.embedding or [settings.embedding_model]:
            for mode in args.search or ["rerank" if settings.reranker_model else "hybrid"]:
                rerankers = (args.reranker or [settings.reranker_model]) if mode == "rerank" else [""]
                if rerankers == [""] and mode == "rerank":
                    raise SystemExit("--search rerank needs RERANKER_MODEL in .env or --reranker <model>")
                for reranker_model in rerankers:
                    result = eval_retrieval(model, mode, reranker_model)
                    output["retrieval"].append(result)
                    print_retrieval(result)
    if args.what in ("agent", "all"):
        for spec in args.llm or [f"{settings.llm_provider}:{settings.llm_model}"]:
            result = asyncio.run(eval_agent(spec, args.case))
            output["agent"].append(result)
            print_agent(result)

    RESULTS.mkdir(exist_ok=True)
    stamp = now_stamp()
    (RESULTS / f"{stamp}.json").write_text(dumps(output), encoding="utf-8")
    report = write_report(output, RESULTS / f"{stamp}.html")
    # a stable name for "the most recent run"
    (RESULTS / "latest.html").write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nReport: {RESULTS / 'latest.html'}")


if __name__ == "__main__":
    main()
