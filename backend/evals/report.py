"""Self-contained HTML report for an eval run (no scripts, no external files)."""

from html import escape
from pathlib import Path

from .checks import DESCRIPTIONS

CSS = """
:root { --bg:#fff; --fg:#1a1f2b; --muted:#5d6678; --line:#d9dee7; --soft:#f4f6f9; --good:#136c3a; --bad:#a3271f; --badbg:#fdecea; --goodbg:#e8f5ec; }
@media (prefers-color-scheme: dark) { :root { --bg:#0e131d; --fg:#e8ecf3; --muted:#98a3b6; --line:#2a3242; --soft:#161d2a; --good:#6fd39a; --bad:#ff9a92; --badbg:#3a1a18; --goodbg:#12301f; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width:1100px; margin:0 auto; padding:28px 16px 80px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:19px; margin:44px 0 6px; padding-top:20px; border-top:1px solid var(--line); } h3 { font-size:15px; margin:26px 0 8px; }
p { margin:6px 0 12px; } .muted { color:var(--muted); } .small { font-size:13px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; margin:8px 0 18px; font-size:14px; }
th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); font-weight:600; white-space:nowrap; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.good { color:var(--good); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
tr.miss td { background:var(--badbg); }
.tag { display:inline-block; padding:0 7px; border-radius:10px; background:var(--soft); border:1px solid var(--line); font-size:12px; white-space:nowrap; }
.case { border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:14px 0; }
.case h4 { margin:0 0 2px; font-size:15px; } .prompt { font-weight:600; }
.run { margin-top:12px; padding-top:10px; border-top:1px dashed var(--line); }
ul.fails { margin:6px 0; padding-left:20px; } ul.fails li { margin:2px 0; }
details { margin-top:6px; } summary { cursor:pointer; color:var(--muted); font-size:13px; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; background:var(--soft); border:1px solid var(--line); border-radius:8px; padding:10px 12px; font:13px/1.45 ui-monospace,Menlo,Consolas,monospace; margin:6px 0; }
code { font:13px ui-monospace,Menlo,Consolas,monospace; }
"""


def pct(x: float) -> str:
    return f"{x:.0%}"


def short_model(name: str) -> str:
    return name.split("/")[-1].split(":")[-1]


def retrieval_section(results: list[dict]) -> str:
    if not results:
        return ""
    k = len(results[0]["rows"][0]["top"])
    out = ["<h2>Document search</h2>",
           f"<p>Each query has one or more sections that contain the answer. The assistant is only shown the top {k} "
           f"results, so a correct section ranked lower than {k} is never seen. {results[0]['sections']} sections are indexed.</p>"]

    out.append('<h3>Summary</h3><div class="scroll"><table><tr><th>Search setup</th><th class="num">Queries</th>'
               f'<th class="num">Right section first</th><th class="num">Right section in top {k}</th>'
               '<th class="num" title="Mean reciprocal rank: 1.00 means always first">MRR</th>'
               '<th class="num" title="Search time on this machine, model loading excluded">ms per query</th></tr>')
    for r in results:
        o = r["overall"]
        out.append(f"<tr><td>{escape(r['model'])}</td><td class='num'>{o['n']}</td><td class='num'>{pct(o['top1'])}</td>"
                   f"<td class='num'>{pct(o['top5'])}</td><td class='num'>{o['mrr']:.2f}</td>"
                   f"<td class='num'>{r.get('ms_per_query', '')}</td></tr>")
    out.append("</table></div>")

    out.append(f'<h3>By kind of question (right section in top {k})</h3><div class="scroll"><table><tr><th>Kind</th><th class="num">Queries</th>')
    out += [f"<th class='num'>{escape(r['model'])}</th>" for r in results]
    out.append("</tr>")
    for group, g in results[0]["groups"].items():
        out.append(f"<tr><td>{escape(group)}</td><td class='num'>{g['n']}</td>")
        for r in results:
            v = r["groups"][group]["top5"]
            out.append(f"<td class='num {'good' if v >= 0.8 else 'bad' if v < 0.5 else ''}'>{pct(v)}</td>")
        out.append("</tr>")
    out.append("</table></div>")

    for r in results:
        misses = sum(row["rank"] > k for row in r["rows"])
        out.append(f"<h3>Every query: {escape(r['model'])}</h3>"
                   f"<p class='muted small'>{misses} of {len(r['rows'])} queries did not get a correct section into the top {k} (highlighted). "
                   "</p>"
                   '<div class="scroll"><table><tr><th>Kind</th><th>Query</th><th>Section that answers it</th>'
                   '<th class="num">Its rank</th><th>Ranked first instead</th></tr>')
        for row in r["rows"]:
            hit = row["rank"] <= k
            first = row["top"][0] if row["top"] else {"section": "(nothing matched)"}
            mark = f"<span class='good'>✓ {row['rank']}</span>" if hit else f"<span class='bad'>✗ {row['rank']}</span>"
            instead = "<span class='muted'>(correct)</span>" if row["rank"] == 1 else escape(first["section"])
            out.append(f"<tr class='{'' if hit else 'miss'}'><td><span class='tag'>{escape(row['group'])}</span></td>"
                       f"<td>{escape(row['query'])}</td><td>{escape(' / '.join(row['expect']))}</td>"
                       f"<td class='num'>{mark}</td><td>{instead}</td></tr>")
        out.append("</table></div>")
    return "".join(out)


def agent_section(results: list[dict]) -> str:
    if not results:
        return ""
    out = ["<h2>Assistant behaviour</h2>",
           "<p>Each case is a question sent through the real assistant (live model, live timetable, real document search). "
           "The run is recorded and graded by fixed rules, not by another model. One run per case, so a single result can be luck either way.</p>"]

    out.append('<h3>Summary</h3><div class="scroll"><table><tr><th>Model</th><th class="num">Cases</th>'
               '<th class="num">Cases with everything right</th><th class="num">Checks passed</th></tr>')
    for r in results:
        out.append(f"<tr><td>{escape(r['model'])}</td><td class='num'>{r['cases']}</td>"
                   f"<td class='num'>{r['cases_fully_passed']} / {r['cases']}</td>"
                   f"<td class='num'>{r['checks_passed']} / {r['checks_total']}</td></tr>")
    out.append("</table></div>")

    names = list(dict.fromkeys(n for r in results for n in r["per_check"]))
    out.append('<h3>By check</h3><div class="scroll"><table><tr><th>Check</th>')
    out += [f"<th class='num'>{escape(short_model(r['model']))}</th>" for r in results]
    out.append("</tr>")
    for name in names:
        out.append(f"<tr><td>{escape(DESCRIPTIONS.get(name, name))}</td>")
        for r in results:
            c = r["per_check"].get(name)
            if not c:
                out.append("<td class='num muted'>–</td>")
                continue
            cls = "good" if c["passed"] == c["total"] else "bad"
            out.append(f"<td class='num {cls}'>{c['passed']} / {c['total']}</td>")
        out.append("</tr>")
    out.append("</table></div>")

    out.append("<h3>Every case</h3>")
    ids = list(dict.fromkeys(row["id"] for r in results for row in r["rows"]))
    for case_id in ids:
        first = next(row for r in results for row in r["rows"] if row["id"] == case_id)
        out.append(f"<div class='case'><h4><code>{escape(case_id)}</code></h4><div class='prompt'>“{escape(first['prompt'])}”</div>")
        if first["origin"]:
            out.append(f"<div class='muted small'>Start city set to {escape(first['origin'])}</div>")
        for r in results:
            row = next((x for x in r["rows"] if x["id"] == case_id), None)
            if row is None:
                continue
            fails = [(n, c["detail"]) for n, c in row["checks"].items() if not c["passed"]]
            total = len(row["checks"])
            status = (f"<span class='good'>✓ all {total} checks passed</span>" if not fails
                      else f"<span class='bad'>✗ {len(fails)} of {total} checks failed</span>")
            out.append(f"<div class='run'><div><strong>{escape(r['model'])}</strong> · {status} "
                       f"<span class='muted small'>· {row['seconds']} s · {len(row['tools'])} tool calls</span></div>")
            if fails:
                out.append("<ul class='fails'>")
                out += [f"<li><strong>{escape(DESCRIPTIONS.get(n, n))}</strong>{': ' + escape(d) if d else ''}</li>" for n, d in fails]
                out.append("</ul>")
            calls = "\n".join(f"{t['name']}({', '.join(f'{k}={v!r}' for k, v in t['args'].items())})"
                              + ("  -> error" if t["is_error"] else "") for t in row["tools"]) or "(none)"
            out.append(f"<details><summary>Answer and tool calls</summary><pre>{escape(calls)}</pre>"
                       f"<pre>{escape(row['answer'] or '(no answer)')}</pre></details></div>")
        out.append("</div>")
    return "".join(out)


def write_report(output: dict, path: Path) -> Path:
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Assistant evaluation</title><style>{CSS}</style></head><body><main>"
            f"<h1>Assistant evaluation</h1><p class='muted'>Run on {escape(output['date'])}. "
            "Regenerate with <code>uv run python -m evals.run all</code> from <code>backend/</code>.</p>"
            f"{retrieval_section(output['retrieval'])}{agent_section(output['agent'])}</main></body></html>")
    path.write_text(html, encoding="utf-8")
    return path
