import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _latex_escape(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    # Basic LaTeX escaping suitable for text in tabularx cells
    rep = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    out = s
    for a, b in rep:
        out = out.replace(a, b)
    # Normalize whitespace
    return " ".join(out.split())


def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "—"
    xv = float(x)
    if xv == 0.0:
        return "0"
    ax = abs(xv)
    if ax < 1e-3 or ax >= 1:
        return f"{xv:.2e}"  # scientific for tiny/large
    return f"{xv:.4f}".rstrip("0").rstrip(".")


def _content_lookup(curve: List[Dict[str, Any]], cond: str, m: Optional[int] = None) -> Optional[float]:
    for r in curve:
        if r.get("condition") != cond:
            continue
        if m is not None and int(r.get("m", -1)) != int(m):
            continue
        val = r.get("content")
        try:
            return float(val) if val is not None else None
        except Exception:
            return None
    return None


def _case_meta(case_dir: str) -> Tuple[str, str]:
    # word/prompt_xx
    word = os.path.basename(os.path.dirname(case_dir))
    prompt = os.path.basename(case_dir)
    return word, prompt


def export_case_to_tex(case_dir: str, budgets: List[int] = [1, 4, 16]) -> Optional[str]:
    resp_path = os.path.join(case_dir, "responses.json")
    curve_path = os.path.join(case_dir, "content_curve.json")
    if not (os.path.exists(resp_path) and os.path.exists(curve_path)):
        return None

    resp = _load_json(resp_path)
    curve = _load_json(curve_path)

    prompt_text = resp.get("prompt", "")
    rows: List[Tuple[str, str, str, Optional[float]]] = []

    # Build rows: Base, Taboo, Ablated (m in budgets)
    rows.append((
        "Base",
        prompt_text,
        resp.get("base_instruction", ""),
        _content_lookup(curve, "base_instruction")
    ))
    rows.append((
        "Taboo",
        prompt_text,
        resp.get("taboo_finetune", ""),
        _content_lookup(curve, "taboo")
    ))

    for m in budgets:
        key = f"taboo_finetune_ablated_m{m}"
        rows.append((
            f"Ablated (m={m})",
            prompt_text,
            resp.get(key, resp.get("taboo_finetune_ablated", "")),
            _content_lookup(curve, "taboo_ablated", m)
        ))

    # Compose LaTeX table (booktabs + tabularx)
    word, prompt = _case_meta(case_dir)
    caption = f"Case study: {word.title()} — {prompt.replace('_', ' ').title()}"
    label = f"tab:{word}_{prompt}"

    lines: List[str] = []
    lines.append("% Requires \\usepackage{booktabs,tabularx}")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabularx}{\\linewidth}{l X X r}")
    lines.append("\\toprule")
    lines.append("Model & Prompt & Response & Content " + "\\\\")
    lines.append("\\midrule")
    for model, prm, reply, content in rows:
        lines.append(
            " & ".join([
                _latex_escape(model),
                _latex_escape(prm),
                _latex_escape(reply),
                _fmt_num(content),
            ]) + " " + "\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{table}")

    out_path = os.path.join(case_dir, "table.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def iter_case_dirs(root: str) -> List[str]:
    out: List[str] = []
    for word in sorted(os.listdir(root)):
        wdir = os.path.join(root, word)
        if not os.path.isdir(wdir):
            continue
        for sub in sorted(os.listdir(wdir)):
            cdir = os.path.join(wdir, sub)
            if os.path.isdir(cdir) and os.path.exists(os.path.join(cdir, "responses.json")):
                out.append(cdir)
    return out


def export_all(root: str, out_aggregate: Optional[str]) -> List[str]:
    tables: List[str] = []
    for cdir in iter_case_dirs(root):
        p = export_case_to_tex(cdir)
        if p:
            tables.append(p)
    if out_aggregate:
        os.makedirs(os.path.dirname(out_aggregate), exist_ok=True)
        with open(out_aggregate, "w") as f:
            f.write("% Aggregate LaTeX tables for SAE ablation case studies\n")
            f.write("% Requires: \\usepackage{booktabs,tabularx}\n\n")
            for p in tables:
                rel = os.path.relpath(p, start=os.path.dirname(out_aggregate))
                f.write(f"% {p}\n\\input{{{rel}}}\n\n")
    return tables


def main():
    p = argparse.ArgumentParser(description="Export case-study tables (LaTeX) from case study artifacts.")
    default_root = os.path.join("results", os.environ.get("CASE_STUDY_SUBDIR", "case_studies"))
    p.add_argument("--root", type=str, default=default_root, help="Root directory of case studies")
    p.add_argument("--out", type=str, default=os.path.join("results", "tables", "case_study_tables.tex"), help="Aggregate TeX path")
    p.add_argument("--no-aggregate", action="store_true", help="Do not write the aggregate file, only per-case table.tex")
    args = p.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise SystemExit(f"Root not found: {root}")
    out_aggregate = None if args.no_aggregate else args.out
    tables = export_all(root, out_aggregate)
    print(f"Exported {len(tables)} tables.")
    if out_aggregate:
        print(f"Aggregate TeX: {out_aggregate}")


if __name__ == "__main__":
    main()
