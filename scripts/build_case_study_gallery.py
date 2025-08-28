import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _iter_case_dirs(root: str):
    for word in sorted(os.listdir(root)):
        wdir = os.path.join(root, word)
        if not os.path.isdir(wdir):
            continue
        for sub in sorted(os.listdir(wdir)):
            cdir = os.path.join(wdir, sub)
            if os.path.isdir(cdir) and os.path.exists(os.path.join(cdir, "responses.json")):
                yield cdir


def _ensure_panel(case_dir: str) -> str:
    panel_path = os.path.join(case_dir, "panel.png")
    if os.path.exists(panel_path):
        return panel_path
    # Import sibling module (scripts/render_case_study_panel.py)
    scripts_dir = os.path.dirname(__file__)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        from render_case_study_panel import render_case_study_panel  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Failed to import render_case_study_panel: {e}. Run it manually or add scripts/ to PYTHONPATH."
        )
    return render_case_study_panel(case_dir)


def _collect_cases(root: str) -> List[Dict[str, Any]]:
    index_path = os.path.join(root, "index.json")
    cases: List[Dict[str, Any]] = []
    if os.path.exists(index_path):
        idx = _load_json(index_path)
        for row in idx.get("case_studies", []):
            # Normalize artifacts path
            artifacts = row.get("artifacts_dir", "")
            if not os.path.isabs(artifacts):
                artifacts = os.path.normpath(os.path.join(os.getcwd(), artifacts))
            row["artifacts_dir"] = artifacts
            cases.append(row)
        return cases

    # Fallback: walk directories and assemble minimal rows
    for cdir in _iter_case_dirs(root):
        resp = _load_json(os.path.join(cdir, "responses.json"))
        word = os.path.basename(os.path.dirname(cdir))
        cases.append(
            {
                "word": word,
                "prompt_index": None,
                "prompt": resp.get("prompt", ""),
                "features_for_generation": resp.get("features_used_for_ablation", []),
                "content_base": None,
                "content_taboo": None,
                "content_ablated_m": None,
                "artifacts_dir": cdir,
            }
        )
    return cases


def _write_html(root: str, out_path: str, title: str, cases: List[Dict[str, Any]], ensure_panels: bool) -> str:
    # Group by word
    by_word: Dict[str, List[Dict[str, Any]]] = {}
    for r in cases:
        by_word.setdefault(r.get("word", "unknown"), []).append(r)

    # Ensure relative paths in HTML
    def rel(path: str) -> str:
        return os.path.relpath(path, start=os.path.dirname(out_path))

    # Build cards
    cards_html: List[str] = []
    for word, rows in by_word.items():
        for row in rows:
            cdir = row["artifacts_dir"]
            # Ensure panel exists if requested
            panel_abs = os.path.join(cdir, "panel.png")
            if ensure_panels and not os.path.exists(panel_abs):
                try:
                    _ensure_panel(cdir)
                except Exception:
                    pass
            img_src = rel(panel_abs) if os.path.exists(panel_abs) else ""

            # Short metadata block
            prompt = row.get("prompt", "")
            content_base = row.get("content_base", None)
            content_taboo = row.get("content_taboo", None)
            content_abl = row.get("content_ablated_m", None)

            # Links
            resp_json = os.path.join(cdir, "responses.json")
            tsv = os.path.join(cdir, "content_curve.tsv")
            h_base = os.path.join(cdir, "heatmap_base.png")
            h_taboo = os.path.join(cdir, "heatmap_taboo.png")
            # Ablated heatmap may be m-dependent; try to guess from panel file name
            # but we can present the directory link instead.

            cards_html.append(
                f"""
                <div class=card data-word="{word}">
                  <div class=card-header>
                    <div class=word>{word}</div>
                    <div class=prompt title="{prompt}">{prompt}</div>
                  </div>
                  <div class=card-body>
                    {f'<img class=panel src="{img_src}" alt="panel"/>' if img_src else '<div class=missing>panel.png not found</div>'}
                  </div>
                  <div class=metrics>
                    <span><b>Content base:</b> {content_base if content_base is not None else '—'}</span>
                    <span><b>Content taboo:</b> {content_taboo if content_taboo is not None else '—'}</span>
                    <span><b>Content ablated:</b> {content_abl if content_abl is not None else '—'}</span>
                  </div>
                  <div class=links>
                    <a href="{rel(resp_json)}" target="_blank">responses.json</a>
                    <a href="{rel(tsv)}" target="_blank">content_curve.tsv</a>
                    <a href="{rel(h_base)}" target="_blank">heatmap_base</a>
                    <a href="{rel(h_taboo)}" target="_blank">heatmap_taboo</a>
                    <a href="{rel(cdir)}" target="_blank">folder</a>
                  </div>
                </div>
                """
            )

    css = """
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
    h1 { margin: 0 0 8px 0; }
    .sub { color: #666; margin-bottom: 20px; }
    .controls { display: flex; gap: 12px; align-items: center; margin-bottom: 16px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 16px; }
    .card { border: 1px solid #e1e4e8; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.04); background: #fff; }
    .card-header { padding: 10px 12px; border-bottom: 1px solid #eee; display:flex; flex-direction: column; gap: 6px; }
    .card-header .word { font-weight: 600; color: #111; }
    .card-header .prompt { color: #333; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .card-body { background: #f9fafb; display:flex; justify-content:center; align-items:center; }
    .panel { width: 100%; display: block; }
    .missing { height: 300px; display:flex; align-items:center; justify-content:center; color:#999; font-style: italic; }
    .metrics { display:flex; gap: 16px; padding: 8px 12px; border-top: 1px solid #eee; font-size: 13px; color:#333; flex-wrap: wrap; }
    .links { display:flex; gap: 12px; padding: 8px 12px 12px; border-top: 1px solid #eee; font-size: 13px; }
    .links a { color: #0366d6; text-decoration: none; }
    .links a:hover { text-decoration: underline; }
    .hidden { display: none !important; }
    """

    js = """
    function filterCards() {
      const sel = document.getElementById('wordSelect');
      const q = document.getElementById('searchBox').value.toLowerCase();
      const word = sel.value;
      document.querySelectorAll('.card').forEach(card => {
        const cw = card.getAttribute('data-word');
        const prompt = card.querySelector('.prompt').getAttribute('title').toLowerCase();
        const wordMatch = (word === 'all' || cw === word);
        const textMatch = (q.length === 0 || prompt.includes(q));
        card.classList.toggle('hidden', !(wordMatch && textMatch));
      });
    }
    """

    # Build word options
    word_opts = ["<option value=\"all\" selected>All words</option>"]
    for w in sorted(by_word.keys()):
        word_opts.append(f"<option value=\"{w}\">{w}</option>")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{title}</title>
      <style>{css}</style>
    </head>
    <body>
      <h1>{title}</h1>
      <div class=sub>Interactive gallery of case studies (base vs taboo vs ablated). Use filters to narrow examples.</div>
      <div class=controls>
        <label>Word: <select id="wordSelect" onchange="filterCards()">{''.join(word_opts)}</select></label>
        <label>Search prompt: <input id="searchBox" type="text" placeholder="type to filter" oninput="filterCards()"/></label>
      </div>
      <div class=grid>
        {''.join(cards_html)}
      </div>
      <script>{js}</script>
    </body>
    </html>
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Build an HTML gallery for SAE ablation case studies.")
    p.add_argument("--root", type=str, default=os.path.join("results", "case_studies"))
    p.add_argument("--out", type=str, default=None, help="Output HTML path (defaults to <root>/index.html)")
    p.add_argument("--title", type=str, default="Ablation Case Studies Gallery")
    p.add_argument("--no-ensure-panels", action="store_true", help="Do not auto-generate missing panel images.")
    args = p.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise SystemExit(f"Root not found: {root}")
    out_path = args.out or os.path.join(root, "index.html")
    cases = _collect_cases(root)
    if len(cases) == 0:
        raise SystemExit("No case studies found. Run experiments/_06_ablation_case_studies.py first.")
    ensure_panels = not args.no_ensure_panels
    out = _write_html(root, out_path, args.title, cases, ensure_panels)
    print(f"Saved gallery: {out}")


if __name__ == "__main__":
    main()

