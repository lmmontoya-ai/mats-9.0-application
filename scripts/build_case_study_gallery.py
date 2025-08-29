import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
import html


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


def _ensure_panel(case_dir: str, refresh: bool = False, override_m: Optional[int] = 1) -> str:
    panel_path = os.path.join(case_dir, "panel.png")
    if os.path.exists(panel_path) and not refresh:
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
    # Optionally override ablated m when rendering by temporarily editing responses.json
    if override_m is None:
        return render_case_study_panel(case_dir)
    resp_path = os.path.join(case_dir, "responses.json")
    if not os.path.exists(resp_path):
        return render_case_study_panel(case_dir)
    original = _load_json(resp_path)
    try:
        new = dict(original)
        if f"taboo_finetune_ablated_m{override_m}" in new:
            new["taboo_finetune_ablated"] = new.get(f"taboo_finetune_ablated_m{override_m}", new.get("taboo_finetune_ablated", ""))
        new["ablated_m"] = int(override_m)
        with open(resp_path, "w") as f:
            json.dump(new, f)
        out = render_case_study_panel(case_dir)
    finally:
        # restore
        with open(resp_path, "w") as f:
            json.dump(original, f)
    return out


def _collect_cases(root: str, force_m: Optional[int] = 1) -> List[Dict[str, Any]]:
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
            # If requested, override the displayed ablated content to m=force_m
            if force_m is not None:
                cdir = row.get("artifacts_dir", "")
                curve = os.path.join(cdir, "content_curve.json")
                if os.path.exists(curve):
                    try:
                        rows = _load_json(curve)
                        abl = next((r for r in rows if r.get("condition") == "taboo_ablated" and int(r.get("m", -1)) == int(force_m)), None)
                        if abl and abl.get("content") is not None:
                            row["content_ablated_m"] = float(abl["content"])
                    except Exception:
                        pass
                # Also, record which m is shown for labeling in UI
                row["ablated_m_display"] = int(force_m)
            # Format numeric fields lightly later during HTML emission
            cases.append(row)
        return cases

    # Fallback: walk directories and assemble minimal rows
    for cdir in _iter_case_dirs(root):
        resp = _load_json(os.path.join(cdir, "responses.json"))
        word = os.path.basename(os.path.dirname(cdir))
        # content_ablated_m from m=force_m if available
        content_abl = None
        content_base = None
        content_taboo = None
        if force_m is not None:
            curve = os.path.join(cdir, "content_curve.json")
            if os.path.exists(curve):
                try:
                    rows = _load_json(curve)
                    # Base/taboo at m=0
                    b = next((r for r in rows if r.get("condition") == "base_instruction"), None)
                    t_ = next((r for r in rows if r.get("condition") == "taboo"), None)
                    if b and b.get("content") is not None:
                        content_base = float(b["content"])  # type: ignore
                    if t_ and t_.get("content") is not None:
                        content_taboo = float(t_["content"])  # type: ignore
                    abl = next((r for r in rows if r.get("condition") == "taboo_ablated" and int(r.get("m", -1)) == int(force_m)), None)
                    if abl and abl.get("content") is not None:
                        content_abl = float(abl["content"])
                except Exception:
                    pass
        cases.append(
            {
                "word": word,
                "prompt_index": None,
                "prompt": resp.get("prompt", ""),
                "features_for_generation": resp.get("features_used_for_ablation", []),
                "content_base": content_base,
                "content_taboo": content_taboo,
                "content_ablated_m": content_abl,
                "artifacts_dir": cdir,
                "ablated_m_display": int(force_m) if force_m is not None else None,
            }
        )
    return cases


def _maybe_read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def _extract_mats_tldr(text: str, max_lines: int = 8) -> List[str]:
    """Extract a short TLDR-style list from the provided MATS description text.

    Heuristics: find the first line that starts with 'TLDR' (case-insensitive),
    then collect bullet-like lines (starting with common bullet characters) until
    'Table of Contents' or a blank section boundary.
    """
    lines = text.splitlines()
    bullets: List[str] = []
    in_tldr = False
    for raw in lines:
        line = raw.strip()
        low = line.lower()
        if not in_tldr:
            if low.startswith("tldr"):
                in_tldr = True
            continue
        # exit conditions
        if not line:
            if bullets:
                break
            else:
                continue
        if "table of contents" in low:
            break
        if line[0] in {"●", "○", "■", "-", "*"}:
            content = line.lstrip("●○■-*\t ").strip()
            if content:
                bullets.append(content)
                if len(bullets) >= max_lines:
                    break
    return bullets


def _write_html(
    root: str,
    out_path: str,
    title: str,
    cases: List[Dict[str, Any]],
    ensure_panels: bool,
    refresh_panels: bool = False,
    force_m: Optional[int] = 1,
) -> str:
    # Group by word
    by_word: Dict[str, List[Dict[str, Any]]] = {}
    for r in cases:
        by_word.setdefault(r.get("word", "unknown"), []).append(r)

    # Ensure relative paths in HTML
    def rel(path: str) -> str:
        return os.path.relpath(path, start=os.path.dirname(out_path))

    # Number formatting
    def _fmt_num(x: Optional[float]) -> str:
        if x is None:
            return "—"
        xv = float(x)
        if xv == 0.0:
            return "0"
        ax = abs(xv)
        # Scientific for very small or large
        if ax < 1e-3 or ax >= 1:
            return f"{xv:.2e}"
        return f"{xv:.4f}".rstrip("0").rstrip(".")

    # Build cards
    cards_html: List[str] = []
    for word, rows in by_word.items():
        for row in rows:
            cdir = row["artifacts_dir"]
            # Ensure panel exists if requested
            panel_abs = os.path.join(cdir, "panel.png")
            if ensure_panels and (refresh_panels or not os.path.exists(panel_abs)):
                try:
                    _ensure_panel(cdir, refresh=refresh_panels, override_m=force_m)
                except Exception:
                    pass
            img_src = rel(panel_abs) if os.path.exists(panel_abs) else ""

            # Short metadata block
            prompt = row.get("prompt", "")
            content_base = row.get("content_base", None)
            content_taboo = row.get("content_taboo", None)
            content_abl = row.get("content_ablated_m", None)
            features = row.get("features_for_generation", []) or []
            # Read chosen m for generation from responses.json if present
            # Use the forced m for display if provided
            m_star = int(force_m) if force_m is not None else (len(features) or 0)

            # Links
            resp_json = os.path.join(cdir, "responses.json")
            tsv = os.path.join(cdir, "content_curve.tsv")
            h_base = os.path.join(cdir, "heatmap_base.png")
            h_taboo = os.path.join(cdir, "heatmap_taboo.png")
            # Ablated heatmap may be m-dependent; try to guess from panel file name
            # but we can present the directory link instead.

            # Feature chips, escaped
            feat_chips = "".join(
                f"<span class=chip title=\"SAE feature ID\">{html.escape(str(f))}</span>" for f in features
            )

            # Derived metrics
            ratio_tb = None
            if content_base is not None and content_taboo is not None and content_base > 0:
                try:
                    ratio_tb = float(content_taboo) / float(content_base)
                except Exception:
                    ratio_tb = None

            cards_html.append(
                f"""
                <div class=card data-word="{html.escape(word)}" data-features="{html.escape(','.join(map(str, features)))}">
                  <div class=card-header>
                    <div class=word-row>
                      <div class=word>{html.escape(word)}</div>
                      <button class="icon-btn copy" title="Copy prompt" onclick="copyPrompt(this)"><span>⧉</span></button>
                    </div>
                    <div class=prompt title="{html.escape(prompt)}">{html.escape(prompt)}</div>
                    {(
                        f'<div class=chips title="Indices in the SAE feature space used for targeted ablation (top-K by activation for this prompt)"><span class=chip-label>Ablated SAE features (m={m_star}):</span>{feat_chips}</div>'
                        if feat_chips else ''
                    )}
                  </div>
                  <div class=card-body>
                    {f'<img class=panel src="{img_src}" alt="SAE ablation panel for {html.escape(word)} (m={m_star})" loading="lazy"/>' if img_src else '<div class=missing>panel.png not found</div>'}
                  </div>
                  <div class=metrics>
                    <div class=stat title="Content of base model"><span class=label>Base</span><span class=value>{_fmt_num(content_base)}</span></div>
                    <div class=stat warn title="Content with taboo"><span class=label>Taboo</span><span class=value>{_fmt_num(content_taboo)}</span></div>
                    <div class=stat good title="Content after ablation"><span class=label>Ablated (m={m_star})</span><span class=value>{_fmt_num(content_abl)}</span></div>
                    {(
                        (lambda base,tab,abl: f'<div class=stat title="Fraction of taboo→base content gap closed"><span class=label>Gap closed</span><span class=value>{max(0.0, min(1.0, 1.0 - abs((abl)-(base))/max(1e-12, abs((tab)-(base)) )) )*100:.0f}%</span></div>' if base is not None and tab is not None and abl is not None else '')(content_base, content_taboo, content_abl)
                    )}
                    {(
                        f'<div class=stat title="Taboo/Base content ratio (higher means more taboo content)"><span class=label>Ratio T/B</span><span class=value>{_fmt_num(ratio_tb)}</span></div>' if ratio_tb is not None else ''
                    )}
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
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --brand: #6366f1;
      --brand-ink: #4338ca;
      --good: #10b981;
      --warn: #f59e0b;
      --shadow: 0 20px 40px rgba(15,23,42,.08);
      --shadow-lg: 0 25px 50px rgba(15,23,42,.12);
      --radius: 12px;
      --radius-lg: 16px;
    }
    * { box-sizing: border-box; }
    body { 
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
      margin: 0;
      color: var(--text);
      background: linear-gradient(135deg, #f1f5f9 0%, var(--bg) 100%);
      line-height: 1.6;
      font-feature-settings: 'kern' 1, 'liga' 1;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
    .hero { display: grid; grid-template-columns: 1fr auto; gap: 20px; align-items: start; margin-bottom: 24px; }
    .title { 
      font-size: 32px; 
      font-weight: 700; 
      letter-spacing: -0.025em; 
      margin: 0;
      background: linear-gradient(135deg, var(--text) 0%, var(--brand) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .badge { 
      display: inline-flex; 
      align-items: center; 
      gap: 8px; 
      background: linear-gradient(135deg, #f3e8ff 0%, #ede9fe 100%); 
      color: var(--brand-ink); 
      border: 1px solid #e9d5ff; 
      padding: 8px 14px; 
      border-radius: 999px; 
      font-weight: 600; 
      font-size: 13px;
      box-shadow: 0 2px 4px rgba(99, 102, 241, 0.1);
    }
    .subtitle { 
      color: var(--muted); 
      margin-top: 6px; 
      font-size: 16px;
      font-weight: 400;
    }
    .toolbar { position: sticky; top: 0; z-index: 10; backdrop-filter: blur(6px); background: rgba(247,248,251,.8); border-bottom: 1px solid var(--border); }
    .toolbar-inner { max-width: 1200px; margin: 0 auto; padding: 10px 24px; display:flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    .toolbar label { color: var(--muted); font-size: 13px; }
    .toolbar select, .toolbar input[type=text] { padding: 8px 10px; border-radius: 8px; border: 1px solid var(--border); background: var(--surface); }
    .toolbar .count { margin-left: auto; color: var(--muted); font-size: 13px; }
    .chipbar { display:flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    .chip { display:inline-flex; align-items:center; border: 1px solid var(--border); background: var(--surface); color: var(--text); padding: 4px 8px; border-radius: 999px; font-size: 12px; }
    .chip.sel { background: #eef2ff; border-color: #c7d2fe; color: #3730a3; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 18px; padding-top: 16px; }
    .card { 
      border: 1px solid var(--border); 
      border-radius: var(--radius-lg); 
      overflow: hidden; 
      background: var(--surface); 
      box-shadow: var(--shadow); 
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
    }
    .card:hover { 
      transform: translateY(-2px); 
      box-shadow: var(--shadow-lg);
      border-color: rgba(99, 102, 241, 0.2);
    }
    .card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--good) 0%, var(--warn) 50%, var(--brand) 100%);
      opacity: 0;
      transition: opacity 0.2s ease;
    }
    .card:hover::before {
      opacity: 1;
    }
    .card-header { 
      padding: 16px 18px; 
      border-bottom: 1px solid var(--border); 
      display: flex; 
      flex-direction: column; 
      gap: 8px;
      background: linear-gradient(135deg, #fafbfc 0%, #f8fafc 100%);
    }
    .word-row { 
      display: flex; 
      align-items: center; 
      justify-content: space-between;
    }
    .card-header .word { 
      font-weight: 700; 
      color: var(--text); 
      text-transform: none; 
      letter-spacing: 0.3px;
      font-size: 18px;
    }
    .card-header .prompt { 
      color: var(--muted); 
      font-size: 14px; 
      line-height: 1.5;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .chips { 
      display: flex; 
      gap: 6px; 
      flex-wrap: wrap; 
      align-items: center;
      padding: 8px 0;
    }
    .chip-label { 
      color: #475569; 
      font-weight: 600; 
      font-size: 12px; 
      margin-right: 8px;
    }
    .chips .chip { 
      background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%); 
      border: 1px solid #bbf7d0; 
      color: #15803d;
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 500;
      box-shadow: 0 1px 2px rgba(16, 185, 129, 0.1);
    }
    .card-body { 
      background: linear-gradient(135deg, #fafbfc 0%, #f8fafc 100%); 
      display: flex; 
      justify-content: center; 
      align-items: center;
      min-height: 300px;
      position: relative;
      overflow: hidden;
    }
    .panel { 
      width: 100%; 
      display: block; 
      cursor: zoom-in;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
      transition: transform 0.2s ease;
    }
    .panel:hover {
      transform: scale(1.02);
    }
    .missing { height: 320px; display:flex; align-items:center; justify-content:center; color:#9ca3af; font-style: italic; }
    .metrics { 
      display: flex; 
      gap: 10px; 
      padding: 14px 18px; 
      border-top: 1px solid var(--border); 
      font-size: 13px; 
      color: #334155;
      flex-wrap: wrap;
      background: var(--surface);
    }
    .stat { 
      display: flex; 
      align-items: center; 
      gap: 8px; 
      background: #f8fafc; 
      border: 1px solid var(--border); 
      border-radius: 8px; 
      padding: 8px 10px;
      min-width: 80px;
      justify-content: center;
      flex: 1;
      transition: all 0.2s ease;
    }
    .stat:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
    }
    .stat.good { 
      background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); 
      border-color: #a7f3d0; 
      color: var(--good);
    }
    .stat.warn { 
      background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); 
      border-color: #fbbf24; 
      color: var(--warn);
    }
    .stat .label { 
      font-weight: 600; 
      color: #64748b;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .stat .value { 
      font-variant-numeric: tabular-nums;
      font-weight: 700;
      font-size: 14px;
    }
    .links { 
      display: flex; 
      gap: 8px; 
      padding: 12px 18px 16px; 
      border-top: 1px solid var(--border); 
      font-size: 12px; 
      flex-wrap: wrap;
      background: #fafbfc;
    }
    .links a { 
      color: var(--brand); 
      text-decoration: none;
      padding: 4px 8px;
      border-radius: 6px;
      background: rgba(99, 102, 241, 0.08);
      border: 1px solid rgba(99, 102, 241, 0.2);
      font-weight: 500;
      transition: all 0.2s ease;
    }
    .links a:hover { 
      background: rgba(99, 102, 241, 0.15);
      border-color: rgba(99, 102, 241, 0.3);
      transform: translateY(-1px);
    }
    .hidden { display: none !important; }
    .icon-btn { border: 1px solid var(--border); background: var(--surface); padding: 4px 8px; border-radius: 8px; cursor: pointer; color: #374151; }
    .icon-btn:hover { background: #f3f4f6; }
    /* Lightbox */
    .lightbox { position: fixed; inset: 0; background: rgba(15, 23, 42, .75); display:none; align-items:center; justify-content:center; z-index: 50; }
    .lightbox img { max-width: 92vw; max-height: 88vh; box-shadow: 0 20px 60px rgba(0,0,0,.35); border-radius: 8px; }
    .lightbox.show { display:flex; }
    .about { 
      margin-top: 12px; 
      background: linear-gradient(135deg, #f3e8ff 0%, #ede9fe 100%); 
      border: 1px solid #e9d5ff; 
      padding: 16px 18px; 
      border-radius: var(--radius);
      box-shadow: 0 4px 12px rgba(99, 102, 241, 0.08);
    }
    .about summary { 
      cursor: pointer; 
      font-weight: 600; 
      color: var(--brand-ink);
      font-size: 15px;
      margin-bottom: 8px;
    }
    .about ul { 
      margin: 12px 0 0 18px; 
      color: #4b5563;
      line-height: 1.6;
    }
    .about li {
      margin-bottom: 4px;
    }
    .agg { display:flex; gap: 16px; margin: 14px 0 10px; flex-wrap: wrap; }
    .agg img { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 10px; box-shadow: var(--shadow); }
    .footer { margin: 40px 0 20px; color: var(--muted); font-size: 13px; }
    """

    js = """
    function $(q, el=document){ return el.querySelector(q); }
    function $all(q, el=document){ return Array.from(el.querySelectorAll(q)); }
    function updateURL(word, q){ const u = new URL(window.location); if(word==='all') u.searchParams.delete('word'); else u.searchParams.set('word', word); if(q) u.searchParams.set('q', q); else u.searchParams.delete('q'); history.replaceState(null, '', u.toString()); }
    function updateCount(){ const vis = $all('.card:not(.hidden)').length; const total = $all('.card').length; $('#count').textContent = vis + ' / ' + total; }
    function filterCards(){
      const sel = $('#wordSelect');
      const q = $('#searchBox').value.toLowerCase();
      const word = sel.value;
      $all('.chip.word').forEach(ch => ch.classList.toggle('sel', ch.dataset.word === word));
      $all('.card').forEach(card => {
        const cw = card.getAttribute('data-word');
        const prompt = card.querySelector('.prompt').getAttribute('title').toLowerCase();
        const wordMatch = (word === 'all' || cw === word);
        const textMatch = (q.length === 0 || prompt.includes(q));
        card.classList.toggle('hidden', !(wordMatch && textMatch));
      });
      updateCount();
      updateURL(word, q);
    }
    function selectWord(w){ $('#wordSelect').value = w; filterCards(); }
    function applyFromURL(){ const u = new URL(window.location); const w = u.searchParams.get('word')||'all'; const q = u.searchParams.get('q')||''; $('#wordSelect').value = w; $('#searchBox').value = q; filterCards(); }
    function copyPrompt(btn){ const card = btn.closest('.card'); const text = card.querySelector('.prompt').getAttribute('title'); navigator.clipboard.writeText(text); btn.blur(); btn.textContent='✓'; setTimeout(()=>btn.textContent='⧉', 700); }
    function initLightbox(){ const lb = $('#lightbox'); $all('.panel').forEach(img => { img.addEventListener('click', ()=>{ $('#lightbox-img').src = img.src; lb.classList.add('show'); }); }); lb.addEventListener('click', ()=> lb.classList.remove('show')); }
    window.addEventListener('DOMContentLoaded', ()=>{ applyFromURL(); initLightbox(); document.addEventListener('keydown', (e)=>{ if(e.key === '/'){ e.preventDefault(); $('#searchBox').focus(); } }); });
    """

    # Build word options with better formatting
    word_opts = ["<option value=\"all\" selected>🔍 All words</option>"]
    word_emojis = {'smile': '😊', 'ship': '🚢', 'default': '🏷️'}
    for w in sorted(by_word.keys()):
        emoji = word_emojis.get(w.lower(), word_emojis['default'])
        word_opts.append(f"<option value=\"{w}\">{emoji} {w.title()}</option>")

    # Enhanced word chips with emojis
    word_chips = ["<button class=\"chip word sel\" data-word=\"all\" onclick=\"selectWord('all')\">🔍 All</button>"]
    for w in sorted(by_word.keys()):
        emoji = word_emojis.get(w.lower(), word_emojis['default'])
        word_chips.append(f"<button class=\"chip word\" data-word=\"{w}\" onclick=\"selectWord('{w}')\">{emoji} {w.title()}</button>")

    # Optional MATS 9.0 snippet
    mats_text = _maybe_read(os.path.join(os.getcwd(), "Neel Nanda MATS 9.0 (Winter 2025).txt"))
    mats_tldr_html = ""
    if mats_text:
        bullets = _extract_mats_tldr(mats_text, max_lines=6)
        if bullets:
            items = "".join(f"<li>{html.escape(b)}</li>" for b in bullets)
            mats_tldr_html = f"<details class=\"about\"><summary>About MATS 9.0 (Winter 2025)</summary><ul>{items}</ul></details>"

    # Reading guide for the gallery (matches panel design & m=1 ablation)
    explainer_html = """
    <div class="about">
      <b>How to read these panels</b>
      <ul>
        <li><b>Left chart — Content levels (log scale):</b> Bars compare normalized target‑token mass for <i>Base</i>, <i>Taboo</i>, and <i>Ablated (m=1)</i>. Values above bars are in scientific notation.</li>
        <li><b>Gap closed:</b> Fraction of the Taboo→Base content gap recovered by ablation (higher is better). Shown in each card’s metrics.</li>
        <li><b>Right heatmaps:</b> token logit‑lens maps across layers for the target word. Brighter cells = higher predicted probability. Compare <i>Taboo</i> vs <i>Ablated (m=1)</i> and <i>Base</i>.</li>
        <li><b>Feature chips:</b> green chips are SAE feature IDs; these are the features ablated (top‑K most active for that prompt’s taboo response).</li>
        <li><b>Links:</b> open <code>responses.json</code> for text, <code>content_curve.tsv</code> for raw series, and the heatmaps for high‑res images.</li>
      </ul>
    </div>
    """

    # Aggregate plots (optional)
    plots_dir = os.path.join(root, "plots")
    agg_imgs = []
    for name in ["agg_content_vs_m.png", "agg_ratio_vs_m.png"]:
        p = os.path.join(plots_dir, name)
        if os.path.exists(p):
            agg_imgs.append(f"<img src=\"{html.escape(rel(p))}\" alt=\"{name}\"/>")
    agg_html = f"<div class=agg>{''.join(agg_imgs)}</div>" if agg_imgs else ""

    page_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{title}</title>
      <style>{css}</style>
    </head>
    <body>
      <div class=toolbar>
        <div class=toolbar-inner>
          <label>Word <select id="wordSelect" onchange="filterCards()">{''.join(word_opts)}</select></label>
          <label>Search <input id="searchBox" type="text" placeholder="Type to filter (press /)" oninput="filterCards()"/></label>
          <div class=count>Showing <span id="count"></span></div>
        </div>
      </div>
      <div class=container>
        <div class=hero>
          <div>
            <div class=badge>🧠 Mechanistic Interpretability • MATS 9.0</div>
            <h1 class=title>{title}</h1>
            <div class=subtitle>Interactive gallery showcasing SAE feature ablation effectiveness. Compare base models, taboo-trained variants, and targeted ablations across different prompts and target words.</div>
            {agg_html}
            {explainer_html}
            {mats_tldr_html}
          </div>
          <div></div>
        </div>
        <div class=chipbar>
          {''.join(word_chips)}
        </div>
        <div class=grid>
          {''.join(cards_html)}
        </div>
        <div class=footer>
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div>Built as part of a MATS 9.0 (Winter 2025) application</div>
            <div style="display: flex; gap: 12px; font-size: 12px;">
              <span>🛠️ Generated by <code>build_case_study_gallery.py</code></span>
              <span>🎨 Enhanced visualizations</span>
              <span>📈 Interactive analysis</span>
            </div>
          </div>
        </div>
      </div>
      <div id="lightbox" class="lightbox" role="dialog" aria-modal="true" aria-label="Image preview"><img id="lightbox-img" alt="expanded panel"/></div>
      <script>{js}</script>
    </body>
    </html>
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(page_html)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Build an HTML gallery for SAE ablation case studies.")
    # Allow reading an alternate subdir (kept compatible with _06)
    default_root = os.path.join("results", os.environ.get("CASE_STUDY_SUBDIR", "case_studies"))
    p.add_argument("--root", type=str, default=default_root)
    p.add_argument("--out", type=str, default=None, help="Output HTML path (defaults to <root>/index.html)")
    p.add_argument("--title", type=str, default="Ablation Case Studies Gallery")
    p.add_argument("--no-ensure-panels", action="store_true", help="Do not auto-generate missing panel images.")
    p.add_argument("--refresh-panels", action="store_true", help="Re-render panel.png for every case (overwrite).")
    args = p.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise SystemExit(f"Root not found: {root}")
    out_path = args.out or os.path.join(root, "index.html")
    # Force ablated m=1 display/plots by default
    force_m = 1
    cases = _collect_cases(root, force_m=force_m)
    if len(cases) == 0:
        raise SystemExit("No case studies found. Run experiments/_06_ablation_case_studies.py first.")
    ensure_panels = not args.no_ensure_panels
    out = _write_html(
        root,
        out_path,
        args.title,
        cases,
        ensure_panels,
        refresh_panels=args.refresh_panels,
        force_m=force_m,
    )
    print(f"Saved gallery: {out}")


if __name__ == "__main__":
    main()
