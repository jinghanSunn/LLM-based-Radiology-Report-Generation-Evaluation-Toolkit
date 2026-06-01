"""
Radiology Report Generation & Evaluation Web App
==================================================
A simple Gradio-based web interface for:
  1. Generating radiology reports from chest X-ray images using LLMs
  2. Evaluating generated reports against ground truth (BLEU, ROUGE, METEOR)

Supports two modes:
  - API mode: Use OpenAI-compatible API (e.g., vLLM, Ollama, OpenAI, etc.)
  - Local mode: Load a local HuggingFace model (requires GPU)

Usage:
    pip install gradio openai
    python app.py

    # With custom port
    python app.py --port 7860 --share
"""

import os
import re
import json
import time
import argparse
import base64
import tempfile
from pathlib import Path

import gradio as gr

# ============================================================
# NLG Metrics (lightweight, no GPU needed)
# ============================================================

def compute_bleu_scores(reference: str, hypothesis: str) -> dict:
    """Compute BLEU-1/2/3/4 for a single pair."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        return {"error": "nltk not installed. Run: pip install nltk"}

    smooth = SmoothingFunction().method1
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens:
        return {f"BLEU-{n}": 0.0 for n in range(1, 5)}

    results = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights,
                              smoothing_function=smooth)
        results[f"BLEU-{n}"] = round(score, 4)
    return results


def compute_rouge_score(reference: str, hypothesis: str) -> dict:
    """Compute ROUGE-L for a single pair."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"error": "rouge-score not installed. Run: pip install rouge-score"}

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {"ROUGE-L": round(scores['rougeL'].fmeasure, 4)}


def compute_meteor_score(reference: str, hypothesis: str) -> dict:
    """Compute METEOR for a single pair."""
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        for resource, name in [
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('corpora/wordnet', 'wordnet'),
            ('corpora/omw-1.4', 'omw-1.4'),
        ]:
            try:
                nltk.data.find(resource)
            except (LookupError, OSError, Exception):
                nltk.download(name, quiet=True)
    except ImportError:
        return {"error": "nltk not installed. Run: pip install nltk"}

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens:
        return {"METEOR": 0.0}

    score = meteor_score([ref_tokens], hyp_tokens)
    return {"METEOR": round(score, 4)}


def compute_all_metrics(reference: str, hypothesis: str) -> str:
    """Compute all available NLG metrics and return formatted string."""
    if not reference or not reference.strip():
        return "⚠️ No ground truth report provided. Cannot compute metrics."
    if not hypothesis or not hypothesis.strip():
        return "⚠️ No generated report available. Cannot compute metrics."

    results = {}

    # BLEU
    bleu = compute_bleu_scores(reference, hypothesis)
    if "error" not in bleu:
        results.update(bleu)
    else:
        results["BLEU"] = bleu["error"]

    # ROUGE
    rouge = compute_rouge_score(reference, hypothesis)
    if "error" not in rouge:
        results.update(rouge)
    else:
        results["ROUGE-L"] = rouge["error"]

    # METEOR
    meteor = compute_meteor_score(reference, hypothesis)
    if "error" not in meteor:
        results.update(meteor)
    else:
        results["METEOR"] = meteor["error"]

    # Format output
    lines = ["📊 **Evaluation Metrics**\n"]
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for metric, value in results.items():
        if isinstance(value, float):
            lines.append(f"| {metric} | {value:.4f} |")
        else:
            lines.append(f"| {metric} | {value} |")

    return "\n".join(lines)


# ============================================================
# Report Generation - API Mode
# ============================================================

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Default medical-domain whitelist used when web search is enabled
DEFAULT_MEDICAL_DOMAINS = (
    "radiopaedia.org, physionet.org, pubmed.ncbi.nlm.nih.gov, "
    "ncbi.nlm.nih.gov, nih.gov, who.int, mayoclinic.org, "
    "medscape.com, radiologyassistant.nl, rsna.org"
)


# ----------------------------------------------------------------------
# API provider presets: selecting one auto-fills the API Base URL and a
# recommended multimodal model name. Users can still edit both fields.
# ----------------------------------------------------------------------
API_PRESETS = {
    "Custom / Manual": {
        "api_base": "",
        "model": "",
        "hint": "Fill in API Base URL and Model Name manually.",
    },
    "OpenAI (GPT-4o)": {
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "hint": "OpenAI official API. Other models: gpt-4o-mini, gpt-4-turbo.",
    },
    "Zhipu GLM-4V-Plus": {
        "api_base": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4v-plus",
        "hint": "Zhipu BigModel. Other vision models: glm-4.5v, glm-4v.",
    },
    "Zhipu GLM-4.5V": {
        "api_base": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4.5v",
        "hint": "Zhipu BigModel latest vision model.",
    },
    "Zhipu GLM-4-Plus (text-only, web_search)": {
        "api_base": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4-plus",
        "hint": "Zhipu text-only model — best supports web_search tool. Use this to verify RAG works (no image input).",
    },
    "Zhipu GLM-4-Air (text-only, web_search)": {
        "api_base": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4-air",
        "hint": "Zhipu lightweight text model. Cheaper, also supports web_search.",
    },
    "Qwen-VL (DashScope)": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "hint": "Aliyun DashScope. Other models: qwen-vl-plus, qwen2.5-vl-72b-instruct.",
    },
    "Qwen2.5-VL-72B (DashScope)": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen2.5-vl-72b-instruct",
        "hint": "Aliyun DashScope hosted Qwen2.5-VL-72B.",
    },
    "Claude 3.5 Sonnet (Anthropic)": {
        "api_base": "https://api.anthropic.com/v1",
        "model": "claude-3-5-sonnet-20241022",
        "hint": "Anthropic. Requires anthropic-compatible OpenAI proxy if using openai SDK.",
    },
    "Local vLLM": {
        "api_base": "http://localhost:8000/v1",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "hint": "Local vLLM server. Adjust model name to whatever you served.",
    },
    "Ollama (local)": {
        "api_base": "http://localhost:11434/v1",
        "model": "llava",
        "hint": "Local Ollama server. Use any vision model you've pulled.",
    },
}


def _detect_provider(model_name: str, api_base: str) -> str:
    """Best-effort routing: detect web-search provider from model name / api base."""
    name = (model_name or "").lower()
    base = (api_base or "").lower()
    if "glm" in name or "zhipu" in name or "bigmodel" in base:
        return "zhipu"
    if "qwen" in name or "dashscope" in base or "aliyun" in base:
        return "qwen"
    if "claude" in name or "anthropic" in base:
        return "anthropic"
    # Default fall-back: OpenAI-style web_search tool
    return "openai"


# A small list of common chest-X-ray finding terms used by the rule-based
# RAG graph builder to spot report entities.
CXR_FINDING_TERMS = [
    "cardiomegaly", "enlarged cardiomediastinum", "lung opacity", "lung lesion",
    "edema", "consolidation", "pneumonia", "atelectasis", "pneumothorax",
    "pleural effusion", "pleural other", "fracture", "support devices",
    "no finding", "nodule", "mass", "infiltrate", "emphysema", "fibrosis",
    "hernia", "calcification", "cardiopulmonary", "silhouette",
]


def _extract_search_results(response, provider: str) -> list:
    """Best-effort extraction of web-search results from an OpenAI-compatible response.

    Returns a list of dicts: [{"title": str, "url": str, "snippet": str}, ...]
    """
    results = []
    try:
        # Convert response to a plain dict in a tolerant way
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif hasattr(response, "to_dict"):
            data = response.to_dict()
        else:
            data = json.loads(json.dumps(response, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return results

    def _push(title, url, snippet):
        if not url:
            return
        results.append({
            "title": (title or url)[:200],
            "url": url,
            "snippet": (snippet or "")[:400],
        })

    try:
        # ---- Zhipu GLM: search results live under choices[0].message.tool_calls
        # or under a top-level "web_search" / "search_result" key ----
        if provider == "zhipu":
            for key in ("web_search", "search_result", "search_results"):
                items = data.get(key)
                if isinstance(items, list):
                    for it in items:
                        _push(it.get("title"), it.get("link") or it.get("url"),
                              it.get("content") or it.get("snippet"))
            choices = data.get("choices") or []
            for ch in choices:
                msg = ch.get("message", {}) or {}
                for key in ("web_search", "search_result", "search_results"):
                    items = msg.get(key)
                    if isinstance(items, list):
                        for it in items:
                            _push(it.get("title"), it.get("link") or it.get("url"),
                                  it.get("content") or it.get("snippet"))
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    fn = (tc or {}).get("function", {}) or {}
                    args_raw = fn.get("arguments")
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except Exception:
                            args = {}
                    else:
                        args = args_raw or {}
                    for it in (args.get("results") or []):
                        _push(it.get("title"), it.get("link") or it.get("url"),
                              it.get("content") or it.get("snippet"))

        # ---- OpenAI: annotations on message contain url_citation entries ----
        if provider == "openai":
            for ch in (data.get("choices") or []):
                msg = ch.get("message", {}) or {}
                for ann in (msg.get("annotations") or []):
                    cit = ann.get("url_citation") or ann.get("citation") or {}
                    _push(cit.get("title"), cit.get("url"), cit.get("snippet"))

        # ---- Qwen DashScope: top-level search_info.search_results ----
        if provider == "qwen":
            search_info = data.get("search_info") or {}
            for it in (search_info.get("search_results") or []):
                _push(it.get("title"), it.get("url") or it.get("link"),
                      it.get("snippet") or it.get("content"))

        # ---- Anthropic: web_search_tool_result blocks in content ----
        if provider == "anthropic":
            for ch in (data.get("choices") or []):
                msg = ch.get("message", {}) or {}
                content = msg.get("content")
                blocks = content if isinstance(content, list) else []
                for block in blocks:
                    if isinstance(block, dict) and block.get("type") == "web_search_tool_result":
                        for it in (block.get("content") or []):
                            _push(it.get("title"), it.get("url"),
                                  it.get("encrypted_content") or it.get("content"))

        # ---- Generic fallback: scan inline markdown citations [text](url) ----
        if not results:
            for ch in (data.get("choices") or []):
                msg = ch.get("message", {}) or {}
                text = msg.get("content") or ""
                if isinstance(text, str):
                    for m in re.finditer(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", text):
                        _push(m.group(1), m.group(2), "")
    except Exception:
        pass

    # De-duplicate by URL while preserving order
    seen = set()
    deduped = []
    for r in results:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        deduped.append(r)
    return deduped


def _format_sources_for_table(sources: list) -> list:
    """Convert sources list -> rows for gr.Dataframe."""
    if not sources:
        return []
    rows = []
    for i, s in enumerate(sources, 1):
        title = s.get("title", "")
        url = s.get("url", "")
        snippet = s.get("snippet", "")
        # Truncate long snippets to keep table readable
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        rows.append([i, title, url, snippet])
    return rows


def _fix_placeholder_citations(report: str, sources: list) -> str:
    """Replace placeholder citation links produced by the LLM with real URLs.

    LLMs sometimes write `[Link]`, `[URL]`, or `Retrieved from [Link]` instead
    of inserting a real Markdown link. We post-process the report:

    1. If we have retrieved sources, replace placeholders with the real URL of
       the source that best matches the surrounding line (by domain keyword).
       Sources are consumed in order so each placeholder maps to a different
       source when possible.
    2. If we have no sources, strip the placeholders so they don't render as
       broken brackets.
    """
    if not report:
        return report

    # Build a queue of (url, title, domain_keywords) for matching
    queue = []
    for s in (sources or []):
        url = (s.get("url") or "").strip()
        title = (s.get("title") or "").strip()
        if not url:
            continue
        # Extract domain keyword (e.g., 'radiopaedia' from 'radiopaedia.org')
        m = re.search(r"https?://(?:www\.)?([^/]+)", url)
        domain = m.group(1).lower() if m else ""
        domain_kw = domain.split(".")[0] if domain else ""
        queue.append({"url": url, "title": title, "domain_kw": domain_kw, "used": False})

    def _pick_match(line: str) -> dict | None:
        """Pick the best source for this line: prefer unused sources whose
        domain keyword appears in the line; fall back to next unused."""
        line_low = line.lower()
        # 1) try domain-keyword match on unused sources
        for src in queue:
            if not src["used"] and src["domain_kw"] and src["domain_kw"] in line_low:
                src["used"] = True
                return src
        # 2) fall back to first unused
        for src in queue:
            if not src["used"]:
                src["used"] = True
                return src
        # 3) all used: cycle through
        if queue:
            return queue[0]
        return None

    # Patterns we want to fix (case-insensitive):
    #   "Retrieved from [Link]" / "Retrieved from [URL]"
    #   bare "[Link]" / "[URL]" / "[link]" placeholders not followed by '('
    placeholder_re = re.compile(
        r"(?:Retrieved\s+from\s+)?\[(?:Link|URL|link|url)\](?!\()",
        re.IGNORECASE,
    )

    fixed_lines = []
    for line in report.splitlines():
        if not placeholder_re.search(line):
            fixed_lines.append(line)
            continue

        if not queue:
            # No real sources: drop "Retrieved from [Link]" entirely, keep rest
            cleaned = placeholder_re.sub("", line).rstrip(" ,;.")
            fixed_lines.append(cleaned)
            continue

        # Replace each placeholder occurrence with a real Markdown link
        def _replace(_match, _line=line):
            src = _pick_match(_line)
            if not src:
                return ""
            label = src["title"] or src["url"]
            # Keep label short
            if len(label) > 80:
                label = label[:77] + "..."
            return f"[{label}]({src['url']})"

        fixed_lines.append(placeholder_re.sub(_replace, line))

    return "\n".join(fixed_lines)


def _build_rag_graph(report: str, sources: list) -> str:
    """Build a Mermaid graph string linking findings <-> retrieved sources.

    Heuristic: detect known CXR finding terms in the report; link each finding
    to all retrieved sources whose title/snippet mentions the same term.
    If there are no sources we still draw a graph of detected findings so the
    user gets visual feedback even without web-search RAG.
    """
    report_low = (report or "").lower()
    findings = []
    for term in CXR_FINDING_TERMS:
        if term in report_low:
            findings.append(term)

    # Sanitize node ids for mermaid
    def _nid(prefix, idx):
        return f"{prefix}{idx}"

    lines = ["graph LR"]
    lines.append("    IMG[\"🖼️ Chest X-ray\"]:::img")
    lines.append("    REP[\"📝 Generated Report\"]:::rep")
    lines.append("    IMG --> REP")

    # Source nodes
    src_ids = []
    for i, s in enumerate(sources):
        sid = _nid("S", i)
        title = (s.get("title") or s.get("url") or "source").replace('"', "'")
        if len(title) > 60:
            title = title[:60] + "…"
        try:
            from urllib.parse import urlparse
            host = urlparse(s.get("url") or "").netloc or "web"
        except Exception:
            host = "web"
        lines.append(f'    {sid}["🔗 {title}<br/><i>{host}</i>"]:::src')
        src_ids.append((sid, s))

    if findings:
        for j, f in enumerate(findings):
            fid = _nid("F", j)
            lines.append(f'    {fid}(["🩺 {f.title()}"]):::find')
            lines.append(f"    REP --> {fid}")
            # Link finding to sources whose title/snippet mention the same term
            for sid, s in src_ids:
                hay = (str(s.get("title", "")) + " " + str(s.get("snippet", ""))).lower()
                if f in hay:
                    lines.append(f"    {fid} -.->|defined_by| {sid}")
    elif src_ids:
        # No specific findings detected: connect REP to all sources
        for sid, _ in src_ids:
            lines.append(f"    REP -.->|cites| {sid}")
    else:
        # No sources and no findings: just show a placeholder note
        lines.append('    NOTE["ℹ️ No web-search sources or known findings detected.<br/>Enable Web Search RAG to retrieve evidence."]:::note')
        lines.append("    REP --> NOTE")
        lines.append("    classDef note fill:#f5f5f5,stroke:#999,stroke-dasharray:4 4;")

    # Styling
    lines.append("    classDef img fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;")
    lines.append("    classDef rep fill:#fff3e0,stroke:#e65100,stroke-width:2px;")
    lines.append("    classDef find fill:#fce4ec,stroke:#ad1457,stroke-width:1.5px;")
    lines.append("    classDef src fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px;")
    return "\n".join(lines)


def _render_mermaid_html(mermaid_code: str) -> str:
    """Render a Mermaid graph as inline HTML.

    Strategy: use the public mermaid.ink renderer, which takes a base64-encoded
    mermaid source in the URL path and returns a rendered SVG/PNG. This avoids
    any client-side <script> tag, which Gradio 6.x's HTML sanitizer strips out.
    The raw source is still shown in a collapsible <details> block so users can
    copy it elsewhere.
    """
    if not mermaid_code:
        return (
            "<div style='padding:1em;color:#888;font-style:italic;'>"
            "💡 Generate a report (with or without web search) to see the RAG graph."
            "</div>"
        )

    # Encode mermaid source for mermaid.ink (URL-safe base64, no padding strip)
    try:
        encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("ascii")
    except Exception as e:
        return f"<pre style='color:red'>Mermaid encode error: {e}</pre>"

    img_url_svg = f"https://mermaid.ink/svg/{encoded}"
    img_url_png = f"https://mermaid.ink/img/{encoded}?type=png&bgColor=ffffff"
    live_url = f"https://mermaid.live/edit#base64:{encoded}"

    # Escape source for display in <pre>
    safe_src = (
        mermaid_code
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    return f"""
<div style=\"width:100%;overflow:auto;background:#fafafa;border:1px solid #ddd;border-radius:8px;padding:12px;margin-bottom:8px;text-align:center;\">
  <img src=\"{img_url_svg}\"
       alt=\"RAG Knowledge Graph\"
       style=\"max-width:100%;height:auto;\"
       onerror=\"this.onerror=null;this.src='{img_url_png}';\" />
</div>
<div style=\"font-size:0.85em;color:#666;margin-bottom:8px;\">
  🔗 <a href=\"{live_url}\" target=\"_blank\" rel=\"noopener\">Open in mermaid.live editor</a>
  &nbsp;·&nbsp;
  <a href=\"{img_url_png}\" target=\"_blank\" rel=\"noopener\">Download PNG</a>
</div>
<details style=\"margin-top:4px;\"><summary style=\"cursor:pointer;color:#555;font-size:0.85em;\">📝 View Mermaid source</summary>
<pre style=\"background:#f5f5f5;padding:8px;border-radius:6px;font-size:0.8em;overflow:auto;white-space:pre-wrap;\">{safe_src}</pre>
</details>
"""


def _build_web_search_kwargs(
    provider: str,
    search_hint: str,
    domain_whitelist: str,
) -> dict:
    """Build the provider-specific kwargs for OpenAI-compatible web search."""
    domains = [d.strip() for d in (domain_whitelist or "").split(",") if d.strip()]
    hint = (search_hint or "").strip()

    if provider == "zhipu":
        web_search_cfg = {
            "enable": True,
            "search_result": True,
        }
        if hint:
            web_search_cfg["search_query"] = hint
        return {
            "extra_body": {
                "tools": [{"type": "web_search", "web_search": web_search_cfg}]
            }
        }

    if provider == "qwen":
        # DashScope OpenAI-compatible mode
        extra_body = {"enable_search": True}
        if hint:
            extra_body["search_options"] = {"forced_search": True, "search_strategy": "pro"}
        return {"extra_body": extra_body}

    if provider == "anthropic":
        tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
        if domains:
            tool["allowed_domains"] = domains
        return {"extra_body": {"tools": [tool]}}

    # provider == "openai" (default)
    tool = {"type": "web_search"}
    if domains:
        tool["filters"] = {"allowed_domains": domains}
    return {"tools": [tool]}


def generate_report_api(
    image_path: str,
    api_key: str,
    api_base: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    enable_web_search: bool = False,
    search_hint: str = "",
    domain_whitelist: str = "",
) -> str:
    """Generate a report using an OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        return "❌ Error: openai package not installed. Run: pip install openai"

    if not api_key or not api_key.strip():
        return "❌ Error: Please provide an API key."
    if not image_path:
        return "❌ Error: Please upload an image."

    # Encode image
    base64_image = encode_image_to_base64(image_path)

    # Determine image MIME type
    ext = Path(image_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif"}
    mime_type = mime_map.get(ext, "image/jpeg")

    client = OpenAI(
        api_key=api_key,
        base_url=api_base if api_base.strip() else None,
        timeout=180.0,  # extra time when web search is enabled
    )

    # Augment prompt when web search is enabled, so the model knows to ground
    # its answer with retrieved evidence and cite sources.
    final_prompt = prompt
    if enable_web_search:
        guidance = (
            "\n\nYou have access to a web_search tool. Before writing the report, "
            "search authoritative medical sources (e.g., Radiopaedia, PhysioNet, PubMed/NCBI) "
            "for findings that are relevant to this chest X-ray. Use the retrieved "
            "information to ground your description and terminology. "
            "\n\nIMPORTANT formatting rules for the final report:\n"
            "1. Use proper Markdown: `**Section:**` for headings and `- ` for bullets.\n"
            "2. End with a `**Citations:**` section.\n"
            "3. For each citation, write the FULL clickable URL inline as a Markdown link, "
            "e.g. `- [Radiopaedia \u2014 Cardiomegaly](https://radiopaedia.org/articles/cardiomegaly)`. "
            "NEVER use placeholder text like `[Link]`, `[URL]`, or `Retrieved from [Link]` \u2014 "
            "only emit citations for sources you actually retrieved with real URLs."
        )
        if search_hint and search_hint.strip():
            guidance += f"\nSearch focus hint: {search_hint.strip()}"
        final_prompt = prompt + guidance

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": final_prompt,
                }
            ]
        }
    ]

    extra_kwargs = {}
    provider_used = ""
    if enable_web_search:
        provider_used = _detect_provider(model_name, api_base)
        extra_kwargs = _build_web_search_kwargs(
            provider_used, search_hint, domain_whitelist
        )
        print("=" * 70)
        print(f"[RAG DEBUG] enable_web_search=True, provider={provider_used}")
        print(f"[RAG DEBUG] extra_kwargs sent to API:")
        print(json.dumps(extra_kwargs, indent=2, ensure_ascii=False)[:2000])
        print("=" * 70)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            **extra_kwargs,
        )
        report = response.choices[0].message.content
        # Clean thinking tags if present
        report = re.sub(r'<think>.*?</think>', '', report, flags=re.DOTALL).strip()
        sources = []

        if enable_web_search and provider_used:
            # Dump the raw response so we can see exactly what came back
            try:
                if hasattr(response, "model_dump"):
                    raw = response.model_dump()
                elif hasattr(response, "to_dict"):
                    raw = response.to_dict()
                else:
                    raw = json.loads(json.dumps(response, default=lambda o: getattr(o, "__dict__", str(o))))
                dump_path = Path(tempfile.gettempdir()) / "last_api_response.json"
                dump_path.write_text(
                    json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(f"[RAG DEBUG] Full raw response written to: {dump_path}")
                # Print top-level keys + a peek at message keys
                print(f"[RAG DEBUG] Response top-level keys: {list(raw.keys())}")
                ch = (raw.get("choices") or [{}])[0]
                msg = ch.get("message", {}) or {}
                print(f"[RAG DEBUG] message keys: {list(msg.keys())}")
                # Print any web-search-ish fields directly
                for k in ("web_search", "search_result", "search_results",
                          "annotations", "tool_calls"):
                    if k in msg and msg[k]:
                        peek = json.dumps(msg[k], ensure_ascii=False)[:600]
                        print(f"[RAG DEBUG] message.{k}: {peek}")
                    if k in raw and raw[k]:
                        peek = json.dumps(raw[k], ensure_ascii=False)[:600]
                        print(f"[RAG DEBUG] root.{k}: {peek}")
                if "search_info" in raw:
                    peek = json.dumps(raw["search_info"], ensure_ascii=False)[:600]
                    print(f"[RAG DEBUG] root.search_info: {peek}")
            except Exception as _e:
                print(f"[RAG DEBUG] Could not dump response: {_e}")

            sources = _extract_search_results(response, provider_used)
            print(f"[RAG DEBUG] Extracted {len(sources)} sources")
            for i, s in enumerate(sources[:5]):
                print(f"  [{i+1}] {s.get('title','')[:60]} -> {s.get('url','')[:100]}")

            tag = f"🔎 Web search enabled · provider: {provider_used} · sources retrieved: {len(sources)}"
            report = f"{report}\n\n_({tag})_"
        return report, sources
    except Exception as e:
        # If the chosen provider does not support web_search, retry without tools.
        err = str(e)
        if enable_web_search and (
            "web_search" in err.lower() or "tool" in err.lower() or "unsupported" in err.lower()
        ):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                report = response.choices[0].message.content
                report = re.sub(r'<think>.*?</think>', '', report, flags=re.DOTALL).strip()
                report += (
                    "\n\n⚠️ _Web search was requested but the API/model does not seem "
                    "to support the web_search tool; report was generated without retrieval._"
                )
                return report, []
            except Exception as e2:
                return f"❌ API Error (after web_search fallback): {str(e2)}", []
        return f"❌ API Error: {err}", []


# ============================================================
# Report Generation - Local Mode
# ============================================================

# Global model cache
_local_model = None
_local_processor = None
_local_model_name = None


def load_local_model(model_name: str):
    """Load a local HuggingFace model (cached globally)."""
    global _local_model, _local_processor, _local_model_name

    if _local_model_name == model_name and _local_model is not None:
        return _local_model, _local_processor

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[INFO] Loading local model: {model_name}")
    _local_model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _local_processor = AutoProcessor.from_pretrained(model_name)
    _local_model_name = model_name
    print(f"[INFO] Model loaded successfully.")
    return _local_model, _local_processor


def generate_report_local(
    image_path: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
) -> str:
    """Generate a report using a local HuggingFace model."""
    if not image_path:
        return "❌ Error: Please upload an image."

    try:
        import torch
        from qwen_vl_utils import process_vision_info
    except ImportError:
        return "❌ Error: Required packages not installed. Run: pip install torch transformers qwen-vl-utils"

    try:
        model, processor = load_local_model(model_name)
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ]
        }
    ]

    try:
        image_inputs, video_inputs = process_vision_info(messages)
        text = processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_tokens)

        generated_ids_trimmed = generated[0][inputs["input_ids"].shape[-1]:]
        output_text = processor.decode(generated_ids_trimmed, skip_special_tokens=True)
        return output_text
    except Exception as e:
        return f"❌ Error during generation: {str(e)}"


# ============================================================
# Main Gradio Interface
# ============================================================

def generate_and_evaluate(
    image,
    mode,
    api_key,
    api_base,
    model_name,
    prompt,
    max_tokens,
    ground_truth,
    enable_web_search,
    search_hint,
    domain_whitelist,
):
    """Main function: generate report, extract sources, build RAG graph, evaluate."""
    empty_table = []
    empty_graph = _render_mermaid_html("")

    if image is None:
        return "❌ Please upload a chest X-ray image.", "", empty_table, empty_graph

    # Generate report
    start_time = time.time()
    sources = []

    if mode == "API (OpenAI-compatible)":
        report, sources = generate_report_api(
            image_path=image,
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            prompt=prompt,
            max_tokens=int(max_tokens),
            enable_web_search=bool(enable_web_search),
            search_hint=search_hint or "",
            domain_whitelist=domain_whitelist or "",
        )
    else:  # Local Model
        report = generate_report_local(
            image_path=image,
            model_name=model_name,
            prompt=prompt,
            max_tokens=int(max_tokens),
        )
        if enable_web_search:
            report = (
                "⚠️ Web search is only available in API mode and was ignored.\n\n"
                + report
            )

    elapsed = time.time() - start_time

    # Post-process: replace placeholder citation links like `[Link]`, `[URL]`,
    # or `Retrieved from [Link]` with real URLs from retrieved sources.
    report = _fix_placeholder_citations(report, sources)

    # Add generation info as a styled footer (Markdown)
    report_display = (
        f"{report}\n\n---\n"
        f"<sub>⏱️ Generation time: {elapsed:.1f}s &nbsp;|&nbsp; "
        f"🤖 Model: `{model_name}`</sub>"
    )

    # Evaluate if ground truth provided
    if ground_truth and ground_truth.strip() and not report.startswith("❌"):
        metrics_output = compute_all_metrics(ground_truth, report)
    else:
        if report.startswith("❌"):
            metrics_output = "⚠️ Report generation failed. Cannot compute metrics."
        else:
            metrics_output = "💡 Provide a ground truth report (left panel) to compute evaluation metrics."

    # Build sources table + RAG graph. Even when web search returned nothing
    # we still draw a graph based on the findings detected in the report,
    # so users can see the structure.
    sources_rows = _format_sources_for_table(sources)
    graph_code = _build_rag_graph(report, sources)
    graph_html = _render_mermaid_html(graph_code)

    return report_display, metrics_output, sources_rows, graph_html


def build_app():
    """Build the Gradio interface."""

    custom_css = """
    * {
        font-family: Georgia, 'Times New Roman', Times, serif !important;
    }
    code, pre, .code, .mono {
        font-family: 'Courier New', Courier, monospace !important;
    }
    """

    with gr.Blocks(
        title="🏥 Radiology Report Generation & Evaluation",
        css=custom_css,
    ) as app:
        gr.Markdown("""
# 🏥 Radiology Report Generation & Evaluation

Upload a chest X-ray image and generate a radiology report using an LLM.
Optionally provide a ground truth report to compute evaluation metrics (BLEU, ROUGE-L, METEOR).

**Two modes available:**
- **API mode**: Use any OpenAI-compatible API (vLLM, Ollama, OpenAI, Together AI, etc.)
- **Local mode**: Load a HuggingFace model locally (requires GPU)
        """)

        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")

                image_input = gr.Image(
                    label="Chest X-ray Image",
                    type="filepath",
                    height=300,
                )

                mode_selector = gr.Radio(
                    choices=["API (OpenAI-compatible)", "Local Model"],
                    value="API (OpenAI-compatible)",
                    label="Generation Mode",
                )

                with gr.Group() as api_group:
                    provider_preset = gr.Dropdown(
                        choices=list(API_PRESETS.keys()),
                        value="OpenAI (GPT-4o)",
                        label="API Provider Preset",
                        info="Select a provider to auto-fill the API Base URL and Model Name below.",
                    )
                    preset_hint = gr.Markdown(
                        value=f"💡 *{API_PRESETS['OpenAI (GPT-4o)']['hint']}*"
                    )
                    api_key_input = gr.Textbox(
                        label="API Key",
                        placeholder="sk-... or your API key",
                        type="password",
                    )
                    api_base_input = gr.Textbox(
                        label="API Base URL",
                        placeholder="https://api.openai.com/v1",
                        value=API_PRESETS["OpenAI (GPT-4o)"]["api_base"],
                    )

                model_name_input = gr.Textbox(
                    label="Model Name",
                    placeholder="e.g., gpt-4o, glm-4v-plus, qwen-vl-max",
                    value=API_PRESETS["OpenAI (GPT-4o)"]["model"],
                )

                prompt_input = gr.Textbox(
                    label="Prompt",
                    value="Please generate a detailed radiology report for this chest X-ray image. "
                          "Include findings and impressions.",
                    lines=3,
                )

                max_tokens_input = gr.Slider(
                    label="Max Tokens",
                    minimum=64,
                    maximum=4096,
                    value=512,
                    step=64,
                )

                gr.Markdown("### 🔎 Web Search RAG (optional, API mode)")
                enable_web_search_input = gr.Checkbox(
                    label="Enable web search (let the LLM retrieve from medical sites)",
                    value=False,
                )
                with gr.Group(visible=False) as web_search_group:
                    gr.Markdown(
                        "The LLM will autonomously search the web for relevant "
                        "information before writing the report. Provider is auto-detected "
                        "from the model name (OpenAI / Zhipu GLM / Qwen / Claude)."
                    )
                    search_hint_input = gr.Textbox(
                        label="Search focus hint (optional)",
                        placeholder="e.g., chest X-ray pneumonia consolidation findings",
                        lines=1,
                    )
                    domain_whitelist_input = gr.Textbox(
                        label="Domain whitelist (medical sites, comma-separated)",
                        value=DEFAULT_MEDICAL_DOMAINS,
                        lines=2,
                    )

                gr.Markdown("### 📋 Ground Truth (optional)")
                ground_truth_input = gr.Textbox(
                    label="Ground Truth Report",
                    placeholder="Paste the reference radiology report here for evaluation...",
                    lines=5,
                )

                generate_btn = gr.Button("🚀 Generate Report", variant="primary", size="lg")

            # Right column: Output
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Generated Report")
                report_output = gr.Markdown(
                    value="_The generated report will appear here (Markdown rendered)._",
                    height=420,
                    line_breaks=True,
                    container=True,
                    sanitize_html=False,
                    elem_id="report_md_output",
                )

                gr.Markdown("### 📊 Evaluation Metrics")
                metrics_output = gr.Markdown(
                    value="💡 Generate a report and provide ground truth to see metrics here."
                )

                with gr.Accordion("🔎 Retrieved Sources (RAG)", open=True):
                    sources_output = gr.Dataframe(
                        headers=["#", "Title", "URL", "Snippet"],
                        datatype=["number", "str", "str", "str"],
                        wrap=True,
                        row_count=(0, "dynamic"),
                        col_count=(4, "fixed"),
                        interactive=False,
                        label="Web search results retrieved by the LLM",
                    )

                with gr.Accordion("🕸️ RAG Knowledge Graph", open=True):
                    graph_output = gr.HTML(
                        value=_render_mermaid_html(""),
                        label="Findings ↔ Sources relationship graph",
                    )

        # Event handlers
        generate_btn.click(
            fn=generate_and_evaluate,
            inputs=[
                image_input,
                mode_selector,
                api_key_input,
                api_base_input,
                model_name_input,
                prompt_input,
                max_tokens_input,
                ground_truth_input,
                enable_web_search_input,
                search_hint_input,
                domain_whitelist_input,
            ],
            outputs=[report_output, metrics_output, sources_output, graph_output],
        )

        # Show/hide the web-search panel based on the checkbox
        enable_web_search_input.change(
            fn=lambda v: gr.update(visible=bool(v)),
            inputs=[enable_web_search_input],
            outputs=[web_search_group],
        )

        # Show/hide API fields based on mode
        def update_mode_visibility(mode):
            if mode == "API (OpenAI-compatible)":
                return gr.update(visible=True), gr.update(value="gpt-4o")
            else:
                return gr.update(visible=True), gr.update(value="Qwen/Qwen2.5-VL-7B-Instruct")

        mode_selector.change(
            fn=update_mode_visibility,
            inputs=[mode_selector],
            outputs=[api_group, model_name_input],
        )

        # Provider preset → auto-fill API Base URL + Model Name + hint
        def apply_preset(preset_name):
            cfg = API_PRESETS.get(preset_name, {})
            api_base = cfg.get("api_base", "")
            model = cfg.get("model", "")
            hint = cfg.get("hint", "")
            return (
                gr.update(value=api_base),
                gr.update(value=model),
                f"💡 *{hint}*" if hint else "",
            )

        provider_preset.change(
            fn=apply_preset,
            inputs=[provider_preset],
            outputs=[api_base_input, model_name_input, preset_hint],
        )

        # Examples
        gr.Markdown("""
---
### 💡 Tips

- **API Mode**: Works with any OpenAI-compatible endpoint. For local vLLM servers, set the base URL to `http://localhost:8000/v1`.
- **Local Mode**: Requires a GPU with sufficient VRAM. The model is cached after first load.
- **Metrics**: BLEU, ROUGE-L, and METEOR are computed instantly (no GPU needed).
- **Web Search RAG**: Lets the LLM autonomously retrieve from medical websites \
  (Radiopaedia, PhysioNet, PubMed/NCBI, etc.) before writing the report — \
  no local index or extra files needed. Provider is auto-routed:
  `gpt-*` → OpenAI · `glm-*` → Zhipu · `qwen-*` → DashScope · `claude-*` → Anthropic.
  If the chosen API does not support web search, the report is generated without retrieval.
- **Supported API providers**: OpenAI, Azure OpenAI, vLLM, Ollama (`http://localhost:11434/v1`), Together AI, Zhipu (`https://open.bigmodel.cn/api/paas/v4`), DashScope (`https://dashscope.aliyuncs.com/compatible-mode/v1`).
        """)

    return app


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Radiology Report Generation Web App")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server name/IP")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )
