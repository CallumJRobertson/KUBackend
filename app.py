import os
import time
import textwrap
import hashlib
from typing import Dict, Any, List

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI

# ============================================================
# Configuration
# ============================================================

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BRAVE_API_KEY:
    raise RuntimeError("BRAVE_API_KEY environment variable is not set")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

CACHE_TTL_SECONDS = 60 * 60 * 12  # 12 hours

# ============================================================
# App
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# Simple in-memory cache
# ============================================================

_cache: Dict[str, Dict[str, Any]] = {}


def _cache_key(title: str, media_type: str) -> str:
    """Create a stable cache key for a title + type."""
    raw = f"{title.lower().strip()}|{media_type}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached_result(key: str):
    entry = _cache.get(key)
    if not entry:
        return None

    if time.time() - entry["timestamp"] > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None

    return entry["data"]


def set_cached_result(key: str, data: dict):
    _cache[key] = {
        "timestamp": time.time(),
        "data": data
    }

# ============================================================
# Brave Search
# ============================================================

def brave_search(show_title: str, media_type: str) -> List[dict]:
    """
    Search Brave for recent articles about a show's status.
    """
    if media_type == "tv":
        query = f'new season of "{show_title}" status release date'
    else:
        query = f'"{show_title}" movie sequel follow-up status'

    params = {
        "q": query,
        "country": "GB",
        "search_lang": "en",
        "ui_lang": "en-GB",
        "count": 10,
        "safesearch": "moderate",
    }

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY,
    }

    response = requests.get(
        BRAVE_SEARCH_URL,
        params=params,
        headers=headers,
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    results = data.get("web", {}).get("results", [])
    return results[:3]

# ============================================================
# OpenAI (GPT-5 nano)
# ============================================================

def summarise_with_openai(show_title: str, media_type: str, sources: List[dict]) -> str:
    """
    Ask GPT-5 nano to summarise the consensus from search snippets.
    """

    snippets = []
    for i, src in enumerate(sources, start=1):
        snippets.append(
            f"""
            Source {i}:
            Title: {src.get('title')}
            URL: {src.get('url')}
            Snippet: {src.get('description')}
            """.strip()
        )

    source_text = "\n\n".join(snippets)[:6000]  # safety truncation

    prompt = textwrap.dedent(f"""
    You help users track TV shows and movies.

    Based ONLY on the web snippets below, determine the most likely
    current status of the next season or continuation of:

    "{show_title}"

    Rules:
    - Be factual and honest about uncertainty.
    - If nothing official exists, say so clearly.
    - If sources conflict, mention that.
    - Keep the response concise (2–4 sentences).

    Web snippets:
    {source_text}
    """)

    completion = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": "You summarise TV show and movie status from web snippets."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=250,
        temperature=0.2,
    )

    return completion.choices[0].message.content.strip()

# ============================================================
# Routes
# ============================================================

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "KeepUp backend",
        "cache_items": len(_cache)
    })


@app.route("/api/show-status", methods=["POST"])
def show_status():
    """
    POST /api/show-status
    Body:
    {
        "showName": "The Bear",
        "isTV": true
    }
    """
    payload = request.get_json(force=True, silent=True) or {}

    show_name = payload.get("showName", "").strip()
    is_tv = bool(payload.get("isTV", True))
    media_type = "tv" if is_tv else "movie"

    if not show_name:
        return jsonify({"error": "showName is required"}), 400

    key = _cache_key(show_name, media_type)
    cached = get_cached_result(key)
    if cached:
        return jsonify({
            **cached,
            "cached": True
        })

    try:
        sources = brave_search(show_name, media_type)

        if not sources:
            result = {
                "summary": f"No reliable recent information found about {show_name}.",
                "sources": [],
            }
            set_cached_result(key, result)
            return jsonify(result)

        summary = summarise_with_openai(show_name, media_type, sources)

        result = {
            "summary": summary,
            "sources": [
                {
                    "title": s.get("title"),
                    "url": s.get("url")
                } for s in sources
            ],
        }

        set_cached_result(key, result)
        return jsonify(result)

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "error": "Failed to determine show status"
        }), 500


# ============================================================
# Local dev
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
