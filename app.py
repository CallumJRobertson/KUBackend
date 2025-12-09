import os
import time
import textwrap
import hashlib
import json
import sys
from typing import Dict, Any, List
import redis 

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI

# ============================================================
# Configuration
# ============================================================

# Use standard spaces for indentation
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0") 
r = None # Redis connection object

# In the Initialization section:
try:
    r = redis.from_url(REDIS_URL, decode_responses=True)
    r.ping()
    print("✅ Redis connected successfully.")
except Exception as e:
    r = None
    print(f"⚠️ WARNING: Could not connect to Redis: {e}", file=sys.stderr)

# Safer Initialization logic
client = None

if not BRAVE_API_KEY:
    print("⚠️ CRITICAL: BRAVE_API_KEY is missing.", file=sys.stderr)

if not OPENAI_API_KEY:
    print("⚠️ CRITICAL: OPENAI_API_KEY is missing.", file=sys.stderr)
else:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"⚠️ Failed to init OpenAI: {e}", file=sys.stderr)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
CACHE_TTL_SECONDS = 60 * 60 * 12  # 12 hours

app = Flask(__name__)
CORS(app)

# ============================================================
# Caching
# ============================================================

_cache: Dict[str, Dict[str, Any]] = {}

def _cache_key(title: str, media_type: str) -> str:
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
# Logic
# ============================================================

def brave_search(show_title: str, media_type: str) -> List[dict]:
    if not BRAVE_API_KEY:
        print("Brave Search skipped: Missing API Key")
        return []

    if media_type == "tv":
        query = f'"{show_title}" new season release date status 2025'
    else:
        query = f'"{show_title}" movie sequel status news 2025'

    params = {
        "q": query,
        "country": "GB",
        "search_lang": "en",
        "ui_lang": "en-GB",
        "count": 5,
        "safesearch": "moderate",
    }
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }

    try:
        response = requests.get(BRAVE_SEARCH_URL, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("web", {}).get("results", [])[:3]
    except Exception as e:
        print(f"Brave Search Error: {e}", file=sys.stderr)
        return []

def summarise_with_openai(show_title: str, media_type: str, sources: List[dict]) -> str:
    if not client:
        return json.dumps({"status": "Unknown", "summary": "AI is unavailable (Missing Key)."})

    snippets = []
    for i, src in enumerate(sources, start=1):
        desc = src.get('description') or src.get('snippet') or "No description."
        snippets.append(f"Source {i}: {desc}")

    source_text = "\n\n".join(snippets)[:6000]

    # CHANGED: Prompt now asks for JSON
    prompt = textwrap.dedent(f"""
    Analyze these search results for "{show_title}" ({media_type}).
    
    Return a valid JSON object with exactly two fields:
    1. "status": One of ["Renewed", "Cancelled", "Concluded", "Released", "Unknown", "Ending", "In Production"]. 
       * Use "Concluded" if the show finished its final intended season (natural ending).
       * Use "Cancelled" only if the show was abruptly stopped by the network/studio with more seasons expected.
    2. "summary": A 2-sentence natural language summary of the situation.

    Search Results:
    {source_text}
    """)

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a TV show tracking assistant. Return JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=250,
            temperature=0.2,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI Error: {e}", file=sys.stderr)
        # Fallback JSON if AI fails
        return json.dumps({"status": "Unknown", "summary": "Could not generate summary."})

# ============================================================
# Routes
# ============================================================

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "KeepUp backend"})

@app.route("/api/show-status", methods=["POST"])
def show_status():
    payload = request.get_json(force=True, silent=True) or {}
    show_name = payload.get("showName", "").strip()
    is_tv = payload.get("isTV", True)
    
    if isinstance(is_tv, str):
        is_tv = is_tv.lower() == 'true'
    media_type = "tv" if is_tv else "movie"

    if not show_name:
        return jsonify({"error": "showName is required"}), 400

    key = _cache_key(show_name, media_type)
    cached = get_cached_result(key)
    if cached:
        return jsonify({**cached, "cached": True})

    try:
        sources = brave_search(show_name, media_type)
        
        # If no sources, return empty
        if not sources:
            result = {
                "status": "Unknown",
                "summary": "No recent information found.",
                "sources": []
            }
            set_cached_result(key, result)
            return jsonify(result)

        # Get JSON string from OpenAI
        ai_json_string = summarise_with_openai(show_name, media_type, sources)
        
        # Parse the JSON string to actual object
        try:
            ai_data = json.loads(ai_json_string)
        except:
            ai_data = {"status": "Unknown", "summary": ai_json_string}

        result = {
            "status": ai_data.get("status", "Unknown"),
            "summary": ai_data.get("summary", "No summary available."),
            "sources": [{"title": s.get("title"), "url": s.get("url")} for s in sources]
        }

        set_cached_result(key, result)
        return jsonify(result)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
