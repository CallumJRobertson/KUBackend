import os
import time
import textwrap
import hashlib
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI
import redis
import pymongo 

# ============================================================
# Configuration
# ============================================================

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0") 
MONGO_URI = os.getenv("MONGO_URI")

# Connection Objects
client = None
r = None
db_collection = None

# Initialization Logic
if not BRAVE_API_KEY or not OPENAI_API_KEY:
    print("⚠️ CRITICAL: API Keys missing.", file=sys.stderr)
else:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"⚠️ Failed to init OpenAI: {e}", file=sys.stderr)

# --- 1. Initialize Redis (Tier 1: Speed) ---
try:
    r = redis.from_url(REDIS_URL, decode_responses=True)
    r.ping()
    print("✅ Redis connected successfully.")
except Exception as e:
    r = None
    print(f"⚠️ WARNING: Redis connection failed: {e}", file=sys.stderr)

# --- 2. Initialize MongoDB (Tier 2: Persistence) ---
if MONGO_URI:
    try:
        mongo_client = pymongo.MongoClient(MONGO_URI)
        db_collection = mongo_client.get_database("keepup_db").get_collection("ai_status")
        db_collection.create_index([("_id", pymongo.ASCENDING)]) 
        print("✅ MongoDB connected successfully.")
    except Exception as e:
        db_collection = None
        print(f"⚠️ WARNING: MongoDB connection failed: {e}", file=sys.stderr)

# --- CACHING CONSTANTS ---
DEFAULT_CACHE_TTL_SECONDS = 60 * 60 * 12
PERMANENT_STATUSES = ["cancelled", "concluded", "ended"] 
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

app = Flask(__name__)
CORS(app)

# ============================================================
# Caching Helper Functions
# ============================================================

def _cache_key(title: str, media_type: str) -> str:
    raw = f"{title.lower().strip()}|{media_type}"
    return hashlib.sha256(raw.encode()).hexdigest()

def get_cached_result(key: str):
    # TIER 1: Redis
    if r is not None:
        try:
            json_data = r.get(key)
            if json_data: return json.loads(json_data)
        except Exception: pass

    # TIER 2: MongoDB
    if db_collection is not None:
        try:
            document = db_collection.find_one({"_id": key})
            if document:
                if time.time() < document.get("expiry_time", 0):
                    if r is not None: set_cached_result(key, document['data'], status="WARM") 
                    return document['data']
                else:
                    db_collection.delete_one({"_id": key})
        except Exception: pass
    return None

def set_cached_result(key: str, data: dict, status: str):
    status_lower = status.lower()
    
    # 1. Save to Redis
    if r is not None:
        try: r.set(key, json.dumps(data), ex=DEFAULT_CACHE_TTL_SECONDS)
        except Exception: pass

    # 2. Save to MongoDB (Persistent)
    if status_lower in PERMANENT_STATUSES or status_lower == "active":
        if db_collection is not None:
            expiry = int(time.time() + (60 * 60 * 24 * 365 * 100)) 
            try:
                db_collection.update_one({"_id": key}, {"$set": {"data": data, "expiry_time": expiry}}, upsert=True)
            except Exception: pass

# ============================================================
# Core Logic (Brave + OpenAI)
# ============================================================

def brave_search(show_title: str, media_type: str) -> List[dict]:
    if not BRAVE_API_KEY: return []
    query = f'"{show_title}" new season release date status 2025' if media_type == "tv" else f'"{show_title}" movie sequel status news 2025'
    
    try:
        response = requests.get(BRAVE_SEARCH_URL, params={"q": query, "count": 5}, headers={"X-Subscription-Token": BRAVE_API_KEY})
        return response.json().get("web", {}).get("results", [])[:3]
    except Exception: return []

def summarise_with_openai(show_title: str, media_type: str, sources: List[dict], current_date: str) -> str:
    if not client: return json.dumps({"status": "Unknown", "summary": "AI unavailable."})
    
    snippets = "\n".join([f"Src {i}: {s.get('description','')} " for i, s in enumerate(sources)])
    prompt = f"""
    Analyze results for "{show_title}" ({media_type}). Date: {current_date}.
    Return JSON: {{"status": "Renewed/Cancelled/...", "summary": "2 sentences max"}}
    
    Results:
    {snippets[:4000]}
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=250
        )
        return completion.choices[0].message.content.strip()
    except Exception: return json.dumps({"status": "Unknown", "summary": "Error."})

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
    is_tv = str(payload.get("isTV", True)).lower() == 'true'
    current_date = payload.get("currentDate", time.strftime("%Y-%m-%d"))

    if not show_name: return jsonify({"error": "showName required"}), 400

    media_type = "tv" if is_tv else "movie"
    key = _cache_key(show_name, media_type)
    
    cached = get_cached_result(key)
    if cached: return jsonify({**cached, "cached": True})

    sources = brave_search(show_name, media_type)
    ai_json = summarise_with_openai(show_name, media_type, sources, current_date)
    
    try: ai_data = json.loads(ai_json)
    except: ai_data = {"status": "Unknown", "summary": ai_json}

    result = {
        "status": ai_data.get("status", "Unknown"),
        "summary": ai_data.get("summary", "No summary."),
        "sources": [{"title": s.get("title"), "url": s.get("url")} for s in sources]
    }
    set_cached_result(key, result, status=result["status"])
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
