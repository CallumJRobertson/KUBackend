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

# ✅ THIS WAS MISSING IN YOUR ERROR:
app = Flask(__name__)
CORS(app)

# ============================================================
# Caching Helper Functions
# ============================================================

def _cache_key(title: str, media_type: str) -> str:
    raw = f"{title.lower().strip()}|{media_type}"
    return hashlib.sha256(raw.encode()).hexdigest()

def _briefing_cache_key(updates: List[dict], user_date: str) -> str:
    # Sort updates by title so the order doesn't break the hash
    sorted_updates = sorted(updates, key=lambda x: x.get('title', ''))
    titles = "|".join([u.get('title', '') for u in sorted_updates])
    raw = f"briefing|{user_date}|{titles}"
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

# ============================================================
# Smart Briefing Endpoint
# ============================================================
@app.route("/api/generate-briefing", methods=["POST"])
def generate_briefing():
    payload = request.get_json(force=True, silent=True) or {}
    updates = payload.get("updates", [])
    user_date_str = payload.get("userDate", time.strftime("%Y-%m-%d"))
    
    if not updates:
        return jsonify({"briefing": "No upcoming shows found."})

    # --- CACHE CHECK ---
    cache_key = _briefing_cache_key(updates, user_date_str)
    cached_data = get_cached_result(cache_key)
    
    if cached_data:
        return jsonify({
            "briefing": cached_data.get("briefing"),
            "cached": True
        })
    # -------------------

    # 1. Parse and Sort Shows by Date
    valid_shows = []
    try:
        user_date = datetime.strptime(user_date_str, "%Y-%m-%d")
        
        for u in updates:
            date_str = u.get("nextAirDate")
            title = u.get("title")
            
            if date_str:
                try:
                    air_date = datetime.strptime(date_str, "%Y-%m-%d")
                    days_diff = (air_date - user_date).days
                    
                    # FILTER: Keep shows coming in the next 30 days only
                    if -1 <= days_diff <= 30:
                        valid_shows.append({
                            "title": title,
                            "date": air_date,
                            "days_diff": days_diff,
                            "date_str": date_str
                        })
                except ValueError:
                    continue 
                    
        valid_shows.sort(key=lambda x: x["date"])
        
    except Exception as e:
        print(f"Date parsing error: {e}")
        return jsonify({"briefing": "Unable to calculate dates."})

    if not valid_shows:
        return jsonify({"briefing": "It's quiet for now. Nothing on your list is airing in the next 30 days."})

    # 2. Build the AI Context
    top_shows = valid_shows[:5]
    show_descriptions = []
    for s in top_shows:
        days = s["days_diff"]
        if days == 0: when = "TODAY"
        elif days == 1: when = "Tomorrow"
        elif days < 7: when = f"this {s['date'].strftime('%A')}"
        else: when = f"on {s['date'].strftime('%b %d')}"
        
        show_descriptions.append(f"- {s['title']} ({when})")

    context_str = "\n".join(show_descriptions)

    # 3. The Prompt
    prompt_text = f"""
    The user tracks these TV shows which are airing very soon (Today is {user_date_str}):
    {context_str}
    
    Write a short, high-energy "Morning Briefing" (2-3 sentences max).
    - Prioritize the shows airing TODAY or TOMORROW if any.
    - Be conversational (e.g., "Clear your schedule for tonight", "The wait is finally over").
    - Mention specific show names.
    - Ignore shows that are weeks away unless there is nothing else.
    """

    try:
        if not client: raise Exception("No OpenAI")
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a hype-man for TV shows."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        briefing = completion.choices[0].message.content.strip()
        
        # --- CACHE SAVE ---
        result_data = {"briefing": briefing}
        set_cached_result(cache_key, result_data, status="active")
        
        return jsonify({"briefing": briefing, "cached": False})

    except Exception as e:
        print(f"Briefing Error: {e}", file=sys.stderr)
        return jsonify({"briefing": "Your shows are coming back soon! Check the list below."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
