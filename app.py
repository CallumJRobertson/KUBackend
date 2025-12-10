import os
import time
import textwrap
import hashlib
import json
import sys
from typing import Dict, Any, List

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI
import redis
import pymongo # MongoDB client

# ============================================================
# Configuration
# ============================================================

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0") 
MONGO_URI = os.getenv("MONGO_URI") # MongoDB Atlas Connection String

# Connection Objects
client = None
r = None        # Redis connection
db_collection = None # MongoDB collection object

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
    print(f"⚠️ WARNING: Redis connection failed (speed cache is unavailable): {e}", file=sys.stderr)

# --- 2. Initialize MongoDB (Tier 2: Persistence) ---
if MONGO_URI:
    try:
        mongo_client = pymongo.MongoClient(MONGO_URI)
        db_collection = mongo_client.get_database("keepup_db").get_collection("ai_status")
        # FIX: Removed redundant unique=True argument on _id index
        db_collection.create_index([("_id", pymongo.ASCENDING)]) 
        print("✅ MongoDB connected successfully.")
    except Exception as e:
        db_collection = None # Set to None on failure
        print(f"⚠️ WARNING: MongoDB connection failed (persistent cache unavailable): {e}", file=sys.stderr)

# --- CACHING CONSTANTS ---
DEFAULT_CACHE_TTL_SECONDS = 60 * 60 * 12
PERMANENT_STATUSES = ["cancelled", "concluded", "ended"] 

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
app = Flask(__name__)
CORS(app)

# ============================================================
# Tiered Caching Functions
# ============================================================

def _cache_key(title: str, media_type: str) -> str:
    # Key format: 'sha256hash'
    raw = f"{title.lower().strip()}|{media_type}"
    return hashlib.sha256(raw.encode()).hexdigest()

def get_cached_result(key: str):
    # TIER 1: Check Redis (Fastest)
    if r is not None:
        try:
            json_data = r.get(key)
            if json_data:
                print("Cache HIT: Redis (Tier 1)")
                return json.loads(json_data)
        except Exception as e:
            print(f"Redis GET Error: {e}", file=sys.stderr)

    # TIER 2: Check MongoDB (Persistent)
    if db_collection is not None:
        try:
            document = db_collection.find_one({"_id": key})
            if document:
                expiry_time = document.get("expiry_time", 0) 
                
                if time.time() < expiry_time:
                    print("Cache HIT: MongoDB (Tier 2)")
                    
                    # Cache Warming: Push back to Redis for next time
                    if r is not None: set_cached_result(key, document['data'], status="WARM") 
                    
                    return document['data']
                else:
                    # Delete stale entry from MongoDB
                    db_collection.delete_one({"_id": key})
                    
        except Exception as e:
            print(f"MongoDB GET Error: {e}", file=sys.stderr)
            
    print("Cache MISS")
    return None

def set_cached_result(key: str, data: dict, status: str):
    status_lower = status.lower()
    
    # 1. Save to REDIS (Tier 1) for 12 hours (always)
    if r is not None:
        try:
            r.set(key, json.dumps(data), ex=DEFAULT_CACHE_TTL_SECONDS)
        except Exception as e:
            print(f"Redis SET Error: {e}", file=sys.stderr)

    # 2. Save to MONGODB (Tier 2) only if status is permanent
    if status_lower in PERMANENT_STATUSES:
        if db_collection is not None:
            # Set expiry far in the future (100 years)
            expiry_time = int(time.time() + (60 * 60 * 24 * 365 * 100)) 
            
            try:
                db_collection.update_one(
                    {"_id": key},
                    {"$set": {
                        "data": data,
                        "expiry_time": expiry_time
                    }},
                    upsert=True # Insert if not found, update if found
                )
            except Exception as e:
                print(f"MongoDB SET Error: {e}", file=sys.stderr)


# ============================================================
# Logic & Routes
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

def summarise_with_openai(show_title: str, media_type: str, sources: List[dict], current_date: str) -> str:
    if not client:
        return json.dumps({"status": "Unknown", "summary": "AI is unavailable (Missing Key)."})

    snippets = []
    for i, src in enumerate(sources, start=1):
        desc = src.get('description') or src.get('snippet') or "No description."
        snippets.append(f"Source {i}: {desc}")

    source_text = "\n\n".join(snippets)[:6000]

    # ADDED: Current date to the prompt for context
    prompt = textwrap.dedent(f"""
    Analyze these search results for "{show_title}" ({media_type}). The current date is {current_date}.
    
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
        return json.dumps({"status": "Unknown", "summary": "Could not generate summary."})

@app.route("/", methods=["GET"])
def health():
    # FIX: Correct checks for Pythonic truthiness
    redis_status = "OK" if r is not None else "Unavailable" 
    mongo_status = "OK" if db_collection is not None else "Unavailable" 
    
    return jsonify({
        "status": "ok", 
        "service": "KeepUp backend",
        "redis_status": redis_status,
        "mongo_status": mongo_status
    })

@app.route("/api/show-status", methods=["POST"])
def show_status():
    payload = request.get_json(force=True, silent=True) or {}
    show_name = payload.get("showName", "").strip()
    is_tv = payload.get("isTV", True)
    
    if isinstance(is_tv, str):
        is_tv = is_tv.lower() == 'true'
    media_type = "tv" if is_tv else "movie"
    
    # NEW: Extract current date from app, or use server date as fallback
    current_date = payload.get("currentDate", time.strftime("%Y-%m-%d"))

    if not show_name:
        return jsonify({"error": "showName is required"}), 400

    key = _cache_key(show_name, media_type)
    cached = get_cached_result(key)
    if cached:
        return jsonify({**cached, "cached": True})

    try:
        sources = brave_search(show_name, media_type)
        
        if not sources:
            result = {
                "status": "Unknown",
                "summary": "No recent information found.",
                "sources": []
            }
            set_cached_result(key, result, status=result["status"]) 
            return jsonify(result)

        # PASS the current_date to the summarization function
        ai_json_string = summarise_with_openai(show_name, media_type, sources, current_date)
        
        try:
            ai_data = json.loads(ai_json_string)
        except:
            ai_data = {"status": "Unknown", "summary": ai_json_string}

        result = {
            "status": ai_data.get("status", "Unknown"),
            "summary": ai_data.get("summary", "No summary available."),
            "sources": [{"title": s.get("title"), "url": s.get("url")} for s in sources]
        }

        set_cached_result(key, result, status=result["status"])
        return jsonify(result)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

# ============================================================
# NEW: Briefing Endpoint
# ============================================================
@app.route("/api/generate-briefing", methods=["POST"])
def generate_briefing():
    # 1. Get the list of shows from the iOS app
    payload = request.get_json(force=True, silent=True) or {}
    updates = payload.get("updates", [])
    
    if not updates:
        return jsonify({"briefing": "No upcoming shows? Time to find some new favorites!"})

    # 2. Prepare the list for the AI (take first 5)
    shows_context = ", ".join([f"{u.get('title', 'Unknown')} ({u.get('network', 'TV')})" for u in updates[:5]])

    # 3. The Prompt
    prompt_text = f"""
    The user tracks these TV shows which have new episodes coming up: {shows_context}.
    
    Write a short, high-energy "Morning Briefing" (2-3 sentences max) for the user.
    - Be conversational and hype them up (e.g. "Clear your schedule", "The wait is over").
    - Mention specific show names from the list.
    - Do NOT use "Here is your summary" or robot speak. Jump right into the hype.
    """

    try:
        if not client:
            raise Exception("OpenAI client not initialized")

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a witty TV tracking assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        briefing = completion.choices[0].message.content.strip()
        return jsonify({"briefing": briefing})

    except Exception as e:
        print(f"Briefing Error: {e}", file=sys.stderr)
        return jsonify({"briefing": "Your shows are returning soon! Check the list below for details."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
