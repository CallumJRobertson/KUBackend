# ... [Keep previous imports and setup] ...
# (The top part of the file remains the same. Replace the generate_briefing function at the bottom)

@app.route("/api/generate-briefing", methods=["POST"])
def generate_briefing():
    payload = request.get_json(force=True, silent=True) or {}
    updates = payload.get("updates", [])
    user_date_str = payload.get("userDate", time.strftime("%Y-%m-%d"))
    
    if not updates:
        return jsonify({"briefing": "No upcoming shows found."})

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
                    # We also allow shows from 'yesterday' (-1) just in case
                    if -1 <= days_diff <= 30:
                        valid_shows.append({
                            "title": title,
                            "date": air_date,
                            "days_diff": days_diff,
                            "date_str": date_str
                        })
                except ValueError:
                    continue # Skip invalid dates
                    
        # SORT: Soonest first
        valid_shows.sort(key=lambda x: x["date"])
        
    except Exception as e:
        print(f"Date parsing error: {e}")
        return jsonify({"briefing": "Unable to calculate dates."})

    if not valid_shows:
        return jsonify({"briefing": "It's quiet for now. Nothing on your list is airing in the next 30 days."})

    # 2. Build the AI Context
    # We take the top 5 most imminent shows
    top_shows = valid_shows[:5]
    
    show_descriptions = []
    for s in top_shows:
        # Create a human readable relative date
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
        return jsonify({"briefing": briefing})

    except Exception as e:
        print(f"Briefing Error: {e}", file=sys.stderr)
        return jsonify({"briefing": "Your shows are coming back soon! Check the list below."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
