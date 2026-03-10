"""
utils/chatbot.py
================
Skin‑wellness chatbot powered by Groq's FREE API
(llama-3.3-70b-versatile model — free tier as of 2026).

How to get a free Groq API key
-------------------------------
1. Visit https://console.groq.com/
2. Sign up (free, no credit card required for free tier)
3. Go to API Keys → Create API Key
4. Paste the key into the GROQ_API_KEY field in .env  OR
   set it as an environment variable:  GROQ_API_KEY=gsk_...

The chatbot:
  • Only answers skin/hydration-related questions
  • Is polite, professional, and empathetic
  • Refuses off-topic questions gracefully
"""

import os
import requests
import json

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")   # set in .env or env var
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL        = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are SkinBot, a friendly and professional AI skin-wellness assistant.
Your ONLY area of expertise is skin health, specifically facial skin hydration,
dehydration, and general skin care recommendations.

Rules you MUST follow:
1. Only answer questions related to skin health, hydration, dehydration,
   moisturising, skin-care routines, ingredients, and related lifestyle tips.
2. If the user asks about ANYTHING else (e.g. cooking, coding, politics, etc.),
   politely decline and redirect:
   "I'm only able to help with skin and hydration topics. Could we focus on your
   skin wellness?"
3. Never diagnose medical conditions — always recommend consulting a dermatologist
   for persistent issues.
4. Be warm, supportive, empathetic, and non-judgmental.
5. Keep responses concise (3–6 sentences) unless the user asks for more detail.
6. When giving product recommendations, prefer ingredient-based advice
   (e.g. hyaluronic acid, ceramides) rather than specific brand names.
7. Never be rude, dismissive, or condescending.
"""

# ── Conversation history management ───────────────────────────────────────────

def build_initial_message(is_dehydrated: bool) -> dict:
    """Returns the assistant's opening message after the model result is shown."""
    if is_dehydrated:
        content = (
            "🔴 Your skin analysis shows signs of **dehydration**. "
            "I'm here to help! Would you like personalised recommendations "
            "to restore your skin's hydration? Just say **Yes** to get started."
        )
    else:
        content = (
            "🟢 Great news — your skin looks **well-hydrated**! "
            "Would you like some tips to maintain this healthy glow, "
            "or do you have any skin-related questions?"
        )
    return {"role": "assistant", "content": content}


def chat(history: list[dict], user_message: str) -> tuple[str, list[dict]]:
    """
    Sends the conversation to Groq and returns the assistant reply + updated history.

    Parameters
    ----------
    history      : list of {"role": "user"|"assistant", "content": str}
    user_message : latest user input

    Returns
    -------
    reply        : str
    new_history  : updated list
    """
    if not GROQ_API_KEY:
        return (
            "⚠️ No Groq API key found. Please set the GROQ_API_KEY environment "
            "variable. Get a free key at https://console.groq.com/",
            history,
        )

    updated = history + [{"role": "user", "content": user_message}]

    payload = {
        "model":    MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + updated,
        "max_tokens": 512,
        "temperature": 0.7,
    }

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data  = resp.json()
        reply = data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as e:
        reply = f"⚠️ API error {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        reply = f"⚠️ Connection error: {e}"

    new_history = updated + [{"role": "assistant", "content": reply}]
    return reply, new_history