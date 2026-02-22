from __future__ import annotations

import os
import time
import tempfile
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.5-flash"


def _ask(prompt: str) -> str:
    """Send a prompt to Gemini and return the text response."""
    try:
        response = _client.models.generate_content(model=MODEL, contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"(Gemini unavailable: {e})"


def _describe_scene(contents: list) -> tuple[str, str]:
    """Ask Gemini to describe the scene AND identify the movie title.
    Returns (description, identified_title). Title is '' if unsure."""
    try:
        response = _client.models.generate_content(
            model=MODEL,
            contents=contents + [(
                "Watch this video clip carefully. Do two things:\n"
                "1. If you can identify the movie, state the exact movie title on the first line as: TITLE: <movie title>\n"
                "   If you are not sure, write: TITLE: UNKNOWN\n"
                "2. Then describe what is happening in the scene: the plot, setting, tone, characters, and themes — "
                "as a detailed description that could help someone find the movie."
            )],
        )
        text = response.text.strip()
        title = ""
        description = text
        if text.startswith("TITLE:"):
            lines = text.split("\n", 1)
            raw_title = lines[0].replace("TITLE:", "").strip()
            if raw_title.upper() != "UNKNOWN":
                title = raw_title
            description = lines[1].strip() if len(lines) > 1 else ""
        return description, title
    except Exception as e:
        return f"(Gemini unavailable: {e})", ""


def describe_video_url(youtube_url: str) -> tuple[str, str]:
    """Describe a YouTube video clip for movie search. Returns (description, title)."""
    try:
        part = types.Part.from_uri(file_uri=youtube_url, mime_type="video/mp4")
        return _describe_scene([part])
    except Exception as e:
        return f"(Gemini unavailable: {e})", ""


def describe_video_clip(video_bytes: bytes, mime_type: str = "video/mp4") -> tuple[str, str]:
    """Upload a video file to Gemini. Returns (description, title)."""
    suffix = "." + mime_type.split("/")[-1]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name

        uploaded = _client.files.upload(path=tmp_path)

        for _ in range(30):
            if uploaded.state.name != "PROCESSING":
                break
            time.sleep(2)
            uploaded = _client.files.get(name=uploaded.name)

        if uploaded.state.name == "FAILED":
            return "(Gemini unavailable: video processing failed)", ""

        return _describe_scene([uploaded])
    except Exception as e:
        return f"(Gemini unavailable: {e})", ""
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def identify_movie(query: str) -> str:
    """Ask Gemini to directly identify the movie title from a description.
    Returns a single movie title string, or empty string if unsure."""
    prompt = (
        "You are a movie identification expert. Based on the description below, "
        "identify the most likely single movie title. "
        "If you are confident, reply with ONLY the movie title — nothing else, no punctuation, no explanation. "
        "If you are not sure, reply with exactly: UNKNOWN\n\n"
        f"Description: {query}"
    )
    result = _ask(prompt)
    if result.startswith("(Gemini unavailable") or result.strip().upper() == "UNKNOWN":
        return ""
    return result.strip()


def expand_query(query: str) -> str:
    """Rewrite the user's vague query into a richer search-friendly description."""
    prompt = (
        "You are a movie search assistant. Rewrite the following vague movie description "
        "into a detailed, keyword-rich description that will help find the right movie. "
        "Include likely genres, themes, plot elements, and mood. "
        "Return only the rewritten description, no explanation.\n\n"
        f"User query: {query}"
    )
    return _ask(prompt)


def explain_match(query: str, title: str, overview: str) -> str:
    """Explain in one sentence why this movie matches the user's query."""
    prompt = (
        f"A user searched for: \"{query}\"\n"
        f"The system returned: \"{title}\" — {overview}\n\n"
        "In one concise sentence, explain why this movie matches the user's search."
    )
    return _ask(prompt)


def summarize_results(query: str, results: list[dict]) -> str:
    """Generate a short conversational summary of the top search results."""
    titles = ", ".join(r["title"] for r in results if r.get("title"))
    prompt = (
        f"A user searched for a movie using this description: \"{query}\"\n"
        f"The top matches found were: {titles}\n\n"
        "Write 2-3 friendly sentences summarizing these results and helping the user "
        "decide which to pick. Be concise."
    )
    return _ask(prompt)


def chat_response(user_message: str, history: list[dict]) -> str:
    """Continue a movie-finding conversation."""
    system = (
        "You are a helpful movie assistant. Help the user find movies based on vague "
        "descriptions, feelings, or partial memories. Ask clarifying questions if needed."
    )
    contents = []
    for h in history:
        role = "user" if h["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=h["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    try:
        response = _client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return response.text.strip()
    except Exception as e:
        return f"(Gemini unavailable: {e})"
