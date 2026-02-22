import os
import re
import time
import requests
from datetime import datetime

import pandas as pd
import streamlit as st
import altair as alt
from search import load_data
from vector_search import index_movies, search_movies_vector, STOPWORDS
from gemini_utils import expand_query, explain_match, summarize_results, identify_movie, describe_video_url, describe_video_clip
from logging_utils import log_query, read_logs, DEFAULT_LOG_PATH

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="CineRecall AI", layout="wide")

# -----------------------------
# Global cinematic theme
# -----------------------------
st.markdown("""
<style>
/* Dark background */
.stApp {
    background: radial-gradient(ellipse at top, #1a0f00 0%, #0a0a0a 60%);
    color: #e8d5a3;
}

/* Main content area */
.block-container {
    padding-top: 2rem;
}

/* Headings */
h1, h2, h3 {
    color: #f5c842 !important;
    font-family: Georgia, serif !important;
    letter-spacing: 0.05em;
}

/* Captions and small text */
.stCaption, small {
    color: #7a6a4a !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #3a2a10;
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #c8a96e;
    border: 1px solid transparent;
    border-radius: 4px 4px 0 0;
    padding: 0.5rem 1.5rem;
    letter-spacing: 0.05em;
}
.stTabs [aria-selected="true"] {
    background: #1a0f00 !important;
    color: #f5c842 !important;
    border-color: #3a2a10 !important;
    border-bottom-color: transparent !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: #f5c842;
    color: #0a0a0a;
    border: none;
    font-weight: 700;
    letter-spacing: 0.08em;
    border-radius: 3px;
}
.stButton > button[kind="primary"]:hover {
    background: #ffd966;
    color: #0a0a0a;
}

/* Secondary buttons */
.stButton > button {
    background: transparent;
    color: #c8a96e;
    border: 1px solid #3a2a10;
    border-radius: 3px;
}
.stButton > button:hover {
    border-color: #f5c842;
    color: #f5c842;
}

/* Text area + inputs */
.stTextArea textarea {
    background: #120c00;
    color: #e8d5a3;
    border: 1px solid #3a2a10;
    border-radius: 4px;
}
.stTextArea textarea:focus {
    border-color: #f5c842 !important;
    box-shadow: 0 0 0 1px #f5c84266;
}

/* Expanders */
.streamlit-expanderHeader {
    background: #120c00 !important;
    color: #c8a96e !important;
    border: 1px solid #3a2a10 !important;
    border-radius: 4px;
}
.streamlit-expanderContent {
    background: #0d0800 !important;
    border: 1px solid #3a2a10 !important;
    border-top: none !important;
    color: #e8d5a3;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #3a2a10;
}

/* Info box */
.stAlert {
    background: #1a0f00 !important;
    border: 1px solid #f5c84244 !important;
    color: #e8d5a3 !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #f5c842 !important;
    font-family: Georgia, serif;
}

/* Dividers */
hr {
    border-color: #3a2a10;
}

/* Progress bar */
.stProgress > div > div {
    background: #f5c842 !important;
}

/* Sidebar (if ever used) */
.stSidebar {
    background: #0d0800;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Intro screen (first load only)
# -----------------------------
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

if not st.session_state.intro_done:
    intro = st.empty()
    intro.markdown("""
    <style>
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInSlow {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes subtitleReveal {
        from { opacity: 0; letter-spacing: 0.6em; }
        to   { opacity: 1; letter-spacing: 0.35em; }
    }
    .cin-wrap {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: radial-gradient(ellipse at center, #1a0f00 0%, #000000 75%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }
    .cin-rule {
        width: 0;
        height: 1px;
        background: #f5c842;
        animation: expandRule 1.2s ease 0.3s forwards;
        margin-bottom: 2rem;
    }
    @keyframes expandRule {
        from { width: 0; opacity: 0; }
        to   { width: 320px; opacity: 1; }
    }
    .cin-title {
        font-size: 3.8rem;
        font-weight: 900;
        color: #f5c842;
        letter-spacing: 0.12em;
        text-shadow: 0 0 40px #f5c84288, 0 0 80px #f5c84233;
        animation: fadeInUp 1.2s ease 0.8s both;
        font-family: Georgia, serif;
        margin: 0;
    }
    .cin-sub {
        font-size: 0.95rem;
        color: #c8a96e;
        letter-spacing: 0.35em;
        text-transform: uppercase;
        animation: subtitleReveal 1.5s ease 1.8s both;
        margin-top: 1rem;
    }
    .cin-rule2 {
        width: 0;
        height: 1px;
        background: #f5c842;
        animation: expandRule 1.2s ease 1.5s forwards;
        margin-top: 2rem;
    }
    .cin-tagline {
        font-size: 0.75rem;
        color: #7a6a4a;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        animation: fadeInSlow 2s ease 3s both;
        margin-top: 2.5rem;
    }
    </style>
    <div class="cin-wrap">
        <div class="cin-rule"></div>
        <h1 class="cin-title">CineRecall AI</h1>
        <div class="cin-sub">Semantic Movie Search</div>
        <div class="cin-rule2"></div>
        <div class="cin-tagline">Remember the movie. Find the memory.</div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(8)
    intro.empty()
    st.session_state.intro_done = True
    st.rerun()

st.title("🎬 CineRecall AI")
st.caption("Find movies from vague memories using semantic search + analytics")

LOG_DIR = str(DEFAULT_LOG_PATH.parent)
LOG_PATH = str(DEFAULT_LOG_PATH)


# -----------------------------
# Helpers
# -----------------------------
_TMDB_KEY = os.getenv("TMDB_API_KEY", "")

@st.cache_data(show_spinner=False)
def get_poster_url(title: str) -> str:
    """Fetch movie poster URL from TMDB. Returns empty string if not found."""
    if not _TMDB_KEY:
        return ""
    try:
        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": _TMDB_KEY, "query": title},
            timeout=5,
        )
        data = resp.json()
        results = data.get("results", [])
        if results and results[0].get("poster_path"):
            return f"https://image.tmdb.org/t/p/w300{results[0]['poster_path']}"
    except Exception:
        pass
    return ""


def clean_tokens(text: str) -> list[str]:
    from vector_search import STOPWORDS
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = re.findall(r"[a-z]{3,}", text)
    return [t for t in tokens if t not in STOPWORDS]


def overlap_keywords(query: str, doc: str, max_items: int = 6) -> list[str]:
    """
    Simple overlap keywords between query and the matched movie text.
    """
    q = set(clean_tokens(query))
    d = set(clean_tokens(doc))
    overlap = sorted(list(q.intersection(d)))
    return overlap[:max_items]


# ----- Backend Integration Point -----
@st.cache_resource
def load_engine():
    df = load_data("data/TMDB_movies.csv")
    index_movies(df, force_reindex=False)  # set True once to wipe & re-index
    return df

_df = load_engine()


# ----- Backend Integration Point -----
# (load_engine already defined above)

tabs = st.tabs(["🔍 Search", "📊 Analytics", "ℹ️ How it works"])

# =============================
# TAB 1: SEARCH
# =============================
with tabs[0]:
    st.subheader("Search")

    search_mode = st.radio(
        "How do you want to search?",
        ["✏️ Describe it", "▶️ YouTube link"],
        horizontal=True,
        label_visibility="collapsed",
    )

    user_query = ""
    video_description = ""
    do_search = False

    video_gemini_title = ""

    if search_mode == "✏️ Describe it":
        user_query = st.text_area(
            "Describe what you remember (plot, scene, quote, actors, vibe):",
            placeholder="Example: a soldier keeps reliving the same battle and resets after dying...",
            height=110,
        )
        do_search = st.button("🔎 Search", type="primary", use_container_width=True)

    elif search_mode == "▶️ YouTube link":
        yt_url = st.text_input(
            "Paste a YouTube link to a scene or clip:",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        do_search = st.button("🔎 Search", type="primary", use_container_width=True)
        if do_search and yt_url.strip():
            loading = st.empty()
            loading.markdown("""
            <div style="text-align:center; padding: 3rem 1rem;">
                <h2 style="color:#f5c842;">🎬 Analysing your clip...</h2>
                <p style="color:#c8a96e;">Gemini is watching the video and describing the scene.<br>This may take up to 30 seconds — please wait.</p>
            </div>
            """, unsafe_allow_html=True)
            video_description, video_gemini_title = describe_video_url(yt_url.strip())
            loading.empty()
            if video_description.startswith("(Gemini unavailable"):
                st.error("Could not process the video. Check the link and try again.")
                do_search = False
            else:
                user_query = video_description

    if do_search and user_query and user_query.strip():
        with st.spinner("Searching the archives..."):
            expanded = expand_query(user_query.strip())
            gemini_ok = not expanded.startswith("(Gemini unavailable")
            search_input = expanded if gemini_ok else user_query.strip()

            results = search_movies_vector(search_input, top_k=3)

            # Use video-identified title first, then fall back to text-based identify_movie
            gemini_title = video_gemini_title or (identify_movie(user_query.strip()) if gemini_ok else "")
            if gemini_title:
                already_in_results = any(
                    r["title"].strip().lower() == gemini_title.lower() for r in results
                )
                if not already_in_results:
                    match = _df[_df["title"].str.strip().str.lower() == gemini_title.lower()]
                    if not match.empty:
                        row = match.iloc[0]
                        results.insert(0, {
                            "title": row["title"],
                            "overview": str(row.get("overview", "")),
                            "confidence": 99.0,
                            "gemini_pick": True,
                        })

        # Log only if we got something
        if results:
            results_df = pd.DataFrame(results)
            log_query(user_query.strip(), results_df)

        st.markdown("---")
        st.subheader("Top Matches")

        if results and gemini_ok:
            summary = summarize_results(user_query.strip(), results)
            if not summary.startswith("(Gemini unavailable"):
                st.info(summary)

        for i, r in enumerate(results):
                conf = float(r["confidence"])
                title = r["title"]
                overview = r["overview"]

                if i == 0:
                    poster_url = get_poster_url(title)
                    if poster_url:
                        pcol, tcol = st.columns([1, 3])
                        with pcol:
                            st.image(poster_url, width=160)
                        with tcol:
                            st.markdown(f"### 🎬 {title}")
                            st.write(f"**Confidence:** {conf:.1f}%")
                            st.progress(conf / 100)
                            with st.expander("Why this matched"):
                                if gemini_ok:
                                    explanation = explain_match(user_query.strip(), title, overview)
                                    if not explanation.startswith("(Gemini unavailable"):
                                        st.write(explanation)
                                else:
                                    keys = overlap_keywords(user_query, f"{title} {overview}", max_items=6)
                                    if keys:
                                        st.write("Shared keywords/themes: " + ", ".join(keys))
                                    else:
                                        st.write("Matched based on overall semantic similarity.")
                            with st.expander("Plot snippet"):
                                st.write(overview)
                    else:
                        st.markdown(f"### 🎬 {title}")
                        st.write(f"**Confidence:** {conf:.1f}%")
                        st.progress(conf / 100)
                        with st.expander("Why this matched"):
                            if gemini_ok:
                                explanation = explain_match(user_query.strip(), title, overview)
                                if not explanation.startswith("(Gemini unavailable"):
                                    st.write(explanation)
                            else:
                                keys = overlap_keywords(user_query, f"{title} {overview}", max_items=6)
                                if keys:
                                    st.write("Shared keywords/themes: " + ", ".join(keys))
                                else:
                                    st.write("Matched based on overall semantic similarity.")
                        with st.expander("Plot snippet"):
                            st.write(overview)

                    st.divider()
                    if len(results) > 1:
                        st.markdown("#### 🎞️ You might also like")

                else:
                    poster_url = get_poster_url(title)
                    if poster_url:
                        pcol, tcol = st.columns([1, 4])
                        with pcol:
                            st.image(poster_url, width=100)
                        with tcol:
                            st.markdown(f"**🎬 {title}**")
                            with st.expander("Plot snippet"):
                                st.write(overview)
                    else:
                        st.markdown(f"**🎬 {title}**")
                        with st.expander("Plot snippet"):
                            st.write(overview)
                    st.divider()

    elif do_search:
        st.warning("Type a description first (even a short one).")


# =============================
# TAB 2: ANALYTICS
# =============================
with tabs[1]:
    st.subheader("Analytics Dashboard")
    logs_list = read_logs()

    if not logs_list:
        st.info("No searches logged yet. Run a few searches on the Search tab, then come back here.")
    else:
        df = pd.DataFrame(logs_list)
        df["top_confidence"] = pd.to_numeric(df["top_confidence"], errors="coerce").fillna(0)
        
        # KPI Metrics
        st.metric("📊 Total Searches", len(df))
        
        st.markdown("---")
        
        st.subheader("Recent Searches")
        filtered_df = df
        
        # Display table
        display_cols = ["ts", "query", "top_title", "top_confidence"]
        st.dataframe(
            filtered_df[display_cols].rename(columns={
                "ts": "Time",
                "query": "Search Query",
                "top_title": "Top Result",
                "top_confidence": "Confidence (%)",
            }),
            use_container_width=True,
        )
        
        st.markdown("---")
        
        # Search Timeline
        st.markdown("#### Search Timeline")
        df["ts_hour"] = pd.to_datetime(df["ts"]).dt.floor("h")
        timeline_data = df.groupby("ts_hour").size().reset_index(name="Searches")
        
        if len(timeline_data) > 0:
            timeline_chart = (
                alt.Chart(timeline_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X("ts_hour:T", title="Time"),
                    y=alt.Y("Searches:Q", title="Number of Searches"),
                    tooltip=["ts_hour:T", "Searches:Q"],
                )
                .properties(height=300)
            )
            st.altair_chart(timeline_chart, use_container_width=True)
        else:
            st.write("No timeline data yet.")
        
        st.markdown("---")
        
        # Top Keywords
        st.markdown("#### Top Keywords in Queries")
        all_tokens = []
        for q in df["query"].dropna().astype(str).tolist():
            all_tokens.extend(clean_tokens(q))
        
        if all_tokens:
            kw_series = pd.Series(all_tokens).value_counts().head(15).reset_index()
            kw_series.columns = ["Keyword", "Frequency"]
            
            kw_chart = (
                alt.Chart(kw_series)
                .mark_bar()
                .encode(
                    x=alt.X("Frequency:Q", title="Frequency"),
                    y=alt.Y("Keyword:N", sort="-x", title="Keyword"),
                    tooltip=["Keyword", "Frequency"],
                )
                .properties(height=max(250, len(kw_series) * 20))
            )
            st.altair_chart(kw_chart, use_container_width=True)
        else:
            st.write("Not enough query data yet.")
        
        st.markdown("---")
        
        # Download and Reset
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df[display_cols].to_csv(index=False)
            st.download_button(
                label="⬇️ Download Search Log as CSV",
                data=csv_data,
                file_name="search_log.csv",
                mime="text/csv",
            )
        
        with col2:
            if st.button("🗑️ Reset Search Log"):
                DEFAULT_LOG_PATH.unlink(missing_ok=True)
                st.success("Search log cleared!")
                st.rerun()


# =============================
# TAB 3: HOW IT WORKS
# =============================
with tabs[2]:
    st.subheader("How It Works")
    st.write(
        "CineRecall AI helps you find movies from vague memories — no title needed. "
        "Just describe what you remember: a scene, a feeling, a plot detail, an actor. "
        "The app figures out what you're thinking of."
    )

    st.markdown("### 🧠 Semantic Search")
    st.write(
        "Unlike a normal keyword search, CineRecall AI understands the *meaning* of your description. "
        "It converts your words into a mathematical representation and compares it against thousands of movie plots, "
        "finding the ones that are conceptually closest — even if you don't use the exact right words."
    )

    st.markdown("### ✨ AI Query Expansion")
    st.write(
        "Before searching, your description is passed to Google Gemini, which rewrites it into a richer, "
        "more detailed version — adding likely genres, themes, and context. "
        "This dramatically improves match quality for vague or partial memories."
    )

    st.markdown("### 🎬 Why This Matched")
    st.write(
        "For each result, Gemini explains in plain language *why* that movie fits your description — "
        "connecting specific details from your query to the film's plot and themes."
    )

    st.markdown("### 📊 Analytics")
    st.write(
        "Every search is logged automatically. The Analytics tab shows you search history, "
        "confidence trends, and the most common themes people search for."
    )