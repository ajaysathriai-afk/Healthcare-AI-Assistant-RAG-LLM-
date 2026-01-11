# app.py
# Streamlit Healthcare AI Assistant with ChatGPT-style interface + streaming
# Usage: streamlit run app.py

import streamlit as st
import requests
from typing import List, Any, Tuple
from datetime import datetime
import re
import os
import shutil
import json

# ---------------- CONFIG ----------------
API_URL = "https://healthcare-ai-assistant-rag-llm.onrender.com/ask_llm"
STREAM_API_URL = "https://healthcare-ai-assistant-rag-llm.onrender.com/ask_llm_stream"


CHAT_WIDTH = 780
DEFAULT_TOP_K = 3
ASSETS_BG_PATH = "assets/bg.jpg"
UPLOADED_PATH = "/mnt/data/Screenshot 2025-11-14 at 4.25.41 PM.png"

# Copy uploaded bg into assets if present (kept for future use)
os.makedirs("assets", exist_ok=True)
try:
    if os.path.exists(UPLOADED_PATH) and not os.path.exists(ASSETS_BG_PATH):
        shutil.copy(UPLOADED_PATH, ASSETS_BG_PATH)
except Exception:
    pass

# ---------------- SESSION-STATE INIT ----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []          # store only latest exchange
if "top_k_input" not in st.session_state:
    st.session_state["top_k_input"] = DEFAULT_TOP_K
if "show_sources_widget" not in st.session_state:
    st.session_state["show_sources_widget"] = True
if "show_welcome" not in st.session_state:
    st.session_state["show_welcome"] = True

# --------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- HELPER FUNCTIONS ----------------
def split_answer_and_embedded_sources(answer: str) -> Tuple[str, str]:
    """
    Strip any trailing 'Sources' / 'Sources used' block from the LLM answer.
    Returns (clean_answer, embedded_sources_text).
    """
    if not answer:
        return "", ""
    ans = answer.strip()

    # Look for a line starting with Sources / Source (case-insensitive)
    m = re.search(r"(?im)^\s*(sources?\b.*)$", ans, flags=re.MULTILINE)
    if m:
        idx = m.start(1)
        return ans[:idx].strip(), ans[idx:].strip()

    # Fallback: 'Sources used:'
    m2 = re.search(r"(?i)(sources?\s+used\s*:)", ans)
    if m2:
        idx = m2.start()
        return ans[:idx].strip(), ans[idx:].strip()

    return ans, ""


def normalize_sources(sources: Any) -> List[str]:
    if not sources:
        return []
    out: List[str] = []
    if isinstance(sources, list):
        for s in sources:
            if isinstance(s, dict):
                out.append(str(s.get("source") or s.get("id") or ""))
            else:
                out.append(str(s))
    else:
        out.append(str(sources))
    deduped: List[str] = []
    seen = set()
    for s in out:
        if s and s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped


DISCLAIMER_PREFIX = "⚠️ Important:"


def strip_leading_disclaimer(text: str) -> str:
    """
    If LLM answer itself starts with our disclaimer (⚠️ Important: ...),
    remove that block so we only show the disclaimer once (below the answer).
    """
    if not text:
        return text
    stripped = text.lstrip()
    if not stripped.startswith("⚠️"):
        return text

    # remove first paragraph (up to first blank line) or first line
    parts = stripped.split("\n\n", 1)
    if len(parts) == 2:
        return parts[1].strip()
    lines = stripped.splitlines()
    if len(lines) > 1:
        return "\n".join(lines[1:]).strip()
    # if the whole answer is just the disclaimer, return empty string
    return ""


def render_final_answer(
    display_text: str,
    structured_sources: List[str],
    answer_placeholder,
    sources_placeholder,
):
    """
    Render the final answer bubble + single disclaimer below.
    Sources appear only when toggle is ON.
    """
    # doctor emoji + answer
    answer_html = f"""
    <div class="message-animate" style="color: #ececf1; line-height: 1.6; font-size: 15px; padding: 0.75rem 0;">
        👨‍⚕️ {display_text}
    </div>
    <br>
    <div style="color: #fbbc04; font-size: 13px; line-height: 1.45; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px;">
        ⚠️ <strong>Important:</strong> This assistant provides general health information only and is <u>not a substitute</u> for professional medical advice, diagnosis, or treatment.<br>
        If you have serious or worsening symptoms, chest pain, trouble breathing, thoughts of self-harm, or any emergency, please consult a doctor or contact local emergency services immediately.
    </div>
    """
    answer_placeholder.markdown(answer_html, unsafe_allow_html=True)

    # sources under answer
    if st.session_state.get("show_sources_widget") and structured_sources:
        tags_html = "".join(
            [f'<span class="source-tag">📄 {s}</span>' for s in structured_sources]
        )
        sources_placeholder.markdown(
            f"""
            <div class="sources-container message-animate">
                <div class="sources-title">Sources:</div>
                <div>{tags_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        sources_placeholder.empty()


def handle_send_message(question: str, answer_placeholder, sources_placeholder):
    """
    Streaming send:
    - Clears previous messages (we only keep the latest in state)
    - Calls /ask_llm_stream (JSONL streaming)
    - Streams tokens into one assistant bubble (without disclaimer)
    - At end, strips any in-answer disclaimer & 'Sources' section
    - Renders answer + single disclaimer below
    - Shows sources only as tags (if toggle ON)
    """
    if not question or not question.strip():
        return

    top_k_val = int(st.session_state.get("top_k_input", DEFAULT_TOP_K) or DEFAULT_TOP_K)

    # Clear previous messages — keep only this exchange for state
    st.session_state["messages"] = []

    # Store user question internally (not rendered)
    st.session_state["messages"].append(
        {
            "role": "user",
            "text": question.strip(),
            "time": datetime.utcnow().isoformat(),
        }
    )

    payload = {"question": question.strip(), "top_k": top_k_val}

    answer_text = ""
    structured_sources: List[str] = []
    had_stream_error = False

    try:
        with st.spinner("Thinking..."):
            resp = requests.post(
                STREAM_API_URL, json=payload, stream=True, timeout=120
            )

            if resp.status_code != 200:
                # Non-streaming error, show once
                try:
                    err_json = resp.json()
                    err_detail = err_json.get("detail") or str(err_json)
                except Exception:
                    err_detail = resp.text
                error_msg = (
                    f"❌ Error: Server returned status {resp.status_code}. {err_detail}"
                )
                had_stream_error = True
                answer_text = error_msg
                # show error without disclaimer
                answer_placeholder.markdown(
                    f"""
                    <div class="message-animate" style="color: #ececf1; line-height: 1.6; font-size: 15px; padding: 0.5rem 0;">
                        {error_msg}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Stream JSON Lines
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    msg_type = obj.get("type")
                    if msg_type == "sources":
                        structured_sources = normalize_sources(obj.get("sources", []))
                    elif msg_type == "token":
                        token = obj.get("token", "")
                        if not token:
                            continue
                        answer_text += token
                        # live streaming of just the answer text (no disclaimer yet)
                        partial = strip_leading_disclaimer(answer_text)
                        if not partial:
                            partial = ""  # avoid showing just disclaimer text
                        answer_placeholder.markdown(
                            f"""
                            <div class="message-animate" style="color: #ececf1; line-height: 1.6; font-size: 15px; padding: 0.75rem 0;">
                                👨‍⚕️ {partial}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif msg_type == "error":
                        err = obj.get("error", "Unknown streaming error")
                        had_stream_error = True
                        answer_text += f"\n\n[Streaming error: {err}]"
                        break
                    elif msg_type == "done":
                        break

        # Clean answer text: strip embedded Sources section and any in-answer disclaimer
        if not had_stream_error and answer_text:
            # first cut off "Sources:" block
            clean_answer, embedded_sources = split_answer_and_embedded_sources(
                answer_text
            )
            # then remove leading disclaimer if the model wrote it
            clean_answer = strip_leading_disclaimer(clean_answer)
            display_text = clean_answer or answer_text
            if not structured_sources and embedded_sources:
                structured_sources = normalize_sources([embedded_sources])
        else:
            display_text = answer_text

        # Final render after cleaning (single disclaimer below)
        
        # DEBUG: Show what sources we have
        print(f"DEBUG: About to render. Sources = {structured_sources}")
        st.sidebar.write(f"🔍 DEBUG: Sources received = {structured_sources}")
        
        render_final_answer(display_text, structured_sources, answer_placeholder, sources_placeholder)

        # Persist final message just for history (used for re-render on toggle)
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "text": display_text.strip(),
                "sources": structured_sources,
                "time": datetime.utcnow().isoformat(),
            }
        )

    except requests.exceptions.RequestException as e:
        error_msg = f"❌ Connection Error: {str(e)}"
        answer_placeholder.markdown(
            f"""
            <div class="message-animate" style="color: #ececf1; line-height: 1.6; font-size: 15px; padding: 0.5rem 0;">
                {error_msg}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "text": error_msg,
                "sources": [],
                "time": datetime.utcnow().isoformat(),
            }
        )

# ---------------- CSS / STYLING ----------------
st.markdown(
    """
<style>
    /* Base styling */
    .stApp {
        background-color: #0f1117;
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 820px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1d24;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ececf1;
    }
    
    [data-testid="stSidebar"] p {
        color: #ececf1 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #ececf1 !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton button {
        background-color: #2a2d35 !important;
        color: #ececf1 !important;
        border: 1px solid #3e3f4b !important;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #343741 !important;
        border-color: #10a37f !important;
    }
    
    /* Text input and textarea */
    .stTextArea {
        position: relative;
    }
    
    .stTextArea textarea {
        background-color: #ffffff !important;
        border: 2px solid #d1d5db !important;
        border-radius: 12px !important;
        color: #000000 !important;
        font-size: 16px !important;
        font-weight: 400 !important;
        padding: 12px 90px 12px 16px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #10a37f !important;
        box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.2) !important;
        background-color: #ffffff !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #6b7280 !important;
        opacity: 1 !important;
    }
    
    /* Number input */
    .stNumberInput input {
        background-color: #0f1117 !important;
        border: 1px solid #3e3f4b !important;
        border-radius: 8px !important;
        color: #ececf1 !important;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
        font-size: 15px !important;
    }
    
    .stButton button[kind="primary"] {
        background-color: #10a37f !important;
        color: white !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        min-height: 36px !important;
        font-size: 14px !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #0d8f6f !important;
        transform: translateY(-1px);
    }
    
    .stButton button[kind="primary"]:disabled {
        background-color: #565b63 !important;
        opacity: 0.5 !important;
        cursor: not-allowed !important;
    }
    
    /* Checkbox */
    .stCheckbox {
        color: #ececf1 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1d24 !important;
        border-radius: 8px !important;
        color: #ececf1 !important;
    }
    
    /* Chat avatar */
    .chat-avatar {
        width: 36px;
        height: 36px;
        border-radius: 6px;
        background-color: #10a37f;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        color: white;
        flex-shrink: 0;
        font-size: 20px;
    }
    
    .sources-container {
        margin-top: 1rem;
        padding: 0.75rem;
        background-color: rgba(255,255,255,0.03);
        border-radius: 8px;
        border: 1px solid #3e3f4b;
    }
    
    .sources-title {
        font-size: 13px;
        font-weight: 600;
        color: #8e8ea0;
        margin-bottom: 0.5rem;
    }
    
    .source-tag {
        display: inline-block;
        background-color: rgba(255,255,255,0.05);
        padding: 0.25rem 0.625rem;
        border-radius: 6px;
        font-size: 13px;
        color: #8e8ea0;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Welcome screen */
    .welcome-container {
        text-align: center;
        padding: 3rem 1rem;
        color: #ececf1;
    }
    
    .welcome-icon {
        font-size: 64px;
        margin-bottom: 1rem;
    }
    
    .welcome-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .welcome-subtitle {
        font-size: 16px;
        color: #8e8ea0;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #10a37f !important;
    }
    
    .stSpinner > div > div {
        color: #ececf1 !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stStatusWidget"] {
        background-color: rgba(16, 163, 127, 0.15) !important;
        border: 1px solid rgba(16, 163, 127, 0.4) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
    }
    
    [data-testid="stStatusWidget"] div,
    [data-testid="stStatusWidget"] p {
        color: #ffffff !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    
    /* Fade-in animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .message-animate {
        animation: fadeInUp 0.5s ease-out;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    st.markdown("**Top K Results**")
    st.number_input(
        "Number of retrieval results",
        min_value=1,
        max_value=10,
        key="top_k_input",
        help="Number of document chunks to retrieve",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # BUTTON APPROACH - More reliable than checkbox
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Show sources in responses**")
    with col2:
        if st.button("🔄"):
            st.session_state["show_sources_widget"] = not st.session_state.get("show_sources_widget", True)
            st.rerun()
    
    # Show current state
    if st.session_state.get("show_sources_widget", True):
        st.success("✅ Sources: ON")
    else:
        st.info("❌ Sources: OFF")

    st.markdown("---")

    if st.button("🗑️ Clear All Messages", key="clear_sidebar", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["show_welcome"] = True
        st.rerun()

    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "This is a prototype healthcare AI assistant. Always consult with healthcare professionals for medical advice."
    )

# ---------------- HEADER ----------------
st.markdown(
    '<h1 style="color: #ececf1; font-size: 32px; font-weight: 700; margin-bottom: 0.5rem;">🏥 Healthcare AI Assistant</h1>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------- WELCOME (when no messages yet) ----------------
welcome_container = st.empty()
if st.session_state.get("show_welcome", True):
    with welcome_container.container():
        st.markdown(
            """
        <div class="welcome-container">
            <div class="welcome-icon">🏥</div>
            <div class="welcome-title">Healthcare AI Assistant</div>
            <div class="welcome-subtitle">
                Ask me general health questions. I can help with symptoms, conditions, and wellness information.<br>
                <strong>Note:</strong> This is a prototype - not medical advice.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.info("💊 What are the symptoms of diabetes?")
            st.info("🥗 What foods are good for heart health?")
        with col2:
            st.info("🏃 How can I improve my cardiovascular health?")
            st.info("😴 Tips for better sleep quality?")

# ---------------- PLACEHOLDERS FOR ANSWER + SOURCES ----------------
answer_placeholder = st.empty()
sources_placeholder = st.empty()

# ---------------- INPUT AREA ----------------
st.markdown("---")

input_container = st.container()
with input_container:
    with st.form(key="chat_form", clear_on_submit=True):
        st.markdown('<div style="position: relative;">', unsafe_allow_html=True)

        user_input = st.text_area(
            "Message",
            height=120,
            placeholder="Type your health question here...",
            label_visibility="collapsed",
            key="user_input",
        )

        submitted = st.form_submit_button("Send", type="primary")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- LOGIC: STREAM OR RE-RENDER ----------------
if submitted and user_input and user_input.strip():
    # Hide welcome screen permanently
    st.session_state["show_welcome"] = False
    welcome_container.empty()
    # Clear messages to show only new answer
    st.session_state["messages"] = []
    # IMMEDIATELY clear placeholders to prevent duplication
    answer_placeholder.empty()
    sources_placeholder.empty()
    # New question -> streaming + update state
    handle_send_message(user_input, answer_placeholder, sources_placeholder)
else:
    # No new submit -> re-render last assistant answer (for toggles, etc.)
    last_assistant = None
    for m in reversed(st.session_state["messages"]):
        if m.get("role") == "assistant":
            last_assistant = m
            break

    if last_assistant:
        text = last_assistant.get("text", "")
        sources = normalize_sources(last_assistant.get("sources", []))
        render_final_answer(text, sources, answer_placeholder, sources_placeholder)

