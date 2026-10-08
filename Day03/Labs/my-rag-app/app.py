import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from pypdf import PdfReader

# The repo root is added first so the shared AskIT modules can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from askit_core import bedrock, config  # noqa: E402

APP_NAME = "PDF Buddy"
AVATAR = "🤖"
TAGLINE = "Upload. Ask. Done."
PERSONALITY = "calm senior engineer"
STYLE = "short, friendly, and lightly humorous"
THEME = "teal on dark"
AUTHOR = "Priya, Team 2"

st.set_page_config(page_title=APP_NAME, page_icon=AVATAR, layout="wide")

# Add a small, readable style block for the themed interface.
st.markdown(
    """
    <style>
    .stApp { background: #071b22; color: #e8fbff; }
    [data-testid="stHeader"] { background: #0b7f87; }
    [data-testid="stChatInput"] { border-radius: 14px; border: 2px solid #18b7b5; }
    [data-testid="stChatMessage"] { border-radius: 14px; margin: 8px 0; }
    h1 { color: #63e6e2; }
    </style>
    """,
    unsafe_allow_html=True,
)


def chunk_words(text, size=120, overlap=30):
    """Split text into overlapping word chunks; overlap keeps nearby facts together."""
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i : i + size]) for i in range(0, len(words), step)]


def embed_text(client, text):
    """Embed one chunk with Titan Text Embeddings v2 and return a 512-value vector."""
    body = json.dumps({"inputText": text, "dimensions": 512, "normalize": True})
    response = client.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=body)
    return np.asarray(json.loads(response["body"].read())["embedding"], dtype=float)


def cosine_similarity(a, b):
    """Return cosine similarity; zero-length vectors are treated as a poor match."""
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def build_index(pdf_file, chunk_size, overlap):
    """Read the PDF, chunk its text, and store page-aware vectors in session state."""
    reader = PdfReader(pdf_file)
    pages = []
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((number, text))
    if not pages:
        return "No text found in this PDF. It may be scanned; use a text-based PDF."

    chunks = []
    for page_number, page_text in pages:
        for text in chunk_words(page_text, chunk_size, overlap):
            chunks.append({"page": page_number, "text": text})

    client = bedrock.client()
    progress = st.progress(0.0)
    for index in range(0, len(chunks), 4):
        batch = chunks[index : index + 4]
        for chunk in batch:
            chunk["vector"] = embed_text(client, chunk["text"])
        progress.progress(min(1.0, (index + len(batch)) / len(chunks)))
    progress.empty()
    st.session_state.index = chunks
    st.session_state.pages = len(pages)
    st.session_state.chunks = len(chunks)
    return None


def find_matches(question, top_k):
    """Embed the question and return the most similar page-aware chunks."""
    query = embed_text(bedrock.client(), question)
    results = []
    for chunk in st.session_state.index:
        score = cosine_similarity(chunk["vector"], query)
        results.append({**chunk, "score": score})
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def answer_question(question, top_k):
    """Retrieve context, call Nova, and format the answer with source pages."""
    matches = find_matches(question, top_k)
    context = "\n\n".join(f"[p.{m['page']}] {m['text']}" for m in matches)
    prompt = (
        f"You are {PERSONALITY}, and you speak with {STYLE}. "
        "Answer ONLY from the context. If the answer is not in the context, say you "
        "could not find it in the PDF. Cite the page like [p.3].\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = bedrock.client().converse(
        modelId=config.SMALL_MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 500, "temperature": 0.2},
    )
    answer = response["output"]["message"]["content"][0]["text"]
    return answer, matches


def clear_index():
    """Reset uploaded PDF data and chat when a new PDF arrives."""
    st.session_state.index = []
    st.session_state.pages = 0
    st.session_state.chunks = 0
    st.session_state.chat = []
    st.session_state.pdf_name = None


st.title(f"{AVATAR} {APP_NAME} — {TAGLINE}")
st.caption(f"Your {PERSONALITY} PDF assistant using {THEME}.")

with st.sidebar:
    st.header("About me")
    st.markdown(f"**{AUTHOR}**\n\nToday: {datetime.now().strftime('%Y-%m-%d')}\n\nOne fun line: I love finding the tiny clue hidden in a PDF.")
    top_k = st.slider("Top-K", 1, 6, 3)
    chunk_size = st.slider("Chunk size", 50, 300, 120)
    overlap = st.slider("Overlap", 0, 80, 30)
    if st.button("Clear chat"):
        st.session_state.chat = []

pdf_file = st.file_uploader("Upload your PDF", type="pdf", key="pdf_uploader")
if pdf_file and pdf_file.name != st.session_state.get("pdf_name"):
    clear_index()
    st.session_state.pdf_name = pdf_file.name

if pdf_file:
    st.success(f"Uploaded: {pdf_file.name}")
    if st.button("Build index"):
        try:
            error = build_index(pdf_file, chunk_size, overlap)
            if error:
                st.warning(error)
            else:
                st.success(f"Index built: {st.session_state.pages} pages, {st.session_state.chunks} chunks.")
        except Exception as exc:
            st.error(f"AWS failed. Check your .env keys, region, and Titan model access. {exc}")

if st.session_state.get("chunks", 0) == 0:
    st.info("Upload a PDF, then click Build index to create your local RAG index.")

if st.session_state.get("chunks", 0):
    if st.button("Not in my PDF?", type="secondary"):
        question = "Who won the last cricket world cup?"
        st.session_state.chat.append({"role": "user", "content": question})
        st.session_state.chat.append({
            "role": "assistant",
            "content": "I could not find it in the PDF.",
            "sources": [],
            "badge": "weak match",
        })

for message in st.session_state.get("chat", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- Page {source['page']} | Score {source['score']:.2f}\n  {source['text']}")
            st.badge(message["badge"])

question = st.chat_input("Ask about the uploaded PDF...")
if question and st.session_state.get("chunks", 0):
    try:
        answer, matches = answer_question(question, top_k)
        badge = "grounded" if max((m["score"] for m in matches), default=0) >= 0.35 else "weak match"
        st.session_state.chat.append({"role": "user", "content": question})
        st.session_state.chat.append({"role": "assistant", "content": answer, "sources": matches, "badge": badge})
    except Exception as exc:
        st.error(f"AWS failed. Check your .env keys, region, and Titan/Nova model access. {exc}")

if st.session_state.get("chat"):
    markdown = "\n\n".join(
        f"**{m['role']}**: {m['content']}" for m in st.session_state.chat
    )
    st.download_button("Download chat as .md", markdown, file_name="chat.md", type="secondary")

st.sidebar.markdown(f"Built by {AUTHOR} with vibe coding at DevPro Academy")
