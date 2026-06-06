import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import pytesseract
from PIL import Image
import json
import time

# -------------------------
# FILE STORAGE
# -------------------------
CHAT_FILE = "chats.json"

def load_chats():
    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r") as f:
                return json.load(f)
        except:
            return {"New Chat": []}
    return {"New Chat": []}

def save_chats(chats):
    with open(CHAT_FILE, "w") as f:
        json.dump(chats, f, indent=2)

# -------------------------
# 🔥 SMART AUTO TITLE FUNCTION (AI BASED)
# -------------------------
def generate_title_with_ai(question, mode):
    try:
        if mode == "image":
            prompt = f"""
            Give a SHORT 2-4 word title for this conversation.

            Context: User is asking about an IMAGE.

            Question:
            {question}

            Examples:
            Image Analysis
            Image Explanation
            Diagram Summary

            Title:
            """

        elif mode == "file":
            prompt = f"""
            Give a SHORT 2-4 word title for this conversation.

            Context: User uploaded a DOCUMENT.

            Question:
            {question}

            Examples:
            PDF Summary
            Document Q&A
            File Analysis

            Title:
            """

        else:
            prompt = f"""
            Give a SHORT 2-4 word title for this conversation.

            Question:
            {question}

            Examples:
            Python Help
            AI Explanation
            Coding Question

            Title:
            """

        response = llm.invoke(prompt).content.strip()
        return response.replace("\n", "")[:40]

    except:
        return question[:40]

# -------------------------
# TESSERACT PATH
# -------------------------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ssmore\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# -------------------------
# INIT
# -------------------------
load_dotenv()

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.markdown("""
<script type="module" src="https://unpkg.com/ionicons@7.1.0/dist/ionicons/ionicons.esm.js"></script>
<script nomodule src="https://unpkg.com/ionicons@7.1.0/dist/ionicons/ionicons.js"></script>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* 🔥 Global font (BIG + DARK) */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 18px;              /* ⬆ increased */
    color: #000000 !important;    /* ✅ pure black text */
}

/* 🔥 Title */
h1 {
    font-size: 32px !important;
    font-weight: 700 !important;
}
h1, h2, h3, h4 {
    color: #000000 !important;
}

/* 🔥 Sidebar */
section[data-testid="stSidebar"] {
    background-color: #e2e8f0;  /* darker gray */
}
section[data-testid="stSidebar"] * {
    color: #111827 !important;
    font-size: 16px;
}
/* 🔥 Upload box */
.stFileUploader {
    border: 2px dashed #cbd5f5;
    padding: 12px;
    border-radius: 12px;
}

/* 🔥 Chat bubbles */
[data-testid="stChatMessage"] {
    padding: 16px;
    border-radius: 14px;
    margin-bottom: 12px;
}

/* User bubble */
[data-testid="stChatMessage"]:has(div:contains("You")) {
    background: linear-gradient(135deg, #dbeafe, #eff6ff);
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has(div:contains("Assistant")) {
    background: #f9fafb;
}

/* 🔥 Chat text */
[data-testid="stChatMessage"] p {
    font-size: 18px !important;   /* ⬆ bigger */
    line-height: 1.7;
    color: #000000 !important;    /* ✅ black */
}

/* 🔥 Chat input */
[data-testid="stChatInput"] {
    border-radius: 14px;
}

/* 🔥 Chat history row */
.chat-row {
    display: flex;
    align-items: center;
    background: transparent;   /* ❌ removed white box */
    border-radius: 12px;
    margin-bottom: 6px;
    padding: 4px 6px;          /* tighter spacing */
    border: none;              /* ❌ removed border */
    transition: background 0.2s ease;
}

/* ✅ Hover = slightly darker (NO RED) */
.chat-row:hover {
    background: #cbd5f5;
}

/* Active chat */
.chat-active {
    background: #e0e7ff !important;
    border: 1px solid #c7d2fe;
}

/* 🔥 Buttons reset */
button {
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* 🎯 Chat name buttons (left side) */
div[data-testid="column"]:nth-child(1) button {
    background: transparent !important;
    border: none !important;
    text-align: left !important;
    color: #374151 !important;
}

/* Hover for chat name */
div[data-testid="column"]:nth-child(1) button:hover {
    background: #cbd5f5 !important;
}

/* 🎯 Delete button (RIGHT SIDE — DO NOT CHANGE ALIGNMENT) */
div[data-testid="column"]:nth-child(2) {
    display: flex;
    justify-content: flex-end;
    align-items: center;
}

/* 🔥 FORCE BLACK BIN ICON */
div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] button {
    color: #000000 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    opacity: 1 !important;
    font-size: 18px !important;
    transition: all 0.2s ease;
}

/* 🔴 STRONG HOVER (RED + SCALE) */
div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] button:hover {
    color: #dc2626 !important;   /* RED */
    transform: scale(1.25);
}

/* 🔥 File card */
.file-card {
    padding: 10px;
    border-radius: 12px;
    background: #f1f5f9;
    font-size: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* 🔥 REMOVE ALL BUTTON WRAPPER BOXES (FINAL FIX) */
div[data-testid="stButton"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Remove inner button styling completely */
div[data-testid="stButton"] > button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 4px !important;
}

/* Specifically fix delete icon */
button[kind="secondary"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* 🔥 GLOBAL BUTTON HOVER EFFECT */
div[data-testid="stButton"] > button:hover {
    background: #cbd5f5 !important;  /* soft dark */
    color: #111827 !important;       /* darker text */
}

/* 🔥 TARGET ACTUAL Browse Files BUTTON */
[data-testid="stFileUploader"] div button {
    transition: all 0.2s ease;
}

/* ✅ Hover effect on actual button */
[data-testid="stFileUploader"] div button:hover {
    background: #e5e7eb !important;
    color: #111827 !important;
    border-color: #9ca3af !important;
}
/* 🔥 ACTUAL Browse Files Button (REAL FIX) */
input[type="file"]::file-selector-button {
    background: transparent;
    border: 1px solid #cbd5f5;
    padding: 6px 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
}

/* ✅ HOVER EFFECT */
input[type="file"]::file-selector-button:hover {
    background: #cbd5f5;
    color: #000000;
    border-color: #6366f1;
}

/* 🔥 SIDEBAR SECTION TITLES (Upload Documents / Chat History) */
section[data-testid="stSidebar"] h3 {
    font-size: 22px !important;     /* ⬆ increase title size */
    font-weight: 700 !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* 🔥 ICON SIZE (emoji) */
section[data-testid="stSidebar"] h3 {
    line-height: 1.4;
}

section[data-testid="stSidebar"] h3::first-letter {
    font-size: 24px; /* ⬆ icon size */
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='
    display:flex;
    justify-content:center;
    align-items:center;
    gap:16px;
    font-size:48px;        /* 🔥 BIG TITLE */
    font-weight:800;
    margin-top:10px;
    margin-bottom:20px;
'>
    <span style="font-size:52px;">🤖</span>
    <span>AI Chatbot Assistant</span>
</h1>
""", unsafe_allow_html=True)

# -------------------------
# SESSION STATE (PERSISTENT)
# -------------------------
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if "current_chat" not in st.session_state:
    # Always create a fresh new chat on app start
    base = "New Chat"
    name = base
    count = 1

    while name in st.session_state.chats:
        name = f"{base} {count}"
        count += 1

    st.session_state.chats[name] = []
    st.session_state.current_chat = name
    save_chats(st.session_state.chats)
# ✅ ADD THIS LINE HERE
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []
# -------------------------
# SIDEBAR (CHAT HISTORY)
# -------------------------
with st.sidebar:

    # -------------------------
    # 📦 BOX 1: Upload Documents
    # -------------------------
    with st.container():
        

        uploaded_files = st.file_uploader(
            "Upload Files",
            type=["png", "jpg", "jpeg", "txt", "pdf"],
            accept_multiple_files=True
        )

        uploaded_image = None
        uploaded_file = None

        uploaded_images = []
        uploaded_docs = []

        if uploaded_files:
            for file in uploaded_files:
                if file.type.startswith("image"):
                    uploaded_images.append(file)
                else:
                    uploaded_docs.append(file)
        uploaded_image = uploaded_images[0] if uploaded_images else None
        uploaded_file = uploaded_docs[0] if uploaded_docs else None

    st.divider()


    # -------------------------
    # 📦 BOX 3: Chat History
    # -------------------------
    with st.container():

        st.markdown("""
        <h3 style="font-size:24px; font-weight:700; display:flex; align-items:center; gap:10px;">
            <span style="font-size:26px;">💬</span> Chat History
        </h3>
        """, unsafe_allow_html=True)

        if st.button("➕ New Chat", use_container_width=True):
            base = "New Chat"
            name = base
            count = 1

            while name in st.session_state.chats:
                name = f"{base} {count}"
                count += 1

            st.session_state.chats[name] = []
            st.session_state.current_chat = name
            save_chats(st.session_state.chats)
            

        st.divider()

        for chat_name in list(st.session_state.chats.keys()):

            is_active = chat_name == st.session_state.current_chat

            row_class = "chat-row chat-active" if is_active else "chat-row"

            cols = st.columns([0.85, 0.15])

            # Chat select
            if cols[0].button(chat_name, key=f"chat_{chat_name}", use_container_width=True):
                st.session_state.current_chat = chat_name
                st.rerun()

            # Delete button (aligned perfectly)
            if cols[1].button("🗑️", key=f"del_{chat_name}", help="Delete chat"):
                if len(st.session_state.chats) > 1:
                    del st.session_state.chats[chat_name]
                    st.session_state.current_chat = list(st.session_state.chats.keys())[0]
                    save_chats(st.session_state.chats)
                    st.rerun()
                            
# -------------------------
# CURRENT CHAT
# -------------------------
messages = st.session_state.chats[st.session_state.current_chat]

# -------------------------
# SHOW UPLOADED CONTENT (CENTER)
# -------------------------


for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])




uploaded_text = ""

if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    if uploaded_file.type == "text/plain":
        uploaded_text = uploaded_file.read().decode("utf-8")

    elif uploaded_file.type == "application/pdf":
        from PyPDF2 import PdfReader

        pdf = PdfReader(uploaded_file)

        for page in pdf.pages:
            text = page.extract_text()
            if text:
                uploaded_text += text

        MAX_CHARS = 12000

        if len(uploaded_text) > MAX_CHARS:
            uploaded_text = uploaded_text[:MAX_CHARS] + "\n\n[Content truncated]"

# -------------------------
# Cache Embeddings
# -------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -------------------------
# Build Knowledge Base
# -------------------------
@st.cache_resource
def load_vector_db():

    loader = TextLoader("data/faq.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    if not os.path.exists("chroma_db"):

        db = Chroma.from_documents(
            docs,
            embeddings,
            collection_name="knowledge_base",
            persist_directory="chroma_db"
        )

    else:

        db = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings,
            collection_name="knowledge_base"
        )

    return db

db = load_vector_db()
retriever = db.as_retriever(search_kwargs={"k": 3})

# -------------------------
# Cache Models
# -------------------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
llm = load_llm()

def generate_full_response(prompt, placeholder, thinking_placeholder):
    full_response = ""

    try:
        stream = llm.stream(prompt)

        first_token = True

        for chunk in stream:
            if chunk and chunk.content:

                # ✅ REMOVE "Thinking..." immediately
                if first_token:
                    thinking_placeholder.empty()
                    first_token = False

                full_response += chunk.content

                safe_text = full_response.replace("�", "")
                placeholder.markdown(safe_text + "▌")

        return full_response

    except Exception as e:
        thinking_placeholder.empty()
        return f"Error: {str(e)}"

# -------------------------
# SHOW UPLOADED FILE (like ChatGPT)
# -------------------------
if uploaded_image:
    st.image(uploaded_image, use_container_width=True)

if uploaded_file:
    st.markdown(f"""
    <div style="
        padding:8px 12px;
        border-radius:8px;
        background:#f1f1f1;
        margin-bottom:10px;
        font-size:14px;">
        📄 {uploaded_file.name}
    </div>
    """, unsafe_allow_html=True)
# -------------------------
# Chat Input
# -------------------------
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

if uploaded_file:
    st.success(f"📄 Uploaded: {uploaded_file.name}")

question = st.chat_input("Ask anything...")

if question:

    current_chat = st.session_state.current_chat
    # 🔥 AUTO TITLE GENERATION (ONLY FIRST MESSAGE)
    if len(messages) == 0:

        if uploaded_image:
            mode = "image"
        elif uploaded_file:
            mode = "file"
        else:
            mode = "text"

        new_title = generate_title_with_ai(question, mode)

        # Ensure unique title
        original_title = new_title
        count = 1
        while new_title in st.session_state.chats:
            new_title = f"{original_title} {count}"
            count += 1

        # Rename chat
        st.session_state.chats[new_title] = st.session_state.chats.pop(current_chat)
        st.session_state.current_chat = new_title
        current_chat = new_title

    # Add user message
    messages.append({"role": "user", "content": question})
    save_chats(st.session_state.chats)

    # ✅ SHOW USER MESSAGE IMMEDIATELY
    with st.chat_message("user"):
        st.markdown(f"🧑‍💻 **You**  \n{question}")

    # ✅ ASSISTANT RESPONSE
    with st.chat_message("assistant"):
        st.markdown("🤖 **Assistant**")

        response_placeholder = st.empty()

        with st.spinner("🔍 Searching documents..."):

            if uploaded_image:
                image = Image.open(uploaded_image)
                extracted_text = pytesseract.image_to_string(image)

                file_content = extracted_text if extracted_text else "No text found in image."

                # 🔥 ADD THIS (RAG for images also)
                results = retriever.invoke(question) or []
                context = "\n".join(
                    [doc.page_content for doc in results if doc.page_content]
                )

                prompt = f"""
                You are a precise AI assistant.

                STRICT RULES:
                - NEVER say "I need more context"
                - NEVER apologize
                - NEVER stop mid-answer
                - If information is incomplete, still give the BEST possible answer
                - DO NOT switch topic
                - COMPLETE the answer fully

                -----------------------
                KNOWLEDGE BASE:
                {context}

                -----------------------
                FILE CONTENT:
                {file_content}

                -----------------------
                QUESTION:
                {question}

                -----------------------
                FINAL ANSWER (complete, no cut-off):
                """

                

            else:
                results = retriever.invoke(question) or []

                context = "\n".join(
                    [doc.page_content for doc in results if doc.page_content]
                )

                file_content = uploaded_text if uploaded_text else "No file uploaded."

                prompt = f"""
                You are a smart AI assistant.

                Knowledge Base:
                {context}

                File Content:
                {file_content}

                Question:
                {question}

                Answer:
                """
        thinking_placeholder = st.empty()

        thinking_placeholder.markdown("Thinking...")
        # ✅ Generate response
        full_response = generate_full_response(
            prompt,
            response_placeholder,
            thinking_placeholder
        )

        response_placeholder.markdown(full_response)

        answer = full_response

    # ✅ Save assistant response
    messages.append({"role": "assistant", "content": answer})
    save_chats(st.session_state.chats)