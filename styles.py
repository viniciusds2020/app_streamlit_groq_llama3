CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #f6f6f8;
    --surface: #ffffff;
    --border: #e7e7ec;
    --text: #1d1d27;
    --text-muted: #70707d;
    --accent: #6f5bf6;
    --accent-hover: #5a46e0;
    --assistant-bubble: #ffffff;
    --assistant-text: #1d1d27;
    --shadow: 0 1px 3px rgba(20, 20, 30, 0.06), 0 1px 2px rgba(20, 20, 30, 0.04);
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg: #111114;
        --surface: #1a1a1f;
        --border: #2b2b33;
        --text: #efeff2;
        --text-muted: #9a9aa6;
        --assistant-bubble: #1f1f26;
        --assistant-text: #efeff2;
        --shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.stApp { background: var(--bg); }

[data-testid="stMainBlockContainer"] {
    max-width: 760px;
    padding-top: 2rem;
}

/* ---- Hero header ---- */
.app-hero {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.75rem;
}
.app-hero .app-icon {
    font-size: 2rem;
    line-height: 1;
}
.app-hero h1 {
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--text);
    margin: 0;
}
.app-hero p {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin: 0;
}

/* ---- Empty state ---- */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 3.5rem 1.5rem;
    border: 1px dashed var(--border);
    border-radius: 20px;
    background: var(--surface);
    color: var(--text-muted);
    gap: 0.4rem;
}
.empty-state .big-icon { font-size: 2.75rem; margin-bottom: 0.5rem; }
.empty-state strong { color: var(--text); font-size: 1.05rem; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .sidebar-brand {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.1rem;
}
section[data-testid="stSidebar"] .sidebar-sub {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 1.25rem;
}
[data-testid="stFileUploaderDropzone"] {
    border-radius: 14px;
    border: 1.5px dashed var(--border);
    background: var(--bg);
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(111, 91, 246, 0.12);
    color: var(--accent);
    margin-top: 0.75rem;
}

/* ---- Chat bubbles ---- */
[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0.35rem 0;
}
[data-testid="stChatMessageContent"] {
    border-radius: 18px;
    padding: 0.75rem 1.05rem;
    box-shadow: var(--shadow);
    font-size: 0.93rem;
    line-height: 1.5;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: var(--accent);
    color: #ffffff;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background: var(--assistant-bubble);
    color: var(--assistant-text);
    border: 1px solid var(--border);
}

/* ---- Chat input ---- */
[data-testid="stChatInput"] {
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface);
    box-shadow: var(--shadow);
}
[data-testid="stChatInputTextArea"] { font-size: 0.93rem; }
[data-testid="stChatInput"] button[kind="primary"],
[data-testid="stChatInputSubmitButton"] {
    background: var(--accent) !important;
    border-radius: 999px !important;
}

/* ---- Buttons ---- */
[data-testid="stBaseButton-secondary"] {
    border-radius: 10px;
    border: 1px solid var(--border);
}
[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--accent);
    color: var(--accent);
}
</style>
"""
