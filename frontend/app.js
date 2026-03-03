/**
 * Mentat frontend — vanilla JS, no build tooling.
 *
 * State shape:
 *   {
 *     conversations: [{ id, title, messages: [{role, content, think?: string}] }],
 *     activeConversationId: string | null
 *   }
 *
 * State is persisted to localStorage on every mutation.
 * All render functions are pure: they rebuild DOM from state.
 */

const API_URL = "/api/chat";
const STREAM_URL = "/api/chat/stream";
const STORAGE_KEY = "mentat_state";

let currentAbortController = null;

// ── State ──────────────────────────────────────────────────────────────────

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch (_) { /* ignore parse errors */ }
  return { conversations: [], activeConversationId: null };
}

function saveState(state) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

let state = loadState();

function setState(updater) {
  state = updater(state);
  saveState(state);
  render();
}

// ── Helpers ────────────────────────────────────────────────────────────────

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function activeConversation() {
  return state.conversations.find((c) => c.id === state.activeConversationId) || null;
}

function titleFromMessage(content) {
  return content.slice(0, 50) + (content.length > 50 ? "…" : "");
}

/** Apply inline markdown: inline code, bold, italic. */
function applyInline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<!\*)\*([^*\n]+?)\*(?!\*)/g, "<em>$1</em>");
}

/** Full markdown → HTML renderer. Handles headers, lists, code blocks, inline formatting. */
function renderMarkdown(text) {
  // Escape HTML characters first
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  const lines = escaped.split("\n");
  const out = [];
  let inCode = false;
  let codeBuf = [];
  let inUl = false;
  let inOl = false;

  const closeList = () => {
    if (inUl) { out.push("</ul>"); inUl = false; }
    if (inOl) { out.push("</ol>"); inOl = false; }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Fenced code block toggle
    if (line.trimStart().startsWith("```")) {
      if (inCode) {
        out.push(`<pre><code>${codeBuf.join("\n")}</code></pre>`);
        codeBuf = [];
        inCode = false;
      } else {
        closeList();
        inCode = true;
      }
      continue;
    }
    if (inCode) { codeBuf.push(line); continue; }

    // Headers
    if (/^### /.test(line)) { closeList(); out.push(`<h4>${applyInline(line.slice(4))}</h4>`); continue; }
    if (/^## /.test(line))  { closeList(); out.push(`<h3>${applyInline(line.slice(3))}</h3>`); continue; }
    if (/^# /.test(line))   { closeList(); out.push(`<h3>${applyInline(line.slice(2))}</h3>`); continue; }

    // Unordered list item
    const ulM = line.match(/^[*-] (.+)/);
    if (ulM) {
      if (inOl) { out.push("</ol>"); inOl = false; }
      if (!inUl) { out.push("<ul>"); inUl = true; }
      out.push(`<li>${applyInline(ulM[1])}</li>`);
      continue;
    }

    // Ordered list item
    const olM = line.match(/^\d+\. (.+)/);
    if (olM) {
      if (inUl) { out.push("</ul>"); inUl = false; }
      if (!inOl) { out.push("<ol>"); inOl = true; }
      out.push(`<li>${applyInline(olM[1])}</li>`);
      continue;
    }

    // Blank line → paragraph break
    if (line.trim() === "") {
      closeList();
      if (i < lines.length - 1) out.push("<br>");
      continue;
    }

    // Regular text line
    closeList();
    out.push(applyInline(line));
  }

  closeList();
  if (inCode) out.push(`<pre><code>${codeBuf.join("\n")}</code></pre>`);

  return out.join("\n");
}

// ── Render ─────────────────────────────────────────────────────────────────

function render() {
  renderSidebar();
  renderMessages();
}

function renderSidebar() {
  const list = document.getElementById("conversation-list");
  list.innerHTML = "";

  [...state.conversations].reverse().forEach((conv) => {
    const item = document.createElement("div");
    item.className = "conv-item" + (conv.id === state.activeConversationId ? " active" : "");
    item.textContent = conv.title || "New Conversation";
    item.addEventListener("click", () => selectConversation(conv.id));
    list.appendChild(item);
  });
}

function renderMessages() {
  const container = document.getElementById("messages");
  const conv = activeConversation();

  if (!conv) {
    container.innerHTML = `
      <div class="empty-state">
        <h2>Welcome to Mentat</h2>
        <p>Your AI executive coach. Start a new conversation to begin.</p>
      </div>`;
    document.getElementById("chat-title").textContent = "Mentat";
    return;
  }

  document.getElementById("chat-title").textContent = conv.title || "New Conversation";

  container.innerHTML = "";
  conv.messages.forEach((msg) => {
    container.appendChild(createMessageEl(msg.role, msg.content, msg.think));
  });

  container.scrollTop = container.scrollHeight;
}

function createMessageEl(role, content, think = null) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (role === "assistant") {
    bubble.innerHTML = renderMarkdown(content);
  } else {
    // User messages: preserve text as-is (CSS handles pre-wrap)
    bubble.textContent = content;
  }

  wrapper.appendChild(bubble);

  if (think) {
    const thinkBlock = document.createElement("div");
    thinkBlock.className = "think-block";
    thinkBlock.innerHTML = renderMarkdown(think);
    wrapper.appendChild(thinkBlock);
  }

  return wrapper;
}

function showTypingIndicator() {
  const container = document.getElementById("messages");
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";
  wrapper.id = "typing-indicator";

  const bubble = document.createElement("div");
  bubble.className = "bubble typing-indicator";
  bubble.innerHTML = "<span></span><span></span><span></span>";

  wrapper.appendChild(bubble);
  container.appendChild(wrapper);
  container.scrollTop = container.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

// ── Status message (SSE ephemeral bubble) ─────────────────────────────────

function showStatusMessage(text) {
  const container = document.getElementById("messages");
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";
  wrapper.id = "status-msg";

  const bubble = document.createElement("div");
  bubble.className = "bubble status-message";
  bubble.textContent = text;

  wrapper.appendChild(bubble);
  container.appendChild(wrapper);
  container.scrollTop = container.scrollHeight;
}

function updateStatusMessage(text) {
  const el = document.getElementById("status-msg");
  if (el) el.querySelector(".bubble").textContent = text;
}

function removeStatusMessage() {
  const el = document.getElementById("status-msg");
  if (el) el.remove();
}

// ── SSE parsing ────────────────────────────────────────────────────────────

/**
 * Parse complete SSE events from a buffer.
 * Returns { parsed: Event[], remainder: string } where remainder is any
 * incomplete trailing data that hasn't yet formed a full event.
 */
function parseSSEBuffer(buffer) {
  const parsed = [];
  const chunks = buffer.split("\n\n");
  // The last element may be an incomplete event; keep it as the remainder
  const complete = chunks.slice(0, -1);
  const remainder = chunks[chunks.length - 1];

  for (const chunk of complete) {
    const line = chunk.trim();
    if (!line.startsWith("data: ")) continue;
    try {
      parsed.push(JSON.parse(line.slice(6)));
    } catch (_) { /* ignore malformed JSON */ }
  }

  return { parsed, remainder };
}

// ── Actions ────────────────────────────────────────────────────────────────

function newConversation() {
  const id = uid();
  setState((s) => ({
    ...s,
    conversations: [...s.conversations, { id, title: "", messages: [] }],
    activeConversationId: id,
  }));
  document.getElementById("message-input").focus();
}

function selectConversation(id) {
  setState((s) => ({ ...s, activeConversationId: id }));
}

function appendMessage(convId, role, content, think = null) {
  setState((s) => ({
    ...s,
    conversations: s.conversations.map((c) => {
      if (c.id !== convId) return c;
      const messages = [...c.messages, { role, content, think }];
      const title = c.title || titleFromMessage(c.messages[0]?.content || content);
      return { ...c, messages, title };
    }),
  }));
}

// ── API ────────────────────────────────────────────────────────────────────

async function sendMessage(content) {
  const conv = activeConversation();
  if (!conv) return;

  const convId = conv.id;

  // Optimistically add user message and show status bubble
  appendMessage(convId, "user", content);
  showStatusMessage("Connecting\u2026");
  setStreaming(true);

  const messages = activeConversation().messages.map((m) => ({
    role: m.role,
    content: m.content,
  }));

  currentAbortController = new AbortController();
  let aborted = false;

  try {
    const response = await fetch(STREAM_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, session_id: convId }),
      signal: currentAbortController.signal,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let thinkContent = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { parsed, remainder } = parseSSEBuffer(buffer);
      buffer = remainder;
      for (const event of parsed) {
        if (event.type === "status") updateStatusMessage(event.message);
        if (event.type === "think") thinkContent = event.content;
        if (event.type === "reply") {
          removeStatusMessage();
          appendMessage(convId, "assistant", event.content, thinkContent);
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      aborted = true;
      removeStatusMessage();
      removeLastMessage(convId);
    } else {
      removeStatusMessage();
      appendMessage(convId, "assistant", `Error: ${err.message}`);
    }
  } finally {
    currentAbortController = null;
    setStreaming(false, aborted ? content : null);
    document.getElementById("message-input").focus();
  }
}

async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/documents/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    const conv = activeConversation();
    if (conv) {
      appendMessage(
        conv.id,
        "assistant",
        `\uD83D\uDCC4 Uploaded: ${data.filename} (${data.chunks_stored} chunks stored)`,
      );
    }
  } catch (err) {
    const conv = activeConversation();
    if (conv) {
      appendMessage(conv.id, "assistant", `Error uploading file: ${err.message}`);
    }
  }
}

// ── UI helpers ─────────────────────────────────────────────────────────────

function setSendDisabled(disabled) {
  document.getElementById("send-btn").disabled = disabled;
}

/**
 * Toggle streaming mode: show stop button / hide send button, or vice versa.
 * @param {boolean} active - true when streaming is in progress
 * @param {string|null} restoreContent - if provided, restore this text to the input
 */
function setStreaming(active, restoreContent = null) {
  const sendBtn = document.getElementById("send-btn");
  const stopBtn = document.getElementById("stop-btn");
  if (active) {
    sendBtn.style.display = "none";
    stopBtn.style.display = "flex";
  } else {
    stopBtn.style.display = "none";
    sendBtn.style.display = "flex";
    if (restoreContent !== null) {
      const input = document.getElementById("message-input");
      input.value = restoreContent;
      autoResize(input);
    }
  }
}

function removeLastMessage(convId) {
  setState((s) => ({
    ...s,
    conversations: s.conversations.map((c) => {
      if (c.id !== convId) return c;
      return { ...c, messages: c.messages.slice(0, -1) };
    }),
  }));
}

function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 180) + "px";
}

// ── Event wiring ───────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message-input");
  const newChatBtn = document.getElementById("new-chat-btn");
  const attachBtn = document.getElementById("attach-btn");
  const fileInput = document.getElementById("file-input");
  const stopBtn = document.getElementById("stop-btn");

  newChatBtn.addEventListener("click", newConversation);

  stopBtn.addEventListener("click", () => {
    if (currentAbortController) {
      currentAbortController.abort();
    }
  });

  input.addEventListener("input", () => autoResize(input));

  // Enter submits; Shift+Enter inserts a newline
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  attachBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    if (!activeConversation()) newConversation();
    uploadDocument(file);
    fileInput.value = ""; // reset so the same file can be re-uploaded
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const content = input.value.trim();
    if (!content) return;

    // Start a new conversation if none is active
    if (!activeConversation()) newConversation();

    input.value = "";
    autoResize(input);
    sendMessage(content);
  });

  const thinkToggleCb = document.getElementById("think-toggle-cb");
  thinkToggleCb.addEventListener("change", (e) => {
    document.body.classList.toggle("show-think", e.target.checked);
  });

  render();

  // Open a new conversation automatically if none exist
  if (state.conversations.length === 0) {
    // Leave empty — show welcome screen
  }
});
