/**
 * Mentat frontend — vanilla JS, no build tooling.
 *
 * State shape:
 *   {
 *     conversations: [{ id, title, messages: [{role, content}] }],
 *     activeConversationId: string | null
 *   }
 *
 * State is persisted to localStorage on every mutation.
 * All render functions are pure: they rebuild DOM from state.
 */

const API_URL = "/api/chat";
const STORAGE_KEY = "mentat_state";

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

/** Minimal markdown → HTML: bold (**text**) only. */
function renderMarkdown(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
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
    container.appendChild(createMessageEl(msg.role, msg.content));
  });

  container.scrollTop = container.scrollHeight;
}

function createMessageEl(role, content) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = renderMarkdown(content);

  wrapper.appendChild(bubble);
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

function appendMessage(convId, role, content) {
  setState((s) => ({
    ...s,
    conversations: s.conversations.map((c) => {
      if (c.id !== convId) return c;
      const messages = [...c.messages, { role, content }];
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

  // Optimistically add user message
  appendMessage(convId, "user", content);
  showTypingIndicator();
  setSendDisabled(true);

  const messages = activeConversation().messages.map((m) => ({
    role: m.role,
    content: m.content,
  }));

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, session_id: convId }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    removeTypingIndicator();
    appendMessage(convId, "assistant", data.reply);
  } catch (err) {
    removeTypingIndicator();
    appendMessage(convId, "assistant", `Error: ${err.message}`);
  } finally {
    setSendDisabled(false);
    document.getElementById("message-input").focus();
  }
}

// ── UI helpers ─────────────────────────────────────────────────────────────

function setSendDisabled(disabled) {
  document.getElementById("send-btn").disabled = disabled;
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

  newChatBtn.addEventListener("click", newConversation);

  input.addEventListener("input", () => autoResize(input));

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
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

  render();

  // Open a new conversation automatically if none exist
  if (state.conversations.length === 0) {
    // Leave empty — show welcome screen
  }
});
