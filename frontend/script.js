// script.js
// ---------------------------------------------------------------
// Drives the ChatGPT-style chat UI and talks to the FastAPI backend
// (server.py) which wraps the LangGraph Agentic RAG pipeline.
// ---------------------------------------------------------------

const API_BASE = ""; // same origin (server.py serves this frontend too)

const messagesEl = document.getElementById("messages");
const emptyStateEl = document.getElementById("emptyState");
const composerForm = document.getElementById("composerForm");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const conversationListEl = document.getElementById("conversationList");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const suggestionGrid = document.getElementById("suggestionGrid");

// In-memory store of all conversations for this session.
// Each conversation: { id, title, messages: [{role, text}] }
let conversations = [];
let activeConversationId = null;
let isStreaming = false;

// ---------------- Initialization ----------------

init();

function init() {
  startNewConversation();
  checkBackendHealth();
  setInterval(checkBackendHealth, 15000);

  composerForm.addEventListener("submit", onSubmit);
  messageInput.addEventListener("input", onInputChange);
  messageInput.addEventListener("keydown", onKeyDown);
  newChatBtn.addEventListener("click", () => startNewConversation());

  suggestionGrid.addEventListener("click", (e) => {
    const card = e.target.closest(".suggestion-card");
    if (!card) return;
    messageInput.value = card.dataset.text;
    onInputChange();
    composerForm.requestSubmit();
  });
}

// ---------------- Backend health check ----------------

async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { method: "GET" });
    if (res.ok) {
      setStatus(true);
    } else {
      setStatus(false);
    }
  } catch (err) {
    setStatus(false);
  }
}

function setStatus(online) {
  statusDot.classList.remove("online", "offline");
  statusDot.classList.add(online ? "online" : "offline");
  statusText.textContent = online ? "Backend connected" : "Backend offline";
}

// ---------------- Conversation management ----------------

function startNewConversation() {
  const id = `conv-${Date.now()}`;
  const conversation = { id, title: "New chat", messages: [] };
  conversations.unshift(conversation);
  activeConversationId = id;
  renderConversationList();
  renderMessages();
  messageInput.value = "";
  onInputChange();
  messageInput.focus();
}

function getActiveConversation() {
  return conversations.find((c) => c.id === activeConversationId);
}

function switchConversation(id) {
  if (isStreaming) return; // avoid switching mid-stream for simplicity
  activeConversationId = id;
  renderConversationList();
  renderMessages();
}

function renderConversationList() {
  conversationListEl.innerHTML = "";
  conversations.forEach((conv) => {
    const item = document.createElement("div");
    item.className = "conversation-item" + (conv.id === activeConversationId ? " active" : "");
    item.textContent = conv.title;
    item.addEventListener("click", () => switchConversation(conv.id));
    conversationListEl.appendChild(item);
  });
}

// ---------------- Composer behavior ----------------

function onInputChange() {
  // Auto-grow the textarea up to its CSS max-height.
  messageInput.style.height = "auto";
  messageInput.style.height = `${messageInput.scrollHeight}px`;
  sendBtn.disabled = messageInput.value.trim().length === 0 || isStreaming;
}

function onKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) {
      composerForm.requestSubmit();
    }
  }
}

async function onSubmit(e) {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text || isStreaming) return;

  const conversation = getActiveConversation();
  conversation.messages.push({ role: "user", text });
  if (conversation.messages.length === 1) {
    conversation.title = text.slice(0, 40) + (text.length > 40 ? "..." : "");
    renderConversationList();
  }

  messageInput.value = "";
  onInputChange();
  renderMessages();
  scrollToBottom();

  await streamAssistantReply(text, conversation);
}

// ---------------- Talking to the backend ----------------

async function streamAssistantReply(userText, conversation) {
  isStreaming = true;
  sendBtn.disabled = true;

  // Placeholder assistant message that we will progressively fill in
  // with step updates, then the final answer.
  const assistantMessage = { role: "assistant", text: "", steps: [], pending: true };
  conversation.messages.push(assistantMessage);
  renderMessages();
  scrollToBottom();

  try {
    const response = await fetch(`${API_BASE}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: userText }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Server responded with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by a blank line.
      const events = buffer.split("\n\n");
      buffer = events.pop(); // keep last partial chunk in buffer

      for (const rawEvent of events) {
        const line = rawEvent.trim();
        if (!line.startsWith("data:")) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;

        let event;
        try {
          event = JSON.parse(jsonStr);
        } catch {
          continue;
        }

        handleStreamEvent(event, assistantMessage);
        renderMessages();
        scrollToBottom();
      }
    }
  } catch (err) {
    assistantMessage.pending = false;
    assistantMessage.error = true;
    assistantMessage.text =
      "Could not reach the backend. Make sure server.py is running " +
      "(uvicorn server:app --port 8000) and try again.";
    renderMessages();
  } finally {
    isStreaming = false;
    onInputChange();
  }
}

function handleStreamEvent(event, assistantMessage) {
  if (event.type === "step") {
    assistantMessage.steps.push(event.label);
  } else if (event.type === "final") {
    assistantMessage.pending = false;
    assistantMessage.text = event.answer || "(No answer returned.)";
  } else if (event.type === "error") {
    assistantMessage.pending = false;
    assistantMessage.error = true;
    assistantMessage.text = `Backend error: ${event.message}`;
  }
}

// ---------------- Rendering ----------------

function renderMessages() {
  const conversation = getActiveConversation();
  messagesEl.innerHTML = "";

  if (!conversation || conversation.messages.length === 0) {
    messagesEl.appendChild(emptyStateEl);
    return;
  }

  conversation.messages.forEach((msg) => {
    messagesEl.appendChild(renderMessageRow(msg));
  });
}

function renderMessageRow(msg) {
  const row = document.createElement("div");
  row.className = `message-row ${msg.role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${msg.role}`;
  avatar.textContent = msg.role === "user" ? "U" : "A";
  row.appendChild(avatar);

  const body = document.createElement("div");
  body.className = "message-body";

  if (msg.role === "assistant" && msg.steps && msg.steps.length > 0 && msg.pending) {
    body.appendChild(renderStepTrace(msg.steps));
  }

  if (msg.role === "assistant" && msg.pending && (!msg.steps || msg.steps.length === 0)) {
    body.appendChild(renderTypingDots());
  }

  if (!msg.pending) {
    const textEl = document.createElement("div");
    textEl.className = msg.error ? "message-text error-text" : "message-text";
    textEl.textContent = msg.text;
    body.appendChild(textEl);
  } else if (msg.role === "user") {
    const textEl = document.createElement("div");
    textEl.className = "message-text";
    textEl.textContent = msg.text;
    body.appendChild(textEl);
  }

  row.appendChild(body);
  return row;
}

function renderStepTrace(steps) {
  const trace = document.createElement("div");
  trace.className = "step-trace";

  steps.forEach((label, idx) => {
    const isLast = idx === steps.length - 1;
    const item = document.createElement("div");
    item.className = "step-item" + (isLast ? "" : " done");
    item.style.animationDelay = `${idx * 0.03}s`;

    const icon = document.createElement("span");
    icon.className = "step-icon";
    if (isLast) {
      icon.innerHTML = '<span class="spinner"></span>';
    } else {
      icon.innerHTML =
        '<svg class="check" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12l5 5L20 7" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    }

    item.appendChild(icon);
    const text = document.createElement("span");
    text.textContent = label;
    item.appendChild(text);

    trace.appendChild(item);
  });

  return trace;
}

function renderTypingDots() {
  const wrap = document.createElement("div");
  wrap.className = "typing-dots";
  wrap.innerHTML = "<span></span><span></span><span></span>";
  return wrap;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
