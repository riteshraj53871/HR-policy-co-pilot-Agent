// Theme toggle handling
const themeToggle = document.getElementById("theme-toggle");
function applyTheme(theme) {
    document.body.classList.toggle("dark", theme === "dark");
    localStorage.setItem("theme", theme);
}
function initTheme() {
    const saved = localStorage.getItem("theme");
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = saved || (prefersDark ? "dark" : "light");
    applyTheme(theme);
}
if (themeToggle) {
    themeToggle.addEventListener("click", () => {
        const isDark = document.body.classList.contains("dark");
        applyTheme(isDark ? "light" : "dark");
    });
}
initTheme();

const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const examples = document.getElementById("examples");

const SESSION_ID = (() => {
    try {
        let id = sessionStorage.getItem("hr_session");
        if (!id) {
            id = crypto.randomUUID().replace(/-/g, "").slice(0, 32);
            sessionStorage.setItem("hr_session", id);
        }
        return id;
    } catch {
        return null; // storage blocked: the chat still works, just without a session label
    }
})();

const BADGES = {
    kb: { label: "Company policy", cls: "kb" },
    web: { label: "External information, verify with HR", cls: "web" },
    none: { label: "Insufficient evidence", cls: "none" },
    direct: { label: "Assistant", cls: "direct" },
};

const EXAMPLES = [
    "How many annual leave days do employees receive?",
    "Can I work from home?",
    "What are the latest public holiday rules in Bangladesh?",
];

// ---------- safe rendering (textContent only, never innerHTML) ----------

function addInline(parent, text) {
    text.split(/\*\*(.+?)\*\*/g).forEach((part, i) => {
        if (!part) return;
        if (i % 2 === 1) {
            const b = document.createElement("strong");
            b.textContent = part;
            parent.appendChild(b);
        } else {
            parent.appendChild(document.createTextNode(part));
        }
    });
}

function renderText(container, text) {
    let list = null;
    text.split("\n").forEach((raw) => {
        const line = raw.trim();
        if (!line) { list = null; return; }
        const item = line.match(/^[-*]\s+(.*)$/);
        if (item) {
            if (!list) { list = document.createElement("ul"); container.appendChild(list); }
            const li = document.createElement("li");
            addInline(li, item[1]);
            list.appendChild(li);
        } else {
            list = null;
            const p = document.createElement("p");
            addInline(p, line);
            container.appendChild(p);
        }
    });
}

function addMessage(role) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
    return el;
}

function renderSources(el, citations) {
    if (!citations || !citations.length) return;
    const box = document.createElement("details");
    box.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = `Sources (${citations.length})`;
    box.appendChild(summary);
    const ul = document.createElement("ul");
    citations.forEach((c) => {
        const li = document.createElement("li");
        if (c.url && /^https?:\/\//i.test(c.url)) {
            const a = document.createElement("a");
            a.href = c.url;
            a.target = "_blank";
            a.rel = "noopener noreferrer";
            a.textContent = c.title || c.url;
            li.appendChild(a);
        } else {
            li.textContent = [c.title, c.section].filter(Boolean).join(" > ");
        }
        ul.appendChild(li);
    });
    box.appendChild(ul);
    el.appendChild(box);
}

function renderFeedback(el, requestId) {
    const row = document.createElement("div");
    row.className = "feedback";
    const label = document.createElement("span");
    label.textContent = "Was this helpful?";
    row.appendChild(label);

    [["up", "\u{1F44D}"], ["down", "\u{1F44E}"]].forEach(([rating, icon]) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = icon;
        btn.setAttribute("aria-label", rating === "up" ? "Helpful" : "Not helpful");
        btn.addEventListener("click", async () => {
            row.querySelectorAll("button").forEach((b) => (b.disabled = true));
            try {
                const res = await fetch("/api/feedback", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ request_id: requestId, rating }),
                });
                label.textContent = res.ok ? "Thanks for the feedback." : "Could not save feedback.";
            } catch {
                label.textContent = "Could not save feedback.";
            }
        });
        row.appendChild(btn);
    });
    el.appendChild(row);
}

function renderBotAnswer(el, data) {
    el.textContent = "";
    const badge = BADGES[data.source_type] || BADGES.direct;
    const tag = document.createElement("span");
    tag.className = `badge ${badge.cls}`;
    tag.textContent = badge.label;
    el.appendChild(tag);
    renderText(el, data.answer);
    renderSources(el, data.citations);
    renderFeedback(el, data.request_id);
}

// ---------- sending ----------

async function ask(question) {
    addMessage("user").textContent = question;
    const bot = addMessage("bot");
    bot.innerHTML = '<span class="thinking">Looking through the HR policies...</span>';
    input.disabled = sendBtn.disabled = true;

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, session_id: SESSION_ID }),
        });
        if (!res.ok) {
            let msg = "Something went wrong. Please try again or contact HR.";
            try {
                const err = await res.json();
                if (typeof err.detail === "string") msg = err.detail;
            } catch { /* keep the default message */ }
            bot.textContent = msg;
            return;
        }
        renderBotAnswer(bot, await res.json());
    } catch {
        bot.textContent = "Could not reach the server. Please check your connection and try again.";
    } finally {
        input.disabled = sendBtn.disabled = false;
        input.focus();
        chat.scrollTop = chat.scrollHeight;
    }
}

form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    examples.hidden = true;
    ask(q);
});

EXAMPLES.forEach((q) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = q;
    b.addEventListener("click", () => { examples.hidden = true; ask(q); });
    examples.appendChild(b);
});

const hello = addMessage("bot");
renderText(hello, "Hi! I can answer questions about company HR policies. Try one of the examples below, or type your own.");