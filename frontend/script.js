// ============================================================
// FIX 1 — collapse long skill-tag lists behind a "show more" toggle.
// Progressive enhancement: if this script fails to run, the full
// list from index.html is shown (nothing is hidden by default CSS).
// ============================================================
(function setupSkillToggles() {
  var VISIBLE_COUNT = 12;
  var categories = document.querySelectorAll(".skill-category");

  categories.forEach(function (category) {
    var list = category.querySelector(".skill-cloud");
    if (!list) return;
    var items = Array.prototype.slice.call(list.querySelectorAll("li"));
    if (items.length <= VISIBLE_COUNT) return;

    var extra = items.slice(VISIBLE_COUNT);
    extra.forEach(function (li) {
      li.classList.add("is-extra");
    });

    var hiddenCount = extra.length;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "skill-toggle";
    btn.setAttribute("aria-expanded", "false");
    btn.innerHTML =
      '<span class="skill-toggle__label">Показать ещё ' +
      hiddenCount +
      '</span><span class="skill-toggle__chevron" aria-hidden="true">▾</span>';

    btn.addEventListener("click", function () {
      var expanded = category.classList.toggle("is-expanded");
      btn.setAttribute("aria-expanded", expanded ? "true" : "false");
      btn.querySelector(".skill-toggle__label").textContent = expanded
        ? "Свернуть"
        : "Показать ещё " + hiddenCount;
      if (!expanded) {
        // keep the button in view instead of leaving the user
        // scrolled into empty space where the tags used to be
        category.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    });

    category.classList.add("is-collapsible");
    list.insertAdjacentElement("afterend", btn);
  });
})();

// ============================================================
// FIX 2 — scroll-triggered reveal for page sections.
// Opt-in: only hides content once body.has-js is present, and that
// class is only added here — so if JS doesn't run, nothing is hidden.
// Respects prefers-reduced-motion.
// ============================================================
(function setupScrollReveal() {
  var reduceMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;
  if (reduceMotion) return;

  var sections = document.querySelectorAll(".section");
  if (!sections.length || !("IntersectionObserver" in window)) return;

  document.body.classList.add("has-js");

  // threshold is a fraction of the TARGET's own area, not the viewport's —
  // for sections taller than viewport-height / threshold, that area is
  // physically never visible at once, so the callback would never fire.
  // Using a near-zero threshold makes this trigger off the section's edge
  // entering the viewport instead, independent of how tall the section is.
  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0, rootMargin: "0px 0px -10% 0px" }
  );

  sections.forEach(function (section) {
    observer.observe(section);
  });
})();

// ============================================================
// FIX 5 (behavior) — chat widget wiring.
// Logic is unchanged from the original widget file (same
// endpoint, same payload, same DOM events) — only colors moved
// to style.css changed. Wrapped in an IIFE so it doesn't leak
// globals into the rest of this file.
// ============================================================
(function setupChatWidget() {
  // Раньше тут был жёстко зашит "http://localhost:8000" — работало только
  // локально, а после деплоя адрес нужно было руками менять в коде. Теперь
  // определяется автоматически:
  //  - открыли файл напрямую (file://) или зашли через localhost/127.0.0.1
  //    (например, `python -m http.server` из папки frontend) — backend
  //    считается локальным, на порту 8000;
  //  - любой другой домен (реальный деплой) — берём тот же домен + /api,
  //    подразумевая reverse-proxy в Nginx, который пробрасывает /api на
  //    backend (см. README, раздел "Деплой").
  // Если backend живёт на отдельном домене/поддомене — впишите его явно
  // вместо строки с window.location.origin, например:
  //   return "https://api.вашдомен.ru";
  var API_BASE = (function () {
    var isLocal =
      window.location.protocol === "file:" ||
      window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1";
    if (isLocal) return "http://localhost:8000";
    return window.location.origin + "/api";
  })();

  var SESSION_STORAGE_KEY = "faq_chat_session_id";

  function getSessionId() {
    try {
      var existing = window.localStorage.getItem(SESSION_STORAGE_KEY);
      if (existing) return existing;
      var fresh =
        window.crypto && window.crypto.randomUUID
          ? window.crypto.randomUUID()
          : "sess-" + Date.now() + "-" + Math.random().toString(16).slice(2);
      window.localStorage.setItem(SESSION_STORAGE_KEY, fresh);
      return fresh;
    } catch (e) {
      // localStorage недоступен (приватный режим и т.п.) — сессия будет
      // жить только в рамках текущей вкладки, без сохранения между визитами.
      return "sess-" + Date.now() + "-" + Math.random().toString(16).slice(2);
    }
  }

  var sessionId = getSessionId();

  var launcher = document.getElementById("chat-launcher");
  var widget = document.getElementById("chat-widget");
  var closeBtn = document.getElementById("chat-close");
  var messagesEl = document.getElementById("chat-messages");
  var inputEl = document.getElementById("chat-input");
  var sendBtn = document.getElementById("chat-send");
  if (!launcher || !widget) return;

  var isSending = false;

  function appendMessage(text, from) {
    var div = document.createElement("div");
    div.className = "chat-message " + from;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function appendTyping() {
    var div = document.createElement("div");
    div.className = "chat-message bot";
    div.id = "typing-indicator";
    div.innerHTML =
      '<div class="typing-indicator"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function removeTyping() {
    var el = document.getElementById("typing-indicator");
    if (el) el.remove();
  }

  function sendMessage() {
    if (isSending) return;
    var text = inputEl.value.trim();
    if (!text) return;

    appendMessage(text, "user");
    inputEl.value = "";

    isSending = true;
    sendBtn.disabled = true;
    appendTyping();

    fetch(API_BASE + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, top_k: 3, session_id: sessionId })
    })
      .then(function (res) {
        if (!res.ok) throw new Error("Server error");
        return res.json();
      })
      .then(function (data) {
        removeTyping();
        appendMessage(data.answer, "bot");
        if (data.session_id) {
          sessionId = data.session_id;
          try {
            window.localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
          } catch (e) {
            /* localStorage недоступен — не критично, работаем дальше в памяти */
          }
        }
      })
      .catch(function (err) {
        console.error(err);
        removeTyping();
        appendMessage("Не удалось получить ответ. Попробуйте позже.", "bot");
      })
      .finally(function () {
        isSending = false;
        sendBtn.disabled = false;
      });
  }

  launcher.addEventListener("click", function () {
    widget.style.display = "flex";
    launcher.style.display = "none";
    if (!messagesEl.hasChildNodes()) {
      appendMessage(
        "Привет! Я ассистент Ивана — спрашивайте про стек технологий, AI-агентов и Telegram-ботов, процесс работы, цены и сроки.",
        "bot"
      );
    }
    setTimeout(function () {
      inputEl.focus();
    }, 50);
  });

  closeBtn.addEventListener("click", function () {
    widget.style.display = "none";
    launcher.style.display = "flex";
  });

  sendBtn.addEventListener("click", sendMessage);

  inputEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
})();
