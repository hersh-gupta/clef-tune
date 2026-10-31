// clef-tune's page behaviour. Forms marked data-post send in the background; the answer holds
// <template data-swap="#id"> blocks that replace just those parts of the page, so only what
// changed moves. Also: polling, menus, the theme, the drop zone, the phone row pager,
// notifications, local clock times, and a way out of Hugging Face's frame.
(function () {
  "use strict";
  const THEME_KEY = "clef-tune-theme";
  const FAILED = "That didn't work. Reload the page and try again.";
  const OFFLINE = "Couldn't reach the app; check your connection and try again.";
  let returnFocus = null; // where focus goes once a menu choice's answer has been swapped in
  const store = {
    get(key) { try { return localStorage.getItem(key); } catch (e) { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* private mode */ } },
  };

  // ---------------------------------------------------------------- theme (light by default)
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme === "dark" ? "dark" : "light";
  }
  applyTheme(store.get(THEME_KEY));
  document.addEventListener("click", (event) => {
    const toggle = event.target.closest("[data-theme-toggle]");
    if (!toggle) return;
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
    store.set(THEME_KEY, next);
  });

  // ---------------------------------------------------------------- swapping parts of the page
  function swap(html) {
    const doc = new DOMParser().parseFromString(`<body>${html}</body>`, "text/html");
    doc.querySelectorAll("template[data-swap]").forEach((template) => {
      const target = document.querySelector(template.dataset.swap);
      if (!target) return;
      const active = document.activeElement;
      const keep = active && target.contains(active) && active.id ? active.id : null;
      const caret = keep && "selectionStart" in active ? [active.selectionStart, active.selectionEnd] : null;
      if (template.dataset.mode === "outer") {
        target.replaceWith(template.content.cloneNode(true));
      } else {
        target.replaceChildren(template.content.cloneNode(true));
      }
      const fresh = document.querySelector(template.dataset.swap);
      if (fresh && template.dataset.animate !== undefined) {
        fresh.classList.remove("swap-in");
        void fresh.offsetWidth;
        fresh.classList.add("swap-in");
      }
      if (keep) {
        const again = document.getElementById(keep);
        if (again) {
          again.focus({ preventScroll: true });
          if (caret && "setSelectionRange" in again) {
            try { again.setSelectionRange(caret[0], caret[1]); } catch (e) { /* not text */ }
          }
        }
      }
      if (fresh) wire(fresh);
    });
    if (returnFocus) {
      const trigger = document.querySelector(returnFocus);
      returnFocus = null;
      if (trigger && !document.activeElement?.closest("main, header")) {
        trigger.focus({ preventScroll: true });
      }
    }
  }

  // The page's error box, for an answer that went wrong. It hides again on the next answer
  // that works, so a poll that recovers clears it.
  function showError(text) {
    const box = document.querySelector("#page-error");
    if (!box) return;
    box.hidden = false;
    box.textContent = text;
    box.dataset.fromScript = "1";
  }
  function clearError() {
    const box = document.querySelector("#page-error");
    if (box && box.dataset.fromScript) {
      box.hidden = true;
      delete box.dataset.fromScript;
    }
  }

  async function send(url, options) {
    const response = await fetch(url, {
      credentials: "same-origin",
      headers: { "X-Requested-With": "clef-tune" },
      ...options,
    });
    if (!response.ok) {
      showError(FAILED);
      return response;
    }
    clearError();
    const ended = response.headers.get("X-Run-Ended");
    if (ended) notifyEnded(ended);
    if (response.headers.get("X-Redirect")) {
      window.location.assign(response.headers.get("X-Redirect"));
      return response;
    }
    if (response.headers.get("X-Refresh")) {
      window.location.reload();
      return response;
    }
    const text = await response.text();
    if (text) swap(text);
    return response;
  }

  function formData(form, submitter) {
    const data = new FormData(form);
    if (submitter && submitter.name) data.append(submitter.name, submitter.value);
    return data;
  }

  async function post(form, submitter) {
    if (form.dataset.busy) { form.dataset.again = "1"; return; }
    form.dataset.busy = "1";
    form.setAttribute("aria-busy", "true");
    try {
      await send(form.getAttribute("action"), { method: "POST", body: formData(form, submitter) });
    } catch (error) {
      showError(OFFLINE);
    } finally {
      returnFocus = null;
      delete form.dataset.busy;
      form.removeAttribute("aria-busy");
      if (form.dataset.again) { delete form.dataset.again; post(form); }
    }
  }

  document.addEventListener("submit", (event) => {
    const form = event.target.closest("form[data-post]");
    if (!form) return;
    event.preventDefault();
    const menu = form.closest(".menu");
    closeMenus(null, true);
    // The answer may replace the menu's button; focus goes to the new one.
    if (menu && menu.id) returnFocus = `[data-menu="#${menu.id}"]`;
    post(form, event.submitter);
  });

  const timers = new WeakMap();
  function autosubmit(event) {
    const form = event.target.closest("form[data-autosubmit]");
    if (!form || !form.dataset.autosubmit.split(" ").includes(event.type)) return;
    if (event.target.matches("[data-no-autosubmit]")) return;
    clearTimeout(timers.get(form));
    const wait = event.type === "input" ? Number(form.dataset.debounce || 600) : 0;
    timers.set(form, setTimeout(() => post(form), wait));
  }
  document.addEventListener("change", autosubmit);
  document.addEventListener("input", autosubmit);

  // ---------------------------------------------------------------- polling and lazy parts
  function poll(element) {
    const every = Number(element.dataset.every || 5) * 1000;
    const tick = async () => {
      if (!document.body.contains(element)) return;
      try {
        const response = await send(element.dataset.poll, { method: "GET" });
        if (response.headers.get("X-Stop-Polling")) return;
      } catch (error) { /* try again next tick */ }
      setTimeout(tick, every);
    };
    setTimeout(tick, every);
  }

  function wire(root) {
    root.querySelectorAll("[data-poll]").forEach((element) => {
      if (element.dataset.wired) return;
      element.dataset.wired = "1";
      poll(element);
    });
    root.querySelectorAll("[data-load]").forEach((element) => {
      if (element.dataset.wired) return;
      element.dataset.wired = "1";
      send(element.dataset.load, { method: "GET" }).catch(() => {
        element.textContent = "Couldn't load this part; reload the page to try again.";
      });
    });
    root.querySelectorAll("[data-eta-seconds]").forEach((element) => {
      const when = new Date(Date.now() + Number(element.dataset.etaSeconds) * 1000);
      element.textContent = when.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    });
    root.querySelectorAll("[data-autosize]").forEach(autosize);
    root.querySelectorAll("[data-notify]").forEach((button) => {
      const on = store.get(`notify:${button.dataset.notify}`) === "1" && "Notification" in window
        && Notification.permission === "granted";
      button.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }

  // ---------------------------------------------------------------- notifications
  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-notify]");
    if (!button) return;
    const key = `notify:${button.dataset.notify}`;
    if (button.getAttribute("aria-pressed") === "true") {
      store.set(key, "0");
      button.setAttribute("aria-pressed", "false");
      return;
    }
    if (!("Notification" in window)) {
      button.querySelector(".switch-note")?.removeAttribute("hidden");
      return;
    }
    const permission = Notification.permission === "granted"
      ? "granted" : await Notification.requestPermission();
    const on = permission === "granted";
    store.set(key, on ? "1" : "0");
    button.setAttribute("aria-pressed", on ? "true" : "false");
  });

  function notifyEnded(how) {
    const button = document.querySelector("[data-notify]");
    if (!button || button.getAttribute("aria-pressed") !== "true") return;
    const name = button.dataset.name || "Your run";
    const text = how === "done" ? `${name} is ready.` : `${name} stopped.`;
    try { new Notification("clef-tune", { body: text }); } catch (e) { /* not allowed */ }
  }

  // ---------------------------------------------------------------- menus, panels and toggles
  // An open menu or panel holds the keyboard: focus moves into it, Tab stays inside it, and
  // closing it with Escape or a choice gives focus back to what opened it.
  let panelOpener = null;

  function focusables(root) {
    return [...root.querySelectorAll("a[href], button, input, select, textarea, [tabindex]")]
      .filter((el) => !el.disabled && el.tabIndex >= 0 && el.getClientRects().length > 0);
  }

  function closeMenus(except, giveBack) {
    document.querySelectorAll("[data-menu][aria-expanded='true']").forEach((button) => {
      if (button === except) return;
      button.setAttribute("aria-expanded", "false");
      const menu = document.querySelector(button.dataset.menu);
      if (menu) menu.hidden = true;
      if (giveBack) button.focus({ preventScroll: true });
    });
  }

  function closePanels() {
    const open = document.querySelectorAll(".panel-wrap.show");
    open.forEach((p) => p.classList.remove("show"));
    if (open.length && panelOpener && document.body.contains(panelOpener)) {
      panelOpener.focus({ preventScroll: true });
    }
    panelOpener = null;
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-menu]");
    if (button) {
      event.preventDefault();
      const menu = document.querySelector(button.dataset.menu);
      const open = button.getAttribute("aria-expanded") !== "true";
      closeMenus(button);
      button.setAttribute("aria-expanded", open ? "true" : "false");
      if (menu) {
        menu.hidden = !open;
        if (open) focusables(menu)[0]?.focus({ preventScroll: true });
      }
      return;
    }
    const link = event.target.closest(".menu a");
    if (!event.target.closest(".menu") || link) closeMenus(null, !!link);
    const toggle = event.target.closest("[data-toggle]");
    if (toggle) {
      const target = document.querySelector(toggle.dataset.toggle);
      if (target) target.hidden = !target.hidden;
    }
    const open = event.target.closest("[data-open-panel]");
    if (open) {
      const panel = document.querySelector(open.dataset.openPanel);
      if (panel) {
        panel.classList.add("show");
        panelOpener = open;
        focusables(panel.querySelector(".panel") || panel)[0]?.focus({ preventScroll: true });
        const url = open.dataset.panelUrl;
        if (url) send(url, { method: "GET" });
      }
    }
    if (event.target.closest("[data-close-panel]")) closePanels();
  });

  function holding() {
    const trigger = document.querySelector("[data-menu][aria-expanded='true']");
    const menu = trigger && document.querySelector(trigger.dataset.menu);
    return menu && !menu.hidden ? menu : document.querySelector(".panel-wrap.show .panel");
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Tab") {
      const box = holding();
      const items = box ? focusables(box) : [];
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      const here = document.activeElement;
      if (!box.contains(here) || (event.shiftKey ? here === first : here === last)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus({ preventScroll: true });
      }
      return;
    }
    if (event.key !== "Escape") return;
    closeMenus(null, true);
    closePanels();
    closeTips();
    if (document.activeElement && document.activeElement.closest(".info-tip")) {
      document.activeElement.blur();
    }
  });

  // ---------------------------------------------------------------- framed on Hugging Face
  // A Space runs inside huggingface.co's frame, where some browsers keep no cookies for it (so
  // nothing set up would stick). Inside a frame, the header offers the page in its own tab.
  // The frame's sandbox also keeps it from taking over the whole tab, so a link meant to
  // (sign-in) opens in its own tab there instead. In-app browsers (LinkedIn's) open no tabs
  // from a frame at all, so there the page says where to go rather than doing nothing.
  function framed() {
    try { return window.top !== window.self; } catch (e) { return true; }
  }
  // Whether this browser keeps cookies where the page is shown: Safari and every iPhone
  // browser keep none inside Hugging Face's frame.
  function cookiesKept() {
    try {
      document.cookie = "clef_tune_probe=1; Path=/; SameSite=None; Secure";
      const kept = document.cookie.includes("clef_tune_probe=1");
      document.cookie = "clef_tune_probe=; Path=/; Max-Age=0; SameSite=None; Secure";
      return kept;
    } catch (e) { return false; }
  }
  // Where it keeps none, a click takes its action to the app's own tab, where it does.
  const elsewhere = framed() && !cookiesKept();
  const IN_APP = /LinkedInApp|FBAN|FBAV|FB_IAB|Instagram/; // in-app browsers that open no tabs
  function stuck(address) {
    showError(`clef-tune can't work inside this in-app browser. Open it in your phone's browser instead: choose “Open in browser” from the menu, or copy ${address} into your browser.`);
    const box = document.querySelector("#page-error");
    if (box) box.scrollIntoView({ block: "nearest" });
  }
  // Tabs are opened here rather than by the link itself, so a refused one is noticed.
  document.addEventListener("click", (event) => {
    const link = event.target.closest("a[href]");
    if (!link || !framed() || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey) return;
    const own = link.origin === window.location.origin;
    if (link.target !== "_blank" && link.target !== "_top" && !(own && elsewhere)) return;
    event.preventDefault();
    event.stopPropagation(); // not swapped in place (data-nav), which would stick no better
    // The own-tab link follows the address, which may have moved (data-nav).
    const url = link.matches("[data-own-tab]") ? window.location.href : link.href;
    const tab = window.open(url, "_blank");
    if (!tab) stuck(own ? window.location.host : url);
    else if (!own) tab.opener = null;
  }, true);
  document.addEventListener("submit", (event) => {
    if (!elsewhere) return;
    event.stopPropagation(); // sent by the browser, not in the background (data-post)
    const name = `clef-tune-${Date.now()}`;
    if (window.open("", name)) {
      event.target.target = name;
    } else {
      event.preventDefault();
      stuck(window.location.host);
    }
  }, true);

  // ---------------------------------------------------------------- tips
  // CSS opens a tip on hover and keyboard focus. A touch screen has no hover, so there a tap
  // on its button toggles it, and a tap anywhere else closes it.
  const touch = window.matchMedia("(hover: none)");
  function closeTips(except) {
    document.querySelectorAll(".info-tip.open").forEach((tip) => {
      if (tip !== except) tip.classList.remove("open");
    });
  }
  document.addEventListener("click", (event) => {
    const trigger = event.target.closest(".info-tip > button");
    const tip = trigger ? trigger.parentElement : null;
    closeTips(tip);
    if (tip && touch.matches) tip.classList.toggle("open");
  });

  // ---------------------------------------------------------------- words that size themselves
  // data-autosize may give the fewest characters to leave room for (3 by default).
  function autosize(input) {
    const fewest = Number(input.dataset.autosize) || 3;
    input.style.width = `${Math.max(input.value.length, fewest) + 1}ch`;
  }
  document.addEventListener("input", (event) => {
    if (event.target.matches("[data-autosize]")) autosize(event.target);
    const mirror = event.target.dataset.mirror;
    if (mirror) {
      const target = document.querySelector(mirror);
      if (target) target.textContent = event.target.value.trim() || event.target.dataset.fallback || "";
    }
  });

  // ---------------------------------------------------------------- the drop zone
  document.addEventListener("dragover", (event) => {
    const zone = event.target.closest(".drop");
    if (!zone) return;
    event.preventDefault();
    zone.classList.add("over");
  });
  document.addEventListener("dragleave", (event) => {
    const zone = event.target.closest(".drop");
    if (zone) zone.classList.remove("over");
  });
  document.addEventListener("drop", (event) => {
    const zone = event.target.closest(".drop");
    if (!zone) return;
    event.preventDefault();
    zone.classList.remove("over");
    const input = zone.querySelector("input[type=file]");
    if (input && event.dataTransfer.files.length) {
      input.files = event.dataTransfer.files;
      zone.closest("form").requestSubmit();
    }
  });
  document.addEventListener("change", (event) => {
    if (event.target.matches(".drop input[type=file]") && event.target.files.length) {
      event.target.closest("form").requestSubmit();
    }
  });

  // ---------------------------------------------------------------- links that swap in place
  // A data-nav link asks for its page in the background; the answer swaps just the parts that
  // change, and the address bar follows so Back and reload still work.
  async function navigate(url, push) {
    try {
      const response = await send(url, { method: "GET" });
      if (!response.ok) throw new Error(String(response.status));
      if (push) history.pushState({ nav: true }, "", url);
    } catch (error) {
      window.location.assign(url);
    }
  }
  document.addEventListener("click", (event) => {
    const link = event.target.closest("a[data-nav]");
    if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey) return;
    event.preventDefault();
    navigate(link.href, true);
  });
  window.addEventListener("popstate", () => {
    if (document.querySelector("a[data-nav]")) navigate(window.location.href, false);
  });

  // ---------------------------------------------------------------- the phone's row pager
  document.addEventListener("click", (event) => {
    const step = event.target.closest("[data-page-row]");
    if (!step) return;
    const pager = step.closest("[data-pager]");
    const cards = [...document.querySelectorAll(pager.dataset.pager + " .kvs")];
    const current = cards.findIndex((card) => !card.hidden);
    const delta = Number(step.dataset.pageRow);
    const next = (current + delta + cards.length) % cards.length;
    cards[current].hidden = true;
    cards[next].hidden = false;
    cards[next].style.animation = "none";
    void cards[next].offsetWidth;
    cards[next].style.animation = `${delta > 0 ? "fromRight" : "fromLeft"} .45s cubic-bezier(.2, .7, .2, 1) both`;
    pager.querySelector("[data-row-label]").textContent = cards[next].dataset.label;
  });

  document.addEventListener("DOMContentLoaded", () => {
    wire(document);
    const own = document.querySelector("[data-own-tab]");
    if (own && framed()) {
      own.href = window.location.href;
      own.hidden = false;
    }
    if (elsewhere && IN_APP.test(navigator.userAgent)) stuck(window.location.host);
  });
})();
