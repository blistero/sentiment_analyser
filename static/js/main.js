// ── Shared utilities ─────────────────────────────────────────────────────────

function formatTime(ts) {
  if (!ts) return "—";
  const d = new Date(ts);
  return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function escHtml(s) {
  return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ── Toast notification system ─────────────────────────────────────────────────
function showToast(msg, type = "info", duration = 3500) {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = msg;
  container.appendChild(toast);

  // Trigger animation
  requestAnimationFrame(() => {
    requestAnimationFrame(() => toast.classList.add("show"));
  });

  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── Sidebar toggle with overlay ───────────────────────────────────────────────
function toggleSidebar() {
  const sidebar = document.querySelector(".sidebar");
  const isOpen = sidebar.classList.toggle("open");

  let overlay = document.querySelector(".sidebar-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.className = "sidebar-overlay";
    overlay.onclick = closeSidebar;
    document.body.appendChild(overlay);
  }
  overlay.classList.toggle("visible", isOpen);
}

function closeSidebar() {
  document.querySelector(".sidebar")?.classList.remove("open");
  document.querySelector(".sidebar-overlay")?.classList.remove("visible");
}

// ── Active nav highlight ──────────────────────────────────────────────────────
document.querySelectorAll(".nav-item").forEach((item) => {
  if (item.getAttribute("href") === window.location.pathname) {
    item.classList.add("active");
  }
});

// ── Debounce helper ───────────────────────────────────────────────────────────
function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

// ── Prevent form double-submit ────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("form").forEach((form) => {
    form.addEventListener("submit", (e) => {
      const btn = form.querySelector('[type="submit"]');
      if (!btn) return;
      if (btn.dataset.submitting === "1") {
        e.preventDefault();
        return;
      }
      btn.dataset.submitting = "1";
      setTimeout(() => delete btn.dataset.submitting, 5000);
    });
  });
});
