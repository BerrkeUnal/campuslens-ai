// Admin giriş ekranı (admin-login.html)
// Demo giriş akışı: gerçek kimlik doğrulama DEĞİLDİR (ayrıntı: js/auth.js).
// Kural: kullanıcı adı ve şifre hiçbir zaman console'a, URL'ye veya storage'a yazılmaz.

import { MOCK_MODE } from "./api.js";
import { AuthError, loginAdmin, redirectIfAuthenticated, resolveNextPage } from "./auth.js";
import { DEMO_ADMIN_CREDENTIALS } from "./mock-data.js";
import { icon, renderError, setButtonBusy } from "./ui.js";

const dom = {};
let submitting = false;

function setFieldError(input, node, message) {
  input.setAttribute("aria-invalid", String(Boolean(message)));
  node.hidden = !message;
  node.replaceChildren(...(message ? [icon("alert", "icon--sm"), message] : []));
}

function setGeneralError(message) {
  dom.error.hidden = !message;
  if (message) renderError(dom.error, { message }, "Giriş yapılamadı");
}

function togglePassword() {
  const show = dom.password.type === "password";
  dom.password.type = show ? "text" : "password";
  dom.toggle.setAttribute("aria-pressed", String(show));
  dom.toggle.setAttribute("aria-label", show ? "Şifreyi gizle" : "Şifreyi göster");
  dom.toggle.replaceChildren(icon(show ? "eye-off" : "eye"));
}

function validate() {
  const username = dom.username.value.trim();
  const password = dom.password.value; // şifre kırpılmaz
  setFieldError(dom.username, dom.usernameError, username ? "" : "Kullanıcı adını girin.");
  setFieldError(dom.password, dom.passwordError, password ? "" : "Şifrenizi girin.");

  if (!username) {
    dom.username.focus();
    return null;
  }
  if (!password) {
    dom.password.focus();
    return null;
  }
  return { username, password };
}

async function submitLogin(event, nextPage) {
  event.preventDefault();
  if (submitting) return;

  setGeneralError("");
  const credentials = validate();
  if (!credentials) return;

  submitting = true;
  setButtonBusy(dom.submit, true, "Giriş yapılıyor…");
  try {
    await loginAdmin(credentials);
    window.location.replace(resolveNextPage(nextPage));
    return; // yönlendirme sürerken buton pasif kalsın
  } catch (error) {
    setGeneralError(
      error instanceof AuthError ? error.message : "Giriş şu anda yapılamıyor. Lütfen daha sonra tekrar deneyin.",
    );
    dom.password.value = "";
    dom.password.focus();
  }
  submitting = false;
  setButtonBusy(dom.submit, false);
}

function init() {
  const nextPage = new URLSearchParams(window.location.search).get("next");
  if (redirectIfAuthenticated(nextPage)) return;

  const byId = (id) => document.getElementById(id);
  Object.assign(dom, {
    form: byId("login-form"),
    username: byId("username"),
    usernameError: byId("username-error"),
    password: byId("password"),
    passwordError: byId("password-error"),
    toggle: byId("password-toggle"),
    submit: byId("login-submit"),
    error: byId("login-error"),
  });

  // Demo bilgileri yalnızca mock modu açıkken gösterilir.
  if (MOCK_MODE) {
    byId("demo-info").hidden = false;
    byId("demo-username").textContent = DEMO_ADMIN_CREDENTIALS.username;
    byId("demo-password").textContent = DEMO_ADMIN_CREDENTIALS.password;
  }

  dom.toggle.addEventListener("click", togglePassword);
  dom.form.addEventListener("submit", (event) => submitLogin(event, nextPage));
  dom.username.addEventListener("input", () => setFieldError(dom.username, dom.usernameError, ""));
  dom.password.addEventListener("input", () => setFieldError(dom.password, dom.passwordError, ""));

  dom.username.focus();
}

init();
