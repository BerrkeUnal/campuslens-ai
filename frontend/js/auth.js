// Admin oturum modülü (demo)
// Sözleşme: loginAdmin, logoutAdmin, isAdminAuthenticated, requireAdminSession,
//           redirectIfAuthenticated, resolveNextPage
//
// ⚠️ BU GERÇEK BİR KİMLİK DOĞRULAMA DEĞİLDİR.
// - Backend'de henüz login endpoint'i yoktur; bu modül yalnızca arayüz akışını gösterir.
// - Admin HTML sayfalarını gerçek saldırılara karşı korumaz. Kullanıcı tarayıcı
//   geliştirici araçlarıyla sessionStorage değerini yazarak kontrolü aşabilir.
// - Gerçek güvenlik için admin API endpoint'leri backend tarafında korunmalıdır
//   (şifre hashleme, güvenli cookie veya token, 401/403 yanıtları, logout endpoint'i).
//
// Backend auth geldiğinde yalnızca AUTH_MODE "api" yapılır ve loginWithApi() içindeki
// istek gerçek yanıta göre tamamlanır; giriş sayfasının HTML/CSS yapısı değişmez.

import { ApiError, apiRequest } from "./api.js";
import { DEMO_ADMIN_CREDENTIALS } from "./mock-data.js";

/** "demo": tarayıcı içi demo kontrolü · "api": POST /api/auth/login (henüz yok) */
export const AUTH_MODE = "demo";

export const LOGIN_PAGE = "admin-login.html";
export const DASHBOARD_PAGE = "admin.html";
export const STUDENT_PAGE = "index.html";

/** Girişten sonra dönülebilecek sayfalar (açık yönlendirmeyi önlemek için beyaz liste). */
const ADMIN_PAGES = ["admin.html", "documents.html", "evaluation.html"];

/** sessionStorage anahtarı. Değer bir güvenlik token'ı değildir; şifre veya kullanıcı bilgisi içermez. */
const SESSION_KEY = "campuslens.admin.session";
const ADMIN_STATE_PREFIX = "campuslens.admin.";

const DEMO_LOGIN_DELAY_MS = 450;
export const INVALID_CREDENTIALS_MESSAGE = "Kullanıcı adı veya şifre hatalı.";

export class AuthError extends Error {
  constructor(message) {
    super(message);
    this.name = "AuthError";
  }
}

// ---------------------------------------------------------------------------
// Oturum kaydı
// ---------------------------------------------------------------------------

function storage() {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function writeSession() {
  // Yalnızca oturumun var olduğunu gösteren basit bir işaret saklanır.
  const value = JSON.stringify({ mode: AUTH_MODE, startedAt: new Date().toISOString() });
  storage()?.setItem(SESSION_KEY, value);
}

export function isAdminAuthenticated() {
  const raw = storage()?.getItem(SESSION_KEY);
  if (!raw) return false;
  try {
    const session = JSON.parse(raw);
    return Boolean(session && typeof session.startedAt === "string");
  } catch {
    return false;
  }
}

function clearAdminState() {
  const store = storage();
  if (!store) return;
  for (const key of Object.keys(store)) {
    if (key.startsWith(ADMIN_STATE_PREFIX)) store.removeItem(key);
  }
}

// ---------------------------------------------------------------------------
// Giriş ve çıkış
// ---------------------------------------------------------------------------

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function loginWithDemo(username, password) {
  await wait(DEMO_LOGIN_DELAY_MS);
  const valid = username === DEMO_ADMIN_CREDENTIALS.username && password === DEMO_ADMIN_CREDENTIALS.password;
  if (!valid) throw new AuthError(INVALID_CREDENTIALS_MESSAGE);
}

// İleride: backend POST /api/auth/login eklendiğinde kullanılacak.
// Token/cookie yönetimi backend'in sorumluluğundadır; frontend şifreyi saklamaz.
async function loginWithApi(username, password) {
  try {
    await apiRequest("/auth/login", { method: "POST", json: { username, password } });
  } catch (error) {
    if (error instanceof ApiError && [400, 401, 403, 422].includes(error.status)) {
      throw new AuthError(INVALID_CREDENTIALS_MESSAGE);
    }
    throw error;
  }
}

/**
 * @param {{username: string, password: string}} credentials
 * @throws {AuthError} bilgiler hatalıysa (tek ve genel bir mesajla)
 */
export async function loginAdmin({ username, password }) {
  if (AUTH_MODE === "api") {
    await loginWithApi(username, password);
  } else {
    await loginWithDemo(username, password);
  }
  writeSession();
}

/** Oturumu ve admin'e ait geçici durumu temizler, öğrenci ekranına döner. Backend isteği yoktur. */
export function logoutAdmin() {
  storage()?.removeItem(SESSION_KEY);
  clearAdminState();
  window.location.replace(STUDENT_PAGE);
}

// ---------------------------------------------------------------------------
// Sayfa koruması (yalnızca arayüz akışı; güvenlik değildir)
// ---------------------------------------------------------------------------

function currentPage() {
  const name = window.location.pathname.split("/").pop();
  return name || STUDENT_PAGE;
}

export function resolveNextPage(value) {
  return ADMIN_PAGES.includes(value) ? value : DASHBOARD_PAGE;
}

function redirectToLogin() {
  const next = currentPage();
  const query = ADMIN_PAGES.includes(next) && next !== DASHBOARD_PAGE ? `?next=${encodeURIComponent(next)}` : "";
  window.location.replace(`${LOGIN_PAGE}${query}`);
}

/**
 * Admin sayfalarının başında çağrılır. Sayfa <html class="auth-pending"> ile gizli açılır;
 * oturum varsa içerik gösterilir, yoksa giriş sayfasına yönlendirilir.
 * @returns {boolean} oturum varsa true
 */
export function requireAdminSession() {
  // Geri tuşuyla önbellekten dönülürse (bfcache) kontrolü tekrar çalıştır.
  window.addEventListener("pageshow", (event) => {
    if (event.persisted && !isAdminAuthenticated()) redirectToLogin();
  });

  if (!isAdminAuthenticated()) {
    redirectToLogin();
    return false;
  }
  document.documentElement.classList.remove("auth-pending");
  return true;
}

/** Giriş sayfasında: oturum zaten açıksa formu göstermeden yönlendirir. */
export function redirectIfAuthenticated(nextPage) {
  const go = () => window.location.replace(resolveNextPage(nextPage));
  window.addEventListener("pageshow", (event) => {
    if (event.persisted && isAdminAuthenticated()) go();
  });
  if (isAdminAuthenticated()) {
    go();
    return true;
  }
  document.documentElement.classList.remove("auth-pending");
  return false;
}
