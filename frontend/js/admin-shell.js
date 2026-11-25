// Admin sayfalarının ortak üst barı: demo oturum kontrolü, çıkış, mock düğmesi, sunucu durumu.

import { logoutAdmin, requireAdminSession } from "./auth.js";
import { initMockBanner, initMockToggle, initStatus } from "./ui.js";

/**
 * Her admin sayfasının başında çağrılır.
 * @returns {boolean} oturum yoksa false (sayfa giriş ekranına yönlendirilir; başka iş yapılmamalı)
 */
export function initAdminShell() {
  if (!requireAdminSession()) return false;

  document.querySelectorAll("[data-logout]").forEach((button) => {
    button.addEventListener("click", logoutAdmin);
  });
  initMockBanner();
  initMockToggle(document.querySelector("[data-mock-toggle]"));
  initStatus(document.querySelector("[data-status]"), { hideWhenMock: true });
  return true;
}
