# CampusLens AI — Frontend

HTML5, CSS3 ve Vanilla JavaScript (ES modules). Framework ve build adımı yok.

## Sayfalar

| Sayfa | Kullanıcı | İçerik |
| --- | --- | --- |
| `index.html` | Öğrenci | Sohbet ekranı. Giriş gerekmez. Sol menüde **Yeni sohbet**, **Hızlı sorular** ve en altta **Admin girişi** bağlantısı; ortada mesaj akışı; sağda kaynak paneli (belge adı, sayfa, alıntı, "Belgeyi aç"). Sorular her zaman `hybrid_rerank` ile gönderilir. |
| `admin-login.html` | Admin | Demo yönetici girişi. Oturum açıksa doğrudan admin paneline yönlendirir. |
| `admin.html` | Admin | Genel bakış: belge/indeks/parça sayıları, sistem durumu, yöntem karşılaştırma grafiği, son yüklenen belgeler, hızlı işlemler. |
| `documents.html` | Admin | Belge yönetimi: sürükle-bırak PDF yükleme, arama, durum filtresi, indeksleme takibi, onaylı silme. |
| `evaluation.html` | Admin | Değerlendirme: koşu başlatma, dört yöntemin karşılaştırma tablosu, koşu geçmişi. |

Öğrenci ekranında admin sayfalarına doğrudan bağlantı yoktur; "Admin girişi" bağlantısı giriş sayfasını açar.
Sohbet geçmişi saklanmaz ("Son sohbetler" yoktur); sohbet yalnızca açık sekmede, bellekte durur.

## Çalıştırma

ES modülleri `file://` üzerinden çalışmaz; sayfaları bir statik sunucuyla açın.
Backend CORS ayarı varsayılan olarak `http://127.0.0.1:5500` adresine izin verir (`FRONTEND_ORIGIN`).

```bash
# 1) Backend (proje kökünde)
uvicorn backend.app.main:app --reload --port 8000

# 2) Frontend (proje kökünde)
python -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

- Öğrenci ekranı: http://127.0.0.1:5500/index.html
- Admin girişi: http://127.0.0.1:5500/admin-login.html

VS Code Live Server da 5500 portunu kullanır; adresin `localhost` değil `127.0.0.1` olduğundan emin olun.
API tabanı `js/api.js` içindeki `API_BASE_URL` sabitindedir: `http://127.0.0.1:8000/api`.

## Admin girişi (demo)

> **Bu yapı gerçek kimlik doğrulama değildir.**
>
> - Backend'de henüz login endpoint'i yoktur; giriş yalnızca tarayıcı içinde, demo olarak kontrol edilir.
> - Admin HTML sayfalarını gerçek saldırılara karşı korumaz. Kullanıcı tarayıcı geliştirici araçlarıyla kontrolü aşabilir.
> - Gerçek kullanım için admin API endpoint'leri backend tarafında korunmalıdır.

Akış:

1. Öğrenci ekranındaki **Admin girişi** → `admin-login.html`.
2. Başarılı girişte `admin.html` açılır. Kullanıcı bir admin sayfasını doğrudan açmışsa (ör. `documents.html`)
   giriş sayfasına `?next=documents.html` ile yönlendirilir ve girişten sonra o sayfaya döner
   (yalnızca admin sayfaları kabul edilir).
3. Oturum `sessionStorage` içinde `campuslens.admin.session` anahtarıyla tutulur: sayfa yenilenince sürer,
   sekme kapanınca biter. Bu değer bir güvenlik token'ı değildir; **şifre, kullanıcı adı veya anahtar saklanmaz.**
4. Admin sayfaları `<html class="auth-pending">` ile gizli açılır; oturum kontrolü geçince gösterilir,
   oturum yoksa giriş sayfasına yönlendirilir. Geri tuşuyla dönülürse kontrol tekrar çalışır.
5. **Çıkış yap** oturumu ve admin'e ait geçici durumu temizler, öğrenci ekranına döner (backend isteği yoktur).

Demo bilgileri `js/mock-data.js` içinde `DEMO_ADMIN_CREDENTIALS` olarak, açıkça demo verisi diye tanımlıdır ve
giriş ekranında **yalnızca mock modu açıkken** gösterilir. Gerçek bir kullanıcı adı veya şifre repoya eklenmemelidir.

Kod: `js/auth.js` (`loginAdmin`, `logoutAdmin`, `isAdminAuthenticated`, `requireAdminSession`),
`js/login.js` (form), `js/admin-shell.js` (admin üst barı).

### Backend auth geldiğinde

`js/auth.js` içinde `AUTH_MODE` değeri `"api"` yapılır; `loginAdmin()` bu durumda `POST /api/auth/login` isteği gönderir.
Giriş sayfasının HTML ve CSS yapısı değişmez. Şifre hashleme, güvenli cookie veya token, token süresi,
yetkisiz isteklerde 401/403, logout endpoint'i ve CORS/cookie ayarları backend tarafında ele alınmalıdır.

## Mock modu (geliştirme ve demo)

Backend çalışmıyorsa veya servisler henüz bağlanmadıysa arayüz mock veriyle çalışabilir:

- Herhangi bir sayfanın adresine `?mock=1` ekleyin (kapatmak için `?mock=0`), ya da
- Admin üst barındaki **Mock** düğmesine basın.

Seçim tarayıcıda hatırlanır ve öğrenci ekranı için de geçerlidir. Öğrenci ekranında düğme yoktur;
mock açıkken üst barda "Mock veri kullanılıyor" yazar. Mock belgeler ve değerlendirmeler sekme boyunca korunur.

| Girdi | Sonuç |
| --- | --- |
| Soruda `#yetersiz` | `insufficient_context: true` yanıtı (nötr durum) |
| Soruda `#hata` | 503 servis hatası (Tekrar dene düğmesiyle) |
| Belge başlığında `hata` | İndeksleme `failed` ile biter |

## Dosyalar

| Dosya | Görev |
| --- | --- |
| `index.html`, `js/chat.js` | Öğrenci sohbet ekranı |
| `admin-login.html`, `js/login.js` | Demo admin giriş ekranı |
| `admin.html`, `js/admin.js` | Admin genel bakış |
| `documents.html`, `js/documents.js` | Admin belge yönetimi |
| `evaluation.html`, `js/evaluation.js` | Admin değerlendirme ekranı |
| `js/auth.js` | Demo oturum: giriş, çıkış, sayfa koruması |
| `js/admin-shell.js` | Admin üst barı: oturum kontrolü, çıkış, mock düğmesi, sunucu durumu |
| `js/api.js` | Tüm fetch çağrıları, hata dönüşümü, mock anahtarı |
| `js/mock-data.js` | Backend şemalarıyla aynı şekilde mock veriler ve demo giriş bilgileri |
| `js/ui.js` | Güvenli DOM yardımcıları, ikonlar, biçimlendirme |
| `css/app.css` | Tek stil dosyası; renk, boşluk ve yarıçaplar CSS değişkenlerinde |
| `img/icons.svg`, `img/favicon.svg` | Arayüz ikonları ve sekme ikonu |

## Kurallar

- Kullanıcı veya API metni `innerHTML` ile basılmaz; tüm metinler `textContent` ile yazılır.
- Bağlantılar yalnızca `http(s)` protokolüyle oluşturulur.
- Frontend'de API anahtarı veya sır bulunmaz; kullanıcı adı ve şifre console'a, URL'ye veya storage'a yazılmaz.
- İstek sürerken ilgili buton devre dışıdır; çift gönderim engellenir.
- Tablet ve mobilde tek kolon; belge tablosu mobilde kart görünümüne geçer.
