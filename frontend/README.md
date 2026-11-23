# CampusLens AI — Frontend

HTML5, CSS3 ve Vanilla JavaScript (ES modules). Framework ve build adımı yok.

## Sayfalar

| Sayfa | Kullanıcı | İçerik |
| --- | --- | --- |
| `index.html` | Öğrenci | Soru-cevap ekranı. Soru sorar; yanıtı, kaynakları ve yetersiz bağlam uyarısını görür. Sorular her zaman `hybrid_rerank` yöntemiyle gönderilir. |
| `admin.html` | Admin | Admin ana ekranı. Belge yönetimi, değerlendirme sonuçları ve öğrenci ekranına bağlantılar; belge ve son koşu özeti. |
| `documents.html` | Admin | Belge yönetimi: PDF yükleme, belge listesi, indeksleme durumu, silme. |
| `evaluation.html` | Admin | Değerlendirme: koşu başlatma, dört yöntemin karşılaştırma tablosu, koşu geçmişi. |

Öğrenci ekranında admin sayfalarına bağlantı, yöntem seçimi ve mock düğmesi yoktur.
Admin paneline doğrudan `admin.html` adresinden girilir.

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
- Admin paneli: http://127.0.0.1:5500/admin.html

VS Code Live Server da 5500 portunu kullanır; adresin `localhost` değil `127.0.0.1` olduğundan emin olun,
aksi hâlde CORS isteği reddedilir.

API tabanı `js/api.js` içindeki `API_BASE_URL` sabitindedir: `http://127.0.0.1:8000/api`.

## Mock modu (geliştirme ve demo)

Backend çalışmıyorsa veya servisler henüz bağlanmadıysa arayüz mock veriyle çalışabilir:

- Herhangi bir sayfanın adresine `?mock=1` ekleyin (kapatmak için `?mock=0`), ya da
- Admin sayfalarındaki **Mock** düğmesine basın.

Seçim tarayıcıda hatırlanır ve öğrenci ekranı için de geçerlidir. Öğrenci ekranında düğme yoktur;
mock açıkken üst barda "Mock veri kullanılıyor" yazar.

Mock modunda demo kısayolları:

| Girdi | Sonuç |
| --- | --- |
| Soruda `#yetersiz` | `insufficient_context: true` yanıtı (nötr durum) |
| Soruda `#hata` | 503 servis hatası |
| Belge başlığında `hata` | İndeksleme `failed` ile biter |

Mock belgeler yüklendikten sonra yaklaşık 2 sn `uploaded`, 4 sn `processing`, sonra `indexed` olur.

## Dosyalar

| Dosya | Görev |
| --- | --- |
| `index.html`, `js/chat.js` | Öğrenci soru-cevap ekranı |
| `admin.html`, `js/admin.js` | Admin ana ekranı ve özetler |
| `documents.html`, `js/documents.js` | Admin belge yönetimi |
| `evaluation.html`, `js/evaluation.js` | Admin değerlendirme ekranı |
| `js/api.js` | Tüm fetch çağrıları, hata dönüşümü, mock anahtarı |
| `js/mock-data.js` | Backend şemalarıyla aynı şekilde mock veriler |
| `js/ui.js` | Güvenli DOM yardımcıları, biçimlendirme, ortak üst bar |
| `css/app.css` | Tek stil dosyası; renk ve boşluklar CSS değişkenlerinde |

## Kurallar

- Kullanıcı veya API metni `innerHTML` ile basılmaz; tüm metinler `textContent` ile yazılır.
- Bağlantılar yalnızca `http(s)` protokolüyle oluşturulur.
- Frontend'de API anahtarı veya sır bulunmaz.
- İstek sürerken ilgili buton devre dışıdır; çift gönderim engellenir.
- 768 px altında tek kolon; belge tablosu mobilde kart görünümüne geçer.

> Not: Admin sayfaları şu an yalnızca ayrı adreste durur; giriş/yetkilendirme yoktur.
> Gerçek erişim kısıtı backend tarafında kimlik doğrulama eklendiğinde sağlanmalıdır.
