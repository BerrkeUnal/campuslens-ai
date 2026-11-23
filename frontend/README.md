# CampusLens AI — Frontend

HTML5, CSS3 ve Vanilla JavaScript (ES modules). Framework ve build adımı yok.

## Çalıştırma

ES modülleri `file://` üzerinden çalışmaz; sayfaları bir statik sunucuyla açın.
Backend CORS ayarı varsayılan olarak `http://127.0.0.1:5500` adresine izin verir (`FRONTEND_ORIGIN`).

```bash
# 1) Backend (proje kökünde)
uvicorn backend.app.main:app --reload --port 8000

# 2) Frontend (proje kökünde)
python -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

Ardından tarayıcıda **http://127.0.0.1:5500** adresini açın.
VS Code Live Server da 5500 portunu kullanır; `localhost` yerine `127.0.0.1` ile açtığınızdan emin olun,
aksi hâlde CORS isteği reddedilir.

## Mock modu

Backend çalışmıyorsa veya servisler henüz bağlanmadıysa arayüz mock veriyle çalışabilir:

- Adrese `?mock=1` ekleyin (kapatmak için `?mock=0`), ya da
- Üst bardaki **Mock** düğmesine basın. Seçim sayfalar arasında hatırlanır.

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
| `index.html`, `js/chat.js` | Soru, yöntem seçimi, yanıt, süre, maliyet, kaynak kartları, 4 yöntem karşılaştırması |
| `documents.html`, `js/documents.js` | PDF yükleme, belge listesi, durum takibi, silme |
| `evaluation.html`, `js/evaluation.js` | Koşu başlatma, yöntem karşılaştırma tablosu, koşu geçmişi |
| `js/api.js` | Tüm fetch çağrıları, hata dönüşümü, mock anahtarı. API tabanı: `API_BASE_URL` |
| `js/mock-data.js` | Backend şemalarıyla aynı şekilde mock veriler |
| `js/ui.js` | Güvenli DOM yardımcıları, biçimlendirme, ortak üst bar |
| `css/app.css` | Tek stil dosyası; renk ve boşluklar CSS değişkenlerinde |

## Kurallar

- Kullanıcı veya API metni `innerHTML` ile basılmaz; tüm metinler `textContent` ile yazılır.
- Bağlantılar yalnızca `http(s)` protokolüyle oluşturulur.
- Frontend'de API anahtarı veya sır bulunmaz.
- İstek sürerken ilgili buton devre dışıdır; çift gönderim engellenir.
- 768 px altında tek kolon; belge tablosu mobilde kart görünümüne geçer.
