# CampusLens AI

Üniversite yönetmelik ve yönergelerinden **kaynak gösteren** cevaplar üreten RAG (Retrieval-Augmented Generation) sistemi.
Öğrenci soruyu sorar; sistem ilgili belge parçalarını bulur (BM25, dense, hybrid, hybrid + rerank), cevabı yerel bir dil modeliyle
üretir ve her cevabı belge adı, sayfa numarası ve alıntıyla birlikte gösterir. Belgelerde yeterli bilgi yoksa cevap uydurmak yerine bunu belirtir.

| Katman | Teknoloji |
| --- | --- |
| Backend | Python, FastAPI, SQLite |
| Vektör arama | Qdrant (yerel klasör modu), `intfloat/multilingual-e5-small`, BM25, `BAAI/bge-reranker-v2-m3` |
| Dil modeli | Ollama (`qwen3:8b`) |
| Frontend | HTML, CSS, Vanilla JavaScript (build adımı yok) |

## Gereksinimler

- **Python 3.12** veya üstü
- **Ollama** (https://ollama.com/download) ve model: `ollama pull qwen3:8b`
- İlk çalıştırmada embedding ve reranker modelleri internetten indirilir (yaklaşık 2,5 GB). Bu yüzden ilk soru birkaç dakika sürebilir.

## Kurulum (bir kez)

Proje kökünde:

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3:8b
```

Ayarlar `.env` dosyasından okunur. Dosya yoksa çalıştırma komutu `.env.example` dosyasını kopyalayarak oluşturur.

## Çalıştırma (tek komut)

```bash
python scripts/run_dev.py
```

Bu komut backend'i (`http://127.0.0.1:8000`) ve frontend'i (`http://127.0.0.1:5500`) birlikte başlatır, öğrenci ekranını tarayıcıda açar.
Ctrl+C ikisini birlikte durdurur. Ollama çalışmıyorsa veya model indirilmemişse başlangıçta uyarı verir.

| Adres | İçerik |
| --- | --- |
| http://127.0.0.1:5500/index.html | Öğrenci sohbet ekranı |
| http://127.0.0.1:5500/admin-login.html | Admin girişi (belge yönetimi ve değerlendirme) |
| http://127.0.0.1:8000/docs | API belgeleri (Swagger) |

Seçenekler: `--no-browser` (tarayıcıyı açma), `--reload` (Python dosyası değişince backend'i yeniden başlat).

Portlar sabittir: frontend API'yi `http://127.0.0.1:8000/api` adresinde arar, backend CORS ayarı yalnızca `http://127.0.0.1:5500` adresine izin verir.
Tarayıcıda `localhost` yerine `127.0.0.1` kullanın.

<details>
<summary>İki ayrı terminalle elle çalıştırma</summary>

```bash
# Terminal 1 — backend
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
python -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

</details>

## Belgeleri yükleme

En kolay yol admin ekranıdır: **Admin girişi → Belgeler → PDF yükle**. Belge yükleme isteği içinde metne çevrilip indekslenir;
büyük belgelerde birkaç dakika sürebilir. Admin girişi şimdilik demo amaçlıdır; demo bilgileri
`frontend/js/mock-data.js` dosyasında `DEMO_ADMIN_CREDENTIALS` olarak tanımlıdır (ayrıntı: [`frontend/README.md`](frontend/README.md)).

Komut satırından da yapılabilir (backend kapalıyken; yerel Qdrant klasörünü aynı anda tek süreç açabilir):

```bash
python -m backend.scripts.process_pdf yonetmelik.pdf --title "Ön Lisans ve Lisans Eğitim-Öğretim Yönetmeliği" --source-url "https://..."
python scripts/index_documents.py --input data/processed/<belge-id>/chunks.json
```

`process_pdf` çıktı klasörünü (`data/processed/<belge-id>/`) ekrana yazar.

Kullanılan İstinye Üniversitesi belgelerinin listesi ve kaynak adresleri: `resources/istinye_documents.json`.

## Değerlendirme

```bash
python -m scripts.run_evaluation --method hybrid_rerank   # bm25 | dense | hybrid | hybrid_rerank
```

Admin ekranındaki **Değerlendirme** sayfası yalnızca gerçek değerlendirme sonuçlarını gösterir. Henüz sonuç yoksa
"Henüz değerlendirme çalıştırılmadı" yazar; backend'in örnek (mock) olarak işaretlediği sonuçlar ekranda gösterilmez.

## Testler

```bash
python -m pytest
ruff check .
```

## Mock modu (yalnızca geliştirme)

Teslim edilen sürümde **kapalıdır**. Frontend'i backend olmadan denemek için adrese `?mock=1` eklenebilir; bu durumda her sayfanın
üstünde "Geliştirme modu (mock)" uyarısı görünür ve ekrandaki veriler örnek veridir. Ayar yalnızca o sekme için geçerlidir;
`?mock=0` veya uyarıdaki **Mock modu kapat** düğmesiyle kapanır.

## Sorun giderme

| Durum | Çözüm |
| --- | --- |
| "8000 / 5500 portu kullanımda" | Açık kalan uvicorn, Live Server veya `http.server` penceresini kapatın. |
| "Sunucuya bağlanılamadı" | Backend çalışmıyor; `python scripts/run_dev.py` çıktısındaki hatayı kontrol edin. |
| İlk soru çok uzun sürüyor | Modeller ilk kullanımda indirilip yükleniyor; sonraki sorular hızlanır. |
| Sorular yanıtlanmıyor | Ollama açık mı ve `ollama list` çıktısında `qwen3:8b` var mı kontrol edin. |
| Admin ekranında "İndeks yok" | Henüz belge indekslenmedi; Belgeler sayfasından PDF yükleyin. |
| PowerShell `Activate.ps1` çalışmıyor | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` komutunu bir kez çalıştırın. |

## Proje yapısı

```
backend/app/       FastAPI uygulaması (api, rag, services, db, schemas)
backend/scripts/   PDF işleme komutu
backend/tests/     Backend testleri
frontend/          Öğrenci ve admin arayüzü (ayrıntı: frontend/README.md)
scripts/           run_dev.py (tek komutla çalıştırma), index_documents.py, run_evaluation.py
resources/         Belge kataloğu
data/              Yerel veritabanı, Qdrant klasörü ve değerlendirme soruları (git'e girmez, cases.jsonl hariç)
```
