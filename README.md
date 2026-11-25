# CampusLens AI

Üniversite yönetmelik ve yönergelerinden **kaynak gösteren** cevaplar üreten RAG (Retrieval-Augmented Generation) sistemi.
Öğrenci soruyu sorar; sistem ilgili belge parçalarını bulur (BM25, dense, hybrid, hybrid + rerank), cevabı yerel bir dil modeliyle
üretir ve her cevabı belge adı, sayfa numarası ve alıntıyla birlikte gösterir. Belgelerde yeterli bilgi yoksa cevap uydurmak yerine bunu belirtir.

| Katman | Teknoloji |
| --- | --- |
| Backend | Python, FastAPI, SQLite |
| Vektör arama | Qdrant (yerel klasör modu), `intfloat/multilingual-e5-small`, BM25, `BAAI/bge-reranker-v2-m3` |
| Dil modeli | Ollama (`qwen3:4b`) |
| Frontend | HTML, CSS, Vanilla JavaScript (build adımı yok) |

## Gereksinimler

- **Python 3.12** veya üstü
- **Ollama** (https://ollama.com/download) ve model: `ollama pull qwen3:4b`
- Embedding, sparse retrieval ve reranker modelleri uygulama başlatılmadan önce `python -m scripts.prepare_models` komutuyla hazırlanmalıdır. Çalışma sırasında Hugging Face modelleri yerel cache üzerinden yüklenir.

## Kurulum (bir kez)

Proje kökünde:

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3:4b
python -m scripts.prepare_models
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

Önce katalogdaki dokuz PDF’yi aynı kaynak URL’leriyle işleyip Qdrant’a indeksleyin. Cevapsız soruların değerlendirilmesi için seçilen LLM sağlayıcısı çalışıyor olmalıdır. Admin ekranından bir yöntem veya tüm yöntemler seçilebilir. API koşuyu arka planda işler ve sonuçları SQLite’tan sunar. Yeni bir veritabanında sonuç listesi boştur.

Aynı 50 soruluk test setini terminalden çalıştırmak için:

```bash
python -m scripts.run_evaluation --method bm25 --split test
python -m scripts.run_evaluation --method dense --split test
python -m scripts.run_evaluation --method hybrid --split test
python -m scripts.run_evaluation --method hybrid_rerank --split test
```

- **Recall@5 ve MRR@5** 42 cevaplanabilir soruda hesaplanır. Doğru kaynak için katalogdaki URL ve bir kabul edilen PDF sayfası eşleşmelidir.
- **Cevapsız soru doğruluğu** sekiz soru için gerçek RAG dönüşündeki `insufficient_context` alanıyla hesaplanır. URL ve doğru sayfa alanları bu sorularda boştur. Model hatası oluşursa hata soru bazında saklanır. Doğruluk yalnızca ölçülebilen sorulardan hesaplanır ve arayüz kaç sorunun ölçülebildiğini gösterir.
- **Ortalama arama süresi** 50 sorgunun retrieval süresini kullanır. Hybrid + Rerank süresine yeniden sıralama dahildir. LLM cevap üretme süresi bu ölçümün dışındadır.

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
| İlk soru çok uzun sürüyor | Model hazırlığının tamamlandığını ve başlangıç warm-up işleminin bittiğini kontrol edin. |
| Sorular yanıtlanmıyor | Ollama açık mı ve `ollama list` çıktısında `qwen3:4b` var mı kontrol edin. |
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


## Model hazırlığı ve GPU desteği

Model dosyalarını uygulamayı başlatmadan önce hazırlayın:

```bash
python -m scripts.prepare_models
```

İlk hazırlık internet bağlantısı gerektirebilir. Normal çalışma
sırasında embedding ve reranker modelleri yerel cache üzerinden
yüklenir; eksik model dosyaları otomatik indirilmez.

### Model readiness

- `MODEL_WARMUP_ENABLED=true`: varsayılan başlangıç davranışı.
- Modeller arka planda embedding, reranker ve Ollama sırasıyla hazırlanır.
- Hazırlık tamamlanmadan gelen soru istekleri HTTP 503 ve
  `model_preparing` hata koduyla reddedilir.
- Hazırlık başarısız olursa soru istekleri HTTP 503 ve
  `model_unavailable` hata koduyla reddedilir.
- `/api/health` mevcut backend, database ve Qdrant sağlık
  sözleşmesini korur; model readiness göstergesi değildir.
- `MODEL_WARMUP_ENABLED=false` warm-up mekanizmasını devre dışı bırakır.
  Bu durumda modeller gerektiğinde yüklenebilir.

### Reranker cihaz seçimi

`.env` dosyasındaki varsayılan ayarlar:

```dotenv
RERANKER_DEVICE=cpu
MODEL_WARMUP_ENABLED=true
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
OLLAMA_KEEP_ALIVE=30m
```

NVIDIA GPU ve CUDA destekli PyTorch bulunan sistemlerde:

```dotenv
RERANKER_DEVICE=cuda:0
```

CUDA destekli PyTorch kurulumu kullanılan işletim sistemi, Python
sürümü ve GPU sürücüsüne uygun olmalıdır. CPU kurulumu taşınabilir
varsayılan olarak korunmuştur.

CUDA açıkça seçildiğinde GPU kullanılamazsa sistem sessizce CPU'ya
geçmez. Model yükleme hatası açık şekilde bildirilir.

GPU seçimi BGE reranker içindir. Embedding cihazını veya Ollama'nın
kendi cihaz yerleşimini değiştirmez.

### Warm-up benchmark

```bash
python -m scripts.benchmark_model_warmup --repeats 5 --output data/model_warmup_benchmark.json
```

Benchmark çıktısı için mevcut olmayan bir dosya yolu seçin.
Cold ölçümler model residency durumunu ifade eder; disk cache'in
silindiği anlamına gelmez.

Deneysel sonuçlar, test kapsamı ve bilinen sınırlamalar:
[Model warm-up validation report](docs/model_warmup_validation.md).
