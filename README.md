# CampusLens AI

CampusLens AI, üniversite yönetmelik ve yönergelerinden **kaynak gösteren Türkçe cevaplar** üreten yerel bir RAG (Retrieval-Augmented Generation) uygulamasıdır.

Öğrenci sorusunu yazar; sistem ilgili belge parçalarını BM25, dense, hybrid veya hybrid + rerank yöntemlerinden biriyle bulur, cevabı yerel bir dil modeliyle üretir ve kullanılan belgeyi, PDF sayfasını ve ilgili metin parçasını gösterir. Belgelerde yeterli bilgi yoksa cevap uydurmak yerine bunu açıkça belirtir.

| Katman | Teknoloji |
| --- | --- |
| Backend | Python 3.11+, FastAPI, SQLite |
| PDF işleme | PyMuPDF, metin temizleme ve token tabanlı chunking |
| Vektör arama | Qdrant yerel klasör modu, `intfloat/multilingual-e5-small`, BM25 |
| Yeniden sıralama | `BAAI/bge-reranker-v2-m3` |
| Dil modeli | Ollama, `qwen3:8b` |
| Frontend | HTML, CSS, Vanilla JavaScript |

## Sistem akışı

1. Admin, resmî üniversite PDF belgesini başlığı ve kaynak bağlantısıyla yükler.
2. Backend PDF metnini çıkarır, temizler ve en fazla 400 tokenlık parçalara böler.
3. Belge bilgileri SQLite'a, aranabilir belge parçaları Qdrant'a kaydedilir.
4. Öğrenci sorusu BM25, dense, hybrid veya hybrid + rerank yöntemiyle aranır.
5. En ilgili kaynak parçaları yerel Ollama modeline gönderilir.
6. Model yalnızca verilen kaynaklara dayanarak Türkçe cevap ve kaynak kimlikleri üretir.
7. Kaynak doğrulayıcı, cevapta kullanılan kaynakların gerçekten getirilen parçalar arasında olduğunu kontrol eder.
8. Yeterli bilgi bulunmazsa sistem `insufficient_context=true` döndürür ve cevap uydurmaz.

## Gereksinimler

- **Python 3.11** veya üzeri
- **Ollama**: <https://ollama.com/download>
- Git
- İlk model indirmeleri için internet bağlantısı
- Önerilen bellek: `qwen3:8b` ve reranker için en az 16 GB RAM

Node.js yalnızca isteğe bağlı frontend sözdizimi kontrolü için gereklidir. Uygulamayı çalıştırmak için frontend build adımı yoktur.

## Kurulum

Proje kökünde sanal ortam oluşturun ve bağımlılıkları yükleyin:

```bash
python -m venv .venv
```

Sanal ortamı etkinleştirin:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Bağımlılıkları ve yerel dil modelini hazırlayın:

```bash
pip install -r requirements.txt
ollama pull qwen3:8b
python -m scripts.prepare_models
```

`scripts.prepare_models` embedding, sparse retrieval, reranker ve Ollama modelinin kullanılabilir olduğunu kontrol eder. İlk çalıştırma internet hızına göre uzun sürebilir. Normal çalışma sırasında Hugging Face modelleri yerel cache üzerinden yüklenir.

Ayarlar `.env` dosyasından okunur. Dosya yoksa `scripts/run_dev.py`, `.env.example` dosyasını kopyalayarak oluşturur. Yerel ayarlar ve API anahtarları Git deposuna eklenmez.

## Belgelerin hazırlanması

Projede kullanılan dokuz resmî PDF, Git deposuna eklenmez. PDF'leri yerel olarak aşağıdaki klasöre koyun:

```text
data/source_pdfs/
```

Dosya adları `resources/istinye_documents.json` içindeki `file_name` alanlarıyla eşleşmelidir.

Backend kapalıyken dokuz belgeyi tek komutla indeksleyin:

```bash
python -m scripts.index_documents \
  --catalog resources/istinye_documents.json \
  --pdf-dir data/source_pdfs
```

Bu işlem:

- PDF'yi doğrular ve metni çıkarır.
- Metni temizleyip parçalara böler.
- Belge kaydını SQLite'a yazar.
- PDF'yi `data/documents/` altında saklar.
- Chunk'ları Qdrant'a indeksler.

Aynı kaynak URL'si daha önce indekslendiyse belge atlanır. Yeniden indekslemek için komuta `--replace` ekleyin.

> Embedded Qdrant klasörünü aynı anda yalnızca bir süreç açabilir. CLI ile indeksleme veya evaluation çalıştırmadan önce backend'i kapatın.

Tek bir PDF de admin panelinden veya CLI ile yüklenebilir:

```bash
python -m scripts.index_documents \
  --pdf yonetmelik.pdf \
  --title "Ön Lisans ve Lisans Eğitim-Öğretim Yönetmeliği" \
  --source-url "https://www.universite.edu.tr/resmi-belge.pdf"
```

Kullanılan belgelerin başlıkları, dosya adları ve resmî kaynak bağlantıları `resources/istinye_documents.json` dosyasındadır.

## Çalıştırma

Backend ve frontend'i birlikte başlatmak için:

```bash
python scripts/run_dev.py
```

Bu komut:

- Backend'i `http://127.0.0.1:8000` adresinde başlatır.
- Frontend'i `http://127.0.0.1:5500` adresinde başlatır.
- Öğrenci ekranını tarayıcıda açar.
- Ollama ve model durumuyla ilgili başlangıç kontrollerini yapar.
- `Ctrl+C` ile iki süreci birlikte durdurur.

| Adres | Açıklama |
| --- | --- |
| <http://127.0.0.1:5500/index.html> | Öğrenci soru-cevap ekranı |
| <http://127.0.0.1:5500/admin-login.html> | Demo admin girişi |
| <http://127.0.0.1:8000/docs> | Swagger API belgeleri |
| <http://127.0.0.1:8000/api/health> | Backend, SQLite ve Qdrant sağlık durumu |

İsteğe bağlı çalıştırma seçenekleri:

```bash
python scripts/run_dev.py --no-browser
python scripts/run_dev.py --reload
```

Frontend backend'e `http://127.0.0.1:8000/api` üzerinden bağlanır. CORS ayarı `http://127.0.0.1:5500` adresine göre yapılandırılmıştır; tarayıcıda `localhost` yerine `127.0.0.1` kullanın.

<details>
<summary>Backend ve frontend'i ayrı terminallerde çalıştırma</summary>

```bash
# Terminal 1
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# Terminal 2
python -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

</details>

## Admin paneli

Admin paneli aşağıdaki işlemleri sağlar:

- PDF yükleme
- Yüklenen belgeleri ve indeksleme durumlarını görüntüleme
- Belge silme
- Gerçek retrieval değerlendirmesi başlatma
- Önceki evaluation sonuçlarını görüntüleme

Demo giriş bilgileri:

```text
Kullanıcı adı: admin
Şifre: campuslens-demo
```

Admin girişi yalnızca yerel sunum akışını göstermek içindir. Gerçek backend authentication veya kullanıcı yetkilendirmesi sağlamaz.

## API endpointleri

| Yöntem | Endpoint | Açıklama |
| --- | --- | --- |
| `GET` | `/api/health` | Backend, SQLite ve Qdrant durumunu döndürür. |
| `POST` | `/api/documents/upload` | PDF'yi kaydeder, işler ve indeksler. |
| `GET` | `/api/documents` | Belge listesini döndürür. |
| `DELETE` | `/api/documents/{document_id}` | Belgeyi, dosyasını ve vektör kayıtlarını siler. |
| `POST` | `/api/questions/ask` | Soruyu seçilen retrieval yöntemiyle yanıtlar. |
| `POST` | `/api/evaluations/run` | Gerçek evaluation koşusu başlatır. |
| `GET` | `/api/evaluations/results` | SQLite'a kaydedilmiş koşuları döndürür. |

Tüm hata cevapları ortak JSON yapısını, Türkçe kullanıcı mesajını ve takip için `request_id` alanını kullanır.

## Arama yöntemleri

| Yöntem | Açıklama |
| --- | --- |
| `bm25` | Anahtar kelime ve terim eşleşmesine dayalı sparse retrieval |
| `dense` | Çok dilli embedding benzerliğine dayalı dense retrieval |
| `hybrid` | BM25 ve dense sonuçlarının birleştirilmesi |
| `hybrid_rerank` | Hybrid adayların BGE cross-encoder ile yeniden sıralanması |

Öğrenci ekranı varsayılan olarak `hybrid_rerank` kullanır.

## Değerlendirme

Evaluation veri seti `data/evaluation/cases.jsonl` dosyasındadır:

- 42 cevaplanabilir soru
- 8 kaynaklarda cevabı bulunmayan soru
- Toplam 50 soru
- 9 resmî PDF'ye dağıtılmış test örnekleri

Admin ekranından tek bir yöntem veya dört yöntemin tamamı çalıştırılabilir. Koşular arka planda yürütülür ve sonuçlar SQLite'a kaydedilir. Yeni bir veritabanında sonuç ekranının boş olması normaldir.

Terminalden çalıştırmak için backend'i kapatın ve aşağıdaki komutlardan birini kullanın:

```bash
python -m scripts.run_evaluation --method bm25 --split test
python -m scripts.run_evaluation --method dense --split test
python -m scripts.run_evaluation --method hybrid --split test
python -m scripts.run_evaluation --method hybrid_rerank --split test
```

Gerçek olarak hesaplanan metrikler:

- **Recall@5:** Doğru belge ve kabul edilen PDF sayfasının ilk beş sonuçta bulunma oranı.
- **MRR@5:** Doğru kaynağın sonuç sırasını dikkate alan reciprocal-rank ortalaması.
- **Cevapsız soru doğruluğu:** Sekiz cevapsız soruda `insufficient_context=true` döndürme başarısı.
- **Ortalama arama süresi:** Retrieval süresi; hybrid + rerank için yeniden sıralama süresi dahildir.

`answer_correctness`, `hallucination_rate` ve `total_cost_usd` alanları API sözleşmesinde gelecekteki geliştirmeler için bulunur ancak mevcut sürümde otomatik olarak hesaplanmaz. Rapor ve sunumda yalnızca gerçekten ölçülen metrikler kullanılmalıdır.

Evaluation özellikle CPU üzerinde uzun sürebilir. Sunumdan önce koşuları tamamlayıp sonuçları SQLite'a kaydetmek önerilir.

## Model hazırlığı ve readiness

Model dosyalarını uygulamayı başlatmadan önce hazırlayın:

```bash
python -m scripts.prepare_models
```

Varsayılan `.env` ayarları:

```dotenv
RERANKER_DEVICE=cpu
MODEL_WARMUP_ENABLED=true
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:8b
OLLAMA_KEEP_ALIVE=30m
```

Uygulama başlatıldığında modeller arka planda hazırlanır:

- Hazırlık sürerken soru ve evaluation istekleri HTTP 503 ve `model_preparing` kodu alır.
- Hazırlık başarısız olursa HTTP 503 ve `model_unavailable` kodu döner.
- `/api/health` backend, SQLite ve Qdrant sağlığını gösterir; model readiness endpointi değildir.
- `MODEL_WARMUP_ENABLED=false` arka plan warm-up işlemini kapatır.

NVIDIA GPU ve CUDA uyumlu PyTorch bulunan sistemlerde reranker için:

```dotenv
RERANKER_DEVICE=cuda:0
```

CUDA açıkça seçildiğinde GPU kullanılamazsa sistem sessizce CPU'ya geçmez; model hazırlama hatası verir. Bu ayar yalnızca BGE reranker içindir, Ollama'nın cihaz seçimini değiştirmez.

Warm-up benchmark komutu:

```bash
python -m scripts.benchmark_model_warmup \
  --repeats 5 \
  --output data/model_warmup_benchmark.json
```

Ayrıntılar: [Model warm-up validation report](docs/model_warmup_validation.md).

## Test ve kod kalitesi

Sanal ortam aktifken proje kökünde:

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Bazı ağır model ve haricî servis entegrasyon testleri gerekli ortam bulunmadığında `skipped` olabilir. Önemli olan test çıktısında `failed` veya `error` bulunmamasıdır.

Node.js kuruluysa frontend JavaScript sözdizimini kontrol edin:

```bash
for file in frontend/js/*.js; do node --check "$file" || exit 1; done
```

## Mock modu

Mock modu yalnızca frontend geliştirme ve gösterim amaçlıdır. Teslim sürümünde varsayılan olarak kapalıdır.

Mock modu açmak için frontend adresine `?mock=1` eklenebilir. Bu durumda sayfada geliştirme modu uyarısı gösterilir. Mock evaluation sabit veya uydurma metrik üretmez; gerçek sonuçlar yalnızca backend evaluation akışından gelir.

Mock modu `?mock=0` veya üst bardaki **Mock modu kapat** düğmesiyle kapatılabilir. Ayar yalnızca mevcut tarayıcı sekmesinde tutulur.

## Teslim öncesi kontrol listesi

- [ ] `ollama list` çıktısında `qwen3:8b` görünüyor.
- [ ] `python -m scripts.prepare_models` başarıyla tamamlanıyor.
- [ ] Dokuz resmî PDF `data/source_pdfs/` klasöründe bulunuyor.
- [ ] Dokuz belge katalog üzerinden indekslenmiş durumda.
- [ ] Admin belge ekranında belgelerin durumu `indexed`.
- [ ] `/api/health` backend, SQLite ve Qdrant için kullanılabilir durum gösteriyor.
- [ ] Kaynakta cevabı bulunan sorular belge ve sayfa numarasıyla yanıtlanıyor.
- [ ] Kaynaklarda cevabı olmayan sorularda `insufficient_context=true` dönüyor.
- [ ] PDF yükleme ve silme işlemleri çalışıyor.
- [ ] En az bir gerçek evaluation koşusu tamamlanmış ve SQLite'a kaydedilmiş durumda.
- [ ] Mock modu kapalı.
- [ ] Pytest, Ruff ve frontend sözdizimi kontrolleri hatasız tamamlanıyor.

## Bilinen sınırlamalar

- Admin girişi demo amaçlıdır; gerçek kullanıcı doğrulaması sağlamaz.
- SQLite ve embedded Qdrant tek bilgisayarlı yerel kullanım için seçilmiştir.
- Aynı embedded Qdrant klasörü backend ve CLI tarafından eş zamanlı açılamaz.
- PDF'ler, model ağırlıkları, SQLite veritabanı ve Qdrant indeksleri Git deposunda saklanmaz.
- CPU üzerinde `qwen3:8b` ve BGE reranker özellikle ilk istekte yavaş çalışabilir.
- Sistem metin tabanlı PDF'leri işler; taranmış görüntü PDF'leri için OCR desteği bulunmaz.
- Aynı PDF admin panelinden farklı kayıt bilgileriyle birden fazla kez yüklenebilir.

## Sorun giderme

| Durum | Çözüm |
| --- | --- |
| 8000 veya 5500 portu kullanımda | Açık kalan uvicorn, Live Server veya `http.server` sürecini kapatın. |
| Sunucuya bağlanılamıyor | `python scripts/run_dev.py` terminal çıktısını kontrol edin. |
| İlk soru uzun sürüyor | Model warm-up işleminin tamamlanmasını bekleyin ve bilinen bir soruyla modeli önceden ısıtın. |
| Sorular yanıtlanmıyor | Ollama'nın çalıştığını ve `ollama list` çıktısında `qwen3:8b` bulunduğunu kontrol edin. |
| `model_preparing` hatası | Modeller hâlâ hazırlanıyor; kısa süre sonra yeniden deneyin. |
| `model_unavailable` hatası | `python -m scripts.prepare_models` çalıştırıp terminaldeki asıl model hatasını kontrol edin. |
| Admin ekranında indeks yok | Belgeleri admin panelinden yükleyin veya katalog indeksleme komutunu çalıştırın. |
| Qdrant klasörü kullanımda | Backend veya başka indeksleme/evaluation sürecini kapatın. |
| PowerShell `Activate.ps1` çalışmıyor | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` komutunu bir kez çalıştırın. |

## Proje yapısı

```text
backend/app/api/       FastAPI route'ları
backend/app/core/      Ayarlar, middleware ve hata yönetimi
backend/app/db/        SQLite veritabanı ve repository'ler
backend/app/rag/       Retrieval, reranking, LLM ve kaynak doğrulama
backend/app/services/  Belge işleme, soru-cevap ve evaluation servisleri
backend/app/schemas/   API ve servis veri modelleri
backend/tests/         Birim ve entegrasyon testleri
frontend/              Öğrenci ve admin HTML/CSS/JavaScript arayüzleri
scripts/               Çalıştırma, indeksleme, evaluation ve model hazırlama CLI'ları
resources/             Dokuz resmî belgenin kataloğu
data/evaluation/       Evaluation soru seti
data/source_pdfs/      Yerel kaynak PDF'ler; Git'e eklenmez
data/documents/        Uygulama tarafından saklanan PDF'ler; Git'e eklenmez
data/qdrant_local/     Yerel Qdrant indeksi; Git'e eklenmez
docs/                  Teknik doğrulama ve benchmark notları
```

## Kapanış

CampusLens AI eğitim amaçlı bir prototiptir. Üretilen cevaplar yalnızca yüklenen resmî belgelere dayanır. Kesin ve güncel idari işlemler için ilgili üniversite birimine ve belgenin resmî kaynağına başvurulmalıdır.
