# M2M100 Translation Service

API penerjemahan multibahasa lokal menggunakan FastAPI, PyTorch,
`facebook/m2m100_418M`, dan Lingua. Model translation dan language detector
berjalan pada komputer sendiri tanpa hosted inference, detection API, atau
translation API berbayar.

Source language dapat diberikan secara manual atau dideteksi otomatis. Input
panjang dipecah menjadi chunk yang aman berdasarkan token M2M100, diterjemahkan
secara berurutan, lalu digabung kembali.

## Technology stack

- Python 3.10–3.12
- FastAPI dan Uvicorn
- PyTorch
- Hugging Face Transformers
- SentencePiece
- `lingua-language-detector`
- Pydantic Settings
- Pytest, HTTPX, dan Ruff

## Menyiapkan project

Buat dan aktifkan virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS atau Linux:

```bash
source .venv/bin/activate
```

Install dependency:

```bash
python -m pip install -r requirements.txt
```

Buat konfigurasi lokal pada Windows:

```powershell
Copy-Item .env.example .env
```

Pada macOS atau Linux:

```bash
cp .env.example .env
```

Semua konfigurasi mempunyai nilai default. File `.env` bersifat opsional dan
diabaikan Git.

## Menjalankan service

Development:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` hanya untuk development dan dapat menyebabkan model dimuat ulang
ketika source berubah.

Menjalankan service tanpa file watcher:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Model dan detector startup

FastAPI memuat tokenizer serta model M2M100 sekali saat startup. Setelah model
siap, aplikasi mengambil daftar bahasa tokenizer dan membangun satu Lingua
detector dari intersection bahasa Lingua dan M2M100. API belum menerima request
sampai keduanya berhasil dimuat.

Loading dilakukan melalui threadpool. Model dan detector tidak dibuat ulang per
request. Detector dilepas lebih dulu saat shutdown, kemudian referensi model
dibersihkan.

Startup pertama dapat lebih lama karena file M2M100 perlu diunduh. Lingua tidak
melakukan download model tambahan dan dapat berjalan offline setelah package
terpasang.

## API endpoints

| Method | Path | Keterangan |
|---|---|---|
| GET | `/` | Informasi dasar service |
| GET | `/health` | Status model dan detector |
| GET | `/languages` | Bahasa translation dan auto-detection |
| POST | `/detect-language` | Deteksi satu bahasa dominan |
| POST | `/translate` | Translation manual atau otomatis |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc |

## Automatic language detection

Automatic detection menggunakan `lingua-language-detector==2.2.0` secara lokal.
Jika `source_language` tidak dikirim, nilainya otomatis menjadi `auto`.
`source_language="auto"` juga dapat diberikan secara eksplisit.

Detector hanya mempertimbangkan bahasa yang dapat dipetakan ke kode tokenizer
M2M100. M2M100 saat ini menyediakan 100 kode translation, sedangkan intersection
aktual dengan Lingua menghasilkan 66 kode auto-detectable.

Lingua membedakan Norwegian Bokmål (`nb`) dan Nynorsk (`nn`), sedangkan
tokenizer M2M100 menyediakan kode `no`. Kedua kode Lingua tersebut dipetakan ke
`no`; candidate API tetap diduplikasi menjadi satu kode dengan confidence
tertinggi.

### Auto translate example

Request:

```json
{
  "text": "Сегодня хорошая погода",
  "target_language": "id"
}
```

Contoh response dari model lokal:

```json
{
  "original_text": "Сегодня хорошая погода",
  "translated_text": "Cuaca yang baik hari ini",
  "source_language": "ru",
  "source_language_mode": "auto",
  "detected_language": "ru",
  "detection_confidence": 0.8865,
  "detection_confidence_margin": 0.8331,
  "target_language": "id",
  "model_name": "facebook/m2m100_418M",
  "device": "cpu",
  "status": "translated",
  "chunked": false,
  "chunk_count": 1,
  "chunk_token_limit": 400
}
```

Detection hanya menentukan source language. Confidence bukan ukuran kualitas
translation.

### Manual translate example

Request lama tetap didukung:

```json
{
  "text": "Good morning",
  "source_language": "en",
  "target_language": "id"
}
```

Response manual mempunyai `source_language_mode="manual"`, sedangkan
`detected_language`, `detection_confidence`, dan
`detection_confidence_margin` bernilai `null`.

Target language selalu wajib dan tidak boleh menggunakan `auto`.

## Long-text translation

Endpoint `/translate` menerima satu string hingga character limit API. Teks
pendek diterjemahkan sebagai satu chunk. Teks panjang dipisahkan berdasarkan
paragraf dan kalimat, lalu fallback token-aware mencari source span terbesar
yang masih muat. Bahasa tanpa spasi seperti Mandarin dan Jepang tetap dapat
dipecah pada batas karakter Unicode tanpa memotong byte UTF-8.

Default chunk limit adalah 400 token, termasuk special token tokenizer.
Hard input limit untuk setiap operasi model tetap 512 token. Tokenizer selalu
dipanggil dengan `truncation=False`; service tidak melakukan silent truncation.
Urutan source span divalidasi dengan merekonstruksi seluruh input secara identik
sebelum inference.

Separator paragraf, CRLF, blank line, serta leading atau trailing whitespace
yang dikeluarkan dari chunk disimpan dan dipasang kembali secara deterministik.
Setiap chunk diterjemahkan secara berurutan. Automatic language detection hanya
dijalankan satu kali terhadap seluruh original text, bukan per chunk.

Jika source dan target sama, original text dikembalikan tanpa tokenisasi,
chunking, atau model inference.

### Chunking process

```text
Full text
→ language detection once
→ paragraph splitting
→ sentence splitting
→ token-aware fallback
→ sequential translation
→ ordered merge
```

Response translation menambahkan metadata:

```json
{
  "chunked": true,
  "chunk_count": 5,
  "chunk_token_limit": 400
}
```

`chunked` bernilai `true` hanya jika lebih dari satu chunk diterjemahkan.
Same-language response mempunyai `chunk_count=0`.

## Detect-language endpoint

Request:

```json
{
  "text": "Сегодня хорошая погода"
}
```

Response:

```json
{
  "language": "ru",
  "confidence": 0.8865,
  "confidence_margin": 0.8331,
  "detector": "lingua",
  "status": "detected",
  "candidates": [
    {
      "language": "ru",
      "confidence": 0.8865
    },
    {
      "language": "uk",
      "confidence": 0.0535
    },
    {
      "language": "be",
      "confidence": 0.043
    }
  ]
}
```

Endpoint tidak mengembalikan atau menyimpan original text.

## Confidence dan margin

Confidence adalah nilai keyakinan Lingua terhadap kandidat teratas.
`confidence_margin` adalah selisih confidence kandidat pertama dan kedua. Jika
hanya satu candidate tersedia, margin sama dengan confidence kandidat tersebut.

Detection diterima hanya jika:

- Confidence mencapai `LANGUAGE_DETECTION_MIN_CONFIDENCE`.
- Margin mencapai `LANGUAGE_DETECTION_MIN_RELATIVE_DISTANCE`.

Jika tidak, API mengembalikan `language_detection_uncertain` dan tidak
menjalankan translation.

## Detection cleaning

Salinan teks khusus detection dibersihkan secara ringan:

- URL `http://`, `https://`, dan `www.` dihapus.
- Mention seperti `@username` dihapus.
- Simbol `#` dihapus tetapi isi hashtag dipertahankan.
- Whitespace dinormalisasi.
- Karakter alfabet dihitung menggunakan dukungan Unicode Python.

Original text tidak diubah. M2M100 menerima string asli, termasuk whitespace,
emoji, mention, hashtag, dan URL.

## Supported languages

`GET /languages` membedakan:

- `languages`: seluruh kode yang dapat diterjemahkan tokenizer M2M100.
- `auto_detectable_languages`: subset yang dapat dideteksi Lingua dan
  diterjemahkan M2M100.

Bahasa M2M100 di luar intersection tetap dapat digunakan dengan source language
manual.

## Health response

`GET /health` mencakup:

```json
{
  "status": "healthy",
  "service": "M2M100 Translation Service",
  "version": "0.1.0",
  "environment": "development",
  "model_loaded": true,
  "model_name": "facebook/m2m100_418M",
  "model_device": "cpu",
  "language_detector_loaded": true,
  "language_detector_name": "lingua",
  "auto_detectable_language_count": 66,
  "long_text_chunking_enabled": true,
  "long_text_chunk_token_limit": 400,
  "long_text_max_chunks": 64
}
```

Health check tidak menjalankan detection atau translation.

## Error responses

Semua error memakai envelope:

```json
{
  "error": {
    "code": "language_detection_uncertain",
    "message": "The source language could not be detected reliably."
  }
}
```

Mapping utama:

| Status | Error code |
|---|---|
| 413 | `text_too_large`, `input_too_long`, `too_many_chunks` |
| 422 | `request_validation_error`, `invalid_detection_input`, `language_detection_uncertain`, `unsupported_language` |
| 500 | `text_chunking_failed`, `translation_output_truncated`, `language_detection_failed`, `translation_failed`, `internal_server_error` |
| 503 | `language_detector_not_loaded`, `language_detector_load_failed`, `model_not_loaded`, `model_load_failed`, `device_configuration_error` |

Response internal error tidak menyertakan traceback, path lokal, original text,
cleaned text, atau pesan mentah dari Lingua/PyTorch.

## Concurrency

Language detection dan M2M100 inference yang synchronous dijalankan melalui
threadpool. Detection berjalan sebelum translation semaphore.

`TRANSLATION_MAX_CONCURRENCY` membatasi request M2M100 bersamaan dalam satu
process FastAPI. Semaphore diperoleh sekali untuk seluruh proses long-text
translation, bukan per chunk. Detection berjalan sebelum semaphore.

Di dalam translation service, satu `threading.RLock` melindungi mutable
`tokenizer.src_lang`, tokenisasi, seluruh urutan generation, decode, model load,
dan model unload. Lock mencegah chunk request lain menyela source-language state.
Semaphore mengatur concurrency pada level API, sedangkan lock menjaga state
internal engine. Ini bukan rate limiter global. Setiap process Uvicorn akan
memuat satu copy model sendiri.

## Configuration

| Variable | Default | Keterangan |
|---|---|---|
| `APP_NAME` | `M2M100 Translation Service` | Nama aplikasi |
| `APP_VERSION` | `0.1.0` | Versi API |
| `APP_ENV` | `development` | Environment |
| `APP_HOST` | `0.0.0.0` | Host Uvicorn |
| `APP_PORT` | `8000` | Port Uvicorn |
| `LOG_LEVEL` | `INFO` | Level logging |
| `MODEL_NAME` | `facebook/m2m100_418M` | Checkpoint model |
| `MODEL_CACHE_DIR` | `./models` | Cache tokenizer dan bobot |
| `MODEL_DEVICE` | `auto` | `auto`, `cpu`, atau `cuda` |
| `MODEL_LOCAL_FILES_ONLY` | `false` | Gunakan cache lokal saja |
| `MODEL_MAX_INPUT_TOKENS` | `512` | Batas token engine |
| `MODEL_MAX_NEW_TOKENS` | `512` | Batas generation per chunk |
| `MODEL_NUM_BEAMS` | `4` | Jumlah beam |
| `API_MAX_TEXT_CHARACTERS` | `10000` | Batas karakter sebelum detection/tokenisasi |
| `TRANSLATION_MAX_CONCURRENCY` | `1` | Inference bersamaan per process |
| `LONG_TEXT_CHUNKING_ENABLED` | `true` | Aktifkan pemecahan input panjang |
| `LONG_TEXT_CHUNK_MAX_TOKENS` | `400` | Token maksimum per chunk |
| `LONG_TEXT_MAX_CHUNKS` | `64` | Chunk maksimum per request |
| `LANGUAGE_DETECTION_MIN_CONFIDENCE` | `0.30` | Confidence minimum kandidat teratas |
| `LANGUAGE_DETECTION_MIN_RELATIVE_DISTANCE` | `0.05` | Margin minimum kandidat pertama dan kedua |
| `LANGUAGE_DETECTION_MIN_ALPHABETIC_CHARACTERS` | `3` | Minimum karakter alfabet setelah cleaning |
| `LANGUAGE_DETECTION_MAX_CANDIDATES` | `3` | Candidate maksimum dalam response |

`LONG_TEXT_CHUNK_MAX_TOKENS` tidak boleh melebihi
`MODEL_MAX_INPUT_TOKENS`. Jika chunking dimatikan, input di atas hard token limit
tetap menghasilkan `input_too_long`. Character limit melindungi API sebelum
detection. Maximum chunk count membatasi pekerjaan satu request.

## Smoke test

Deteksi offline:

```powershell
python scripts/smoke_test_detection.py `
  --text "Сегодня хорошая погода" `
  --show-candidates
```

Bash:

```bash
python scripts/smoke_test_detection.py \
  --text "Сегодня хорошая погода" \
  --show-candidates
```

Smoke test model langsung:

```bash
python scripts/smoke_test_model.py \
  --text "Good morning" \
  --source en \
  --target id
```

Smoke test long text dengan PowerShell:

```powershell
python scripts/smoke_test_long_text.py `
  --file tests/fixtures/long_text/general_opinion_en.txt `
  --source auto `
  --target id
```

Bash:

```bash
python scripts/smoke_test_long_text.py \
  --file tests/fixtures/long_text/general_opinion_en.txt \
  --source auto \
  --target id
```

Secara default script hanya menampilkan preview awal dan akhir. Tambahkan
`--show-output` untuk melihat seluruh hasil tanpa menyimpannya ke repository.

## Testing

Unit dan integration test menggunakan fake tokenizer, model, dan detector.
Menjalankan test biasa tidak mengunduh M2M100 atau melakukan network request:

```bash
pytest
ruff check .
ruff format --check .
```

## Model dan cache files

Bobot M2M100 bukan bagian repository. Folder `models/`, `cache/`,
`huggingface_cache/`, `.venv`, cache tooling, log, dan `.env` diabaikan Git.
Binary wheel Lingua juga tidak disimpan dalam repository.

## Test fixtures

Repository mempunyai tiga fixture English original:

- Postingan opini umum tentang algorithmic social media feed.
- Review tiga minggu untuk wireless headphones fiktif Aurora X1.
- Kronologi/spill project kelompok fiktif.

Chronology sepenuhnya fiktif dan hanya digunakan untuk test. Fixture bukan
salinan artikel, review, thread, atau tulisan pihak lain.

## Current limitations

- Teks sangat pendek dapat ambigu.
- Indonesian dan Malay sulit dibedakan pada beberapa kalimat pendek.
- Sebagai contoh aktual, “Selamat pagi” dapat terdeteksi sebagai `ms`.
- Mixed-language post menggunakan satu bahasa dominan.
- Belum ada segment-level mixed-language translation.
- Emoji-only, URL-only, atau mention-only tidak dapat dideteksi.
- Auto detection hanya mencakup intersection Lingua dan M2M100.
- Bahasa di luar intersection harus menggunakan source manual.
- Detection confidence bukan translation confidence.
- Setiap chunk diterjemahkan terpisah sehingga konteks lintas chunk berkurang.
- Pronoun, istilah, atau gaya dapat sedikit berbeda antar-chunk.
- Chunking mempertahankan urutan dan separator paragraf, tetapi model dapat
  mengubah spacing internal chunk.
- Mixed-language text tetap memakai satu dominant source language.
- CPU inference untuk input sangat panjang dapat berjalan lambat.
- Request dibatasi character limit dan maximum chunk count.
- Tidak ada streaming atau partial response; kegagalan satu chunk menggagalkan
  seluruh request.
- Belum ada translation cache atau background processing.
- Belum ada batch translation.
- Belum ada Docker.
- Belum ada authentication atau rate limiting.
- CPU inference dapat lambat.

## Status pengembangan

Selesai:

- FastAPI foundation
- M2M100 engine dan lifecycle
- Manual translation
- Automatic language detection
- Endpoint `/detect-language`
- Auto `/translate`
- Token-aware long-text chunking
- Consistent error handling
- Unit dan integration tests

Belum selesai:

- Mixed-language segmentation
- Batch translation
- Streaming
- Translation cache
- Background worker
- Docker
- Authentication
- Rate limiting
- Performance benchmark
- Production deployment configuration

## Lisensi

Source code repository menggunakan MIT License. Bobot
`facebook/m2m100_418M` diunduh terpisah dan bukan bagian repository.

`lingua-language-detector` dilisensikan Apache-2.0 menurut metadata package.
Pengguna deployment harus memeriksa dan mematuhi lisensi serta ketentuan semua
dependency dan model upstream.
