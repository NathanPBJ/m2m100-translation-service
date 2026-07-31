# M2M100 Translation Service

API penerjemahan multibahasa lokal yang dibangun dengan FastAPI, PyTorch, dan
model `facebook/m2m100_418M`. Model berjalan pada komputer sendiri dan tidak
memanggil hosted translation API.

Source language dan target language masih harus diberikan secara manual.
Automatic language detection belum tersedia.

## Technology stack

- Python 3.10–3.12
- FastAPI dan Uvicorn
- PyTorch
- Hugging Face Transformers
- SentencePiece
- Pydantic Settings
- Pytest dan HTTPX
- Ruff

## Prasyarat

- Python 3.10, 3.11, atau 3.12
- Git
- Ruang penyimpanan dan RAM yang cukup untuk M2M100 418M

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

Buat konfigurasi lokal pada Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Pada macOS atau Linux:

```bash
cp .env.example .env
```

Semua konfigurasi mempunyai nilai default, sehingga file `.env` tidak wajib.
File tersebut diabaikan Git.

## Menjalankan service

Development:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

`--reload` hanya untuk development dan dapat menyebabkan model dimuat ulang
ketika source code berubah.

Menjalankan service tanpa file watcher:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Dokumentasi API tersedia di:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Model startup

FastAPI memuat tokenizer dan model sekali selama application startup. Loading
dijalankan melalui threadpool agar tidak memblokir event loop. API belum siap
menerima request sampai loading berhasil.

Startup pertama dapat membutuhkan waktu lebih lama karena file model harus
diunduh. Startup berikutnya menggunakan cache lokal. Model tidak dimuat ulang
untuk setiap request, dan referensi model dilepaskan saat application shutdown.

## API endpoints

| Method | Path | Keterangan |
|---|---|---|
| GET | `/` | Informasi dasar service |
| GET | `/health` | Status aplikasi dan model |
| GET | `/languages` | Kode bahasa dari tokenizer yang sedang dimuat |
| POST | `/translate` | Menerjemahkan satu teks |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc |

### Translate request

```json
{
  "text": "Good morning",
  "source_language": "en",
  "target_language": "id"
}
```

`source_language` wajib diberikan. Nilai `auto` belum didukung. Daftar kode
bahasa yang valid dapat diperoleh melalui `GET /languages`.

### Translate response

```json
{
  "original_text": "Good morning",
  "translated_text": "Selamat pagi",
  "source_language": "en",
  "target_language": "id",
  "model_name": "facebook/m2m100_418M",
  "device": "cpu",
  "status": "translated"
}
```

Jika source dan target sama, model tidak menjalankan inference. Teks asli
dikembalikan dengan status `unchanged`.

### Supported languages

`GET /languages` mengembalikan kode bahasa yang benar-benar tersedia pada
tokenizer:

```json
{
  "model_name": "facebook/m2m100_418M",
  "count": 100,
  "languages": ["af", "ar", "en", "id", "ja"]
}
```

Nilai `count` dan daftar di atas hanya contoh format. Response aktual selalu
dibuat dari tokenizer, diurutkan, dan tidak diduplikasi.

### Health response

```json
{
  "status": "healthy",
  "service": "M2M100 Translation Service",
  "version": "0.1.0",
  "environment": "development",
  "model_loaded": true,
  "model_name": "facebook/m2m100_418M",
  "model_device": "cpu"
}
```

Health check tidak menjalankan inference.

### Error responses

Semua error API memakai envelope yang sama:

```json
{
  "error": {
    "code": "unsupported_language",
    "message": "Source language code 'xx' is not supported."
  }
}
```

Mapping status utama:

| Status | Error code |
|---|---|
| 413 | `text_too_large`, `input_too_long` |
| 422 | `request_validation_error`, `invalid_translation_input`, `unsupported_language` |
| 500 | `translation_failed`, `internal_server_error` |
| 503 | `model_not_loaded`, `model_load_failed`, `device_configuration_error` |

Response internal error tidak menyertakan traceback, path lokal, isi teks, atau
pesan mentah PyTorch/Transformers.

## Concurrency

Inference PyTorch yang blocking dijalankan di threadpool. Sebuah
`asyncio.Semaphore` membatasi inference bersamaan dalam satu process FastAPI
berdasarkan `TRANSLATION_MAX_CONCURRENCY`. Default-nya satu agar sesuai untuk
CPU. Mekanisme ini bukan rate limiter global.

Jangan menjalankan beberapa Uvicorn worker tanpa memperhitungkan bahwa setiap
process akan memuat satu copy model sendiri.

## Configuration

| Variable | Default | Keterangan |
|---|---|---|
| `APP_NAME` | `M2M100 Translation Service` | Nama aplikasi |
| `APP_VERSION` | `0.1.0` | Versi API |
| `APP_ENV` | `development` | Nama environment |
| `APP_HOST` | `0.0.0.0` | Host Uvicorn |
| `APP_PORT` | `8000` | Port Uvicorn |
| `LOG_LEVEL` | `INFO` | Level logging |
| `MODEL_NAME` | `facebook/m2m100_418M` | Checkpoint Hugging Face |
| `MODEL_CACHE_DIR` | `./models` | Cache tokenizer dan bobot |
| `MODEL_DEVICE` | `auto` | `auto`, `cpu`, atau `cuda` |
| `MODEL_LOCAL_FILES_ONLY` | `false` | Gunakan cache lokal saja jika `true` |
| `MODEL_MAX_INPUT_TOKENS` | `512` | Batas token input engine |
| `MODEL_MAX_NEW_TOKENS` | `256` | Batas token hasil generation |
| `MODEL_NUM_BEAMS` | `4` | Jumlah beam generation |
| `API_MAX_TEXT_CHARACTERS` | `10000` | Batas karakter sebelum tokenisasi |
| `TRANSLATION_MAX_CONCURRENCY` | `1` | Batas inference bersamaan per process |

`MODEL_DEVICE=auto` memilih CUDA hanya jika PyTorch menyatakan CUDA tersedia.
Nilai `cuda` gagal secara eksplisit jika CUDA tidak tersedia.

Character limit melindungi API sebelum tokenisasi. Token limit merupakan
perlindungan terpisah setelah tokenizer menghitung panjang sequence. Input tidak
dipotong secara diam-diam.

## Menjalankan engine smoke test

Engine juga dapat digunakan tanpa FastAPI. Script berikut memuat model asli dan
menerjemahkan satu teks.

Windows PowerShell:

```powershell
python scripts/smoke_test_model.py `
  --text "Good morning" `
  --source en `
  --target id
```

Bash:

```bash
python scripts/smoke_test_model.py \
  --text "Good morning" \
  --source en \
  --target id
```

## Testing dan pemeriksaan kualitas

Unit dan integration test menggunakan fake tokenizer, fake model, dan fake
translation service. Test biasa tidak mengunduh model:

```bash
pytest
ruff check .
ruff format --check .
```

## Struktur project

```text
app/
├── api/
│   ├── dependencies.py
│   ├── error_handlers.py
│   └── routes/
│       ├── health.py
│       └── translation.py
├── core/
│   ├── config.py
│   └── exceptions.py
├── domain/translation.py
├── schemas/
│   ├── error.py
│   ├── health.py
│   └── translation.py
├── services/translation.py
└── main.py
tests/
├── conftest.py
├── test_error_handlers.py
├── test_health.py
├── test_translation_api.py
└── test_translation_service.py
```

## Model files

Bobot model diunduh terpisah dan bukan bagian dari repository. Folder
`models/`, `cache/`, dan `huggingface_cache/` diabaikan Git, bersama `.env`,
`.venv`, log, dan cache tooling.

## Current limitations

- Source language harus diberikan manual
- `source_language="auto"` belum didukung
- Belum ada batch translation
- Belum ada authentication
- Belum ada rate limiting
- Belum ada Docker
- Inference CPU dapat lambat
- Satu FastAPI process memuat satu copy model
- Input dibatasi oleh character limit dan token limit

## Status pengembangan

Selesai:

- FastAPI foundation
- M2M100 engine
- Model lifecycle
- Endpoint `/translate`
- Endpoint `/languages`
- Consistent error responses
- CPU/CUDA selection
- Unit dan integration tests

Belum selesai:

- Automatic language detection
- Batch translation
- Docker
- Authentication
- Performance benchmark
- Production deployment configuration

## Lisensi

Source code repository menggunakan MIT License. Bobot model
`facebook/m2m100_418M` diunduh terpisah dari Hugging Face dan bukan bagian dari
repository. Pengguna deployment harus memeriksa serta mematuhi lisensi dan
ketentuan model upstream.
