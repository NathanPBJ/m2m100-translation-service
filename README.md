# M2M100 Translation Service

Service penerjemahan multibahasa lokal yang dibangun dengan FastAPI dan model
`facebook/m2m100_418M`. Project masih dalam pengembangan. Fondasi API dan
translation engine sudah tersedia, tetapi engine belum dihubungkan ke endpoint
FastAPI.

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
- Ruang penyimpanan dan RAM yang cukup jika menjalankan model asli

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

Buat file konfigurasi lokal dari contoh:

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

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Opsi `--reload` hanya digunakan saat development. Setelah aktif, service dapat
diakses di `http://localhost:8000`, Swagger UI di
`http://localhost:8000/docs`, dan ReDoc di `http://localhost:8000/redoc`.

## Translation engine

Engine menggunakan checkpoint `facebook/m2m100_418M` melalui PyTorch dan
Transformers. Model berjalan pada komputer sendiri dan tidak memanggil API
translation berbayar atau hosted inference API.

Tokenizer dan bobot model diunduh saat `load_model()` pertama kali dijalankan,
kemudian digunakan kembali dari cache lokal. Constructor dan import module
tidak mengunduh model. Source dan target language harus diberikan secara manual
dengan kode yang didukung tokenizer M2M100. Automatic language detection dan
endpoint `POST /translate` belum tersedia.

Contoh penggunaan dari Python:

```python
from app.services.translation import M2M100TranslationService

service = M2M100TranslationService()
service.load_model()
result = service.translate("Good morning", "en", "id")
print(result.translated_text)
```

## Model configuration

| Variable | Default | Keterangan |
|---|---|---|
| `MODEL_NAME` | `facebook/m2m100_418M` | Checkpoint model di Hugging Face |
| `MODEL_CACHE_DIR` | `./models` | Lokasi cache tokenizer dan bobot model |
| `MODEL_DEVICE` | `auto` | Perangkat inference: `auto`, `cpu`, atau `cuda` |
| `MODEL_LOCAL_FILES_ONLY` | `false` | Jika `true`, hanya gunakan file yang sudah ada di cache |
| `MODEL_MAX_INPUT_TOKENS` | `512` | Batas token input sebelum translation ditolak |
| `MODEL_MAX_NEW_TOKENS` | `256` | Batas token baru yang dihasilkan model |
| `MODEL_NUM_BEAMS` | `4` | Jumlah beam untuk generation |

`MODEL_DEVICE=auto` memilih CUDA jika PyTorch menyatakan CUDA tersedia dan
memilih CPU jika tidak. Nilai `cpu` selalu menggunakan CPU. Nilai `cuda`
mewajibkan CUDA; service akan menghasilkan error konfigurasi jika CUDA tidak
tersedia dan tidak akan diam-diam kembali ke CPU.

Nilai token dan jumlah beam harus lebih besar dari nol. Input yang melebihi
`MODEL_MAX_INPUT_TOKENS` ditolak tanpa silent truncation.

## Menjalankan model smoke test

Smoke test berikut benar-benar memuat model dan mungkin mengunduh file berukuran
besar pada pemanggilan pertama.

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

Script menampilkan bahasa sumber, bahasa tujuan, device, status, dan hasil
translation. Exit code bernilai non-zero jika loading atau inference gagal.

## Testing dan pemeriksaan kualitas

Unit test menggunakan fake tokenizer dan fake model. Menjalankan test biasa
tidak membutuhkan internet dan tidak mengunduh model asli:

```bash
pytest
```

Jalankan lint:

```bash
ruff check .
```

Periksa formatting:

```bash
ruff format --check .
```

Untuk menerapkan formatting secara otomatis, jalankan `ruff format .`.

## Struktur project

```text
app/
├── api/routes/health.py       # Route health check
├── core/
│   ├── config.py              # Konfigurasi aplikasi dan model
│   └── exceptions.py          # Error khusus translation engine
├── domain/translation.py      # Immutable translation result
├── services/translation.py    # Local M2M100 engine
└── main.py                    # FastAPI application
scripts/
└── smoke_test_model.py        # Pengujian manual model asli
tests/
├── test_health.py
└── test_translation_service.py
```

## Endpoint saat ini

| Method | Path | Keterangan |
|---|---|---|
| GET | `/` | Informasi dasar service |
| GET | `/health` | Status kesehatan service |
| GET | `/docs` | Dokumentasi interaktif Swagger UI |
| GET | `/redoc` | Dokumentasi ReDoc |

Belum ada endpoint HTTP untuk translation atau daftar bahasa.

## Model files

File model tidak disimpan di Git. Folder berikut diabaikan:

```text
models/
cache/
huggingface_cache/
```

File `.env`, virtual environment, log, dan cache tooling juga diabaikan.

## Status pengembangan

Selesai:

- Fondasi FastAPI
- Health endpoint
- M2M100 translation engine
- Konfigurasi CPU/CUDA
- Unit test engine tanpa download model

Belum selesai:

- FastAPI lifespan model loading
- Endpoint `POST /translate`
- Endpoint `GET /languages`
- Automatic language detection
- Batch translation
- Docker
- Performance testing

## Lisensi

Source code repository menggunakan MIT License. Bobot model diunduh terpisah
dari Hugging Face dan bukan bagian dari repository ini. Model yang digunakan
adalah `facebook/m2m100_418M`; pengguna deployment tetap harus memeriksa dan
mematuhi lisensi serta ketentuan model upstream.
