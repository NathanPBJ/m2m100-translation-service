# M2M100 Translation Service

API penerjemahan multibahasa lokal yang dibangun dengan FastAPI dan nantinya
akan menggunakan M2M100. Project ini masih berada pada tahap awal: fondasi API,
konfigurasi, pemeriksaan kesehatan, dan tooling pengembangan sudah tersedia,
sedangkan mesin penerjemahan belum diimplementasikan.

## Technology stack

- Python 3.10–3.12
- FastAPI dan Uvicorn
- Pydantic Settings
- Pytest dan HTTPX
- Ruff

## Prasyarat

- Python 3.10, 3.11, atau 3.12
- Git

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

Buat file konfigurasi lokal dari contoh yang tersedia:

```bash
cp .env.example .env
```

Pada Windows PowerShell, gunakan:

```powershell
Copy-Item .env.example .env
```

Semua konfigurasi memiliki nilai default, sehingga file `.env` tidak wajib
untuk menjalankan aplikasi.

## Menjalankan service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Opsi `--reload` hanya ditujukan untuk development. Setelah aktif, service dapat
diakses di `http://localhost:8000`, Swagger UI di
`http://localhost:8000/docs`, dan ReDoc di `http://localhost:8000/redoc`.

## Pemeriksaan kualitas

Jalankan unit test:

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
├── api/routes/health.py  # Route health check
├── core/config.py        # Konfigurasi dari environment
├── schemas/health.py     # Response model health
└── main.py               # FastAPI application
tests/
└── test_health.py        # Test endpoint dasar
```

## Endpoint saat ini

| Method | Path      | Keterangan                         |
|--------|-----------|------------------------------------|
| GET    | `/`       | Informasi dasar service            |
| GET    | `/health` | Status kesehatan service           |
| GET    | `/docs`   | Dokumentasi interaktif Swagger UI  |
| GET    | `/redoc`  | Dokumentasi ReDoc                  |

## Rencana pengembangan berikutnya

- Integrasi M2M100 418M
- Deteksi bahasa otomatis
- Endpoint translation
- Daftar bahasa yang didukung
- Docker support
- Performance testing

