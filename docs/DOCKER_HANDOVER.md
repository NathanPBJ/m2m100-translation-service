# Docker handover

## Overview

Deployment ini menjalankan satu FastAPI translation API dengan model
`facebook/m2m100_418M` dan language detector Lingua secara lokal. Long-text
chunking tetap ditangani aplikasi. Tidak ada paid translation API atau layanan
eksternal untuk inference.

Image ditujukan untuk Linux container dan CPU. Satu Uvicorn worker dipakai agar
hanya ada satu copy model di RAM. Aplikasi sudah memiliki semaphore internal
untuk membatasi translation concurrency.

## Requirements

- Docker Engine atau Docker Desktop dengan Linux containers.
- Docker Compose v2 (`docker compose`).
- Koneksi internet pada first run untuk mengunduh model.
- Storage yang cukup untuk image, dependency, dan model cache.
- CPU dan RAM yang memadai untuk M2M100.
- Port host 8000, atau port lain yang masih tersedia.

Kebutuhan RAM minimum belum diukur sebagai requirement universal. CPU inference,
terutama untuk long text, dapat membutuhkan waktu cukup lama.

## Quick start

Dari root repository:

```bash
docker compose up --build -d
```

Pantau status dan log:

```bash
docker compose ps
docker compose logs -f translation-service
```

Endpoint default:

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

Pada first startup, image dibangun, dependency dipasang, lalu model diunduh ke
named volume. Container yang berstatus `running` belum tentu siap menerima
request. Tunggu sampai health status menjadi `healthy`; startup pertama diberi
healthcheck start period 15 menit.

## Routine operations

Stop service tanpa menghapus model cache:

```bash
docker compose down
```

Restart service:

```bash
docker compose restart translation-service
```

Lihat 100 baris log terakhir dan lanjutkan streaming:

```bash
docker compose logs -f --tail=100 translation-service
```

Rebuild setelah source atau dependency berubah:

```bash
docker compose up --build -d
```

Uvicorn menjadi process utama dengan satu worker dan tanpa `--reload`. Docker
`init` meneruskan signal, Uvicorn menerima SIGTERM, lalu FastAPI lifespan
melepas detector dan model. Compose memberi waktu shutdown sampai dua menit.

## Configuration override

Service dapat berjalan tanpa `.env`. Untuk override lokal, salin
`.env.example` menjadi `.env`, lalu ubah nilai yang diperlukan. Compose membaca
file itu secara otomatis. Jangan commit `.env`.

Contoh:

```text
APP_PORT=8080
MODEL_LOCAL_FILES_ONLY=true
TRANSLATION_MAX_CONCURRENCY=1
```

`APP_PORT` memilih port host; port container tetap 8000.
`MODEL_CACHE_DIR` selalu `/models` dan `MODEL_DEVICE` default ke `cpu` pada
Compose, walaupun default native application tetap `./models` dan `auto`.

## Model cache

Model dan tokenizer tidak dibangun ke image dan tidak disimpan di Git. Compose
memasang named volume `model-cache` ke `/models`. Docker memberi nama volume
dengan project prefix, sehingga nama aktual biasanya terlihat seperti
`m2m100-translation-service_model-cache`.

Lihat volume:

```bash
docker volume ls
```

Alur first run:

```text
empty named volume
→ model downloaded into /models
→ tokenizer and model loaded
→ service becomes healthy
```

Alur container recreation:

```text
docker compose down
docker compose up -d
→ named volume remains
→ model is loaded from the existing cache
```

Setelah cache terisi, `MODEL_LOCAL_FILES_ONLY=true` dapat digunakan untuk
memaksa load dari cache. Jika cache belum lengkap, startup akan gagal dan
container tidak akan menjadi healthy; aplikasi tidak memiliki fallback ke
external translation API.

Perintah berikut menghapus container tetapi mempertahankan cache:

```bash
docker compose down
```

Perintah berikut juga menghapus named volume:

```bash
docker compose down -v
```

Gunakan `down -v` hanya jika memang ingin menghapus model cache. First startup
dan download model akan terjadi lagi.

## Healthcheck

Docker menjalankan healthcheck setiap 30 detik dengan timeout 10 detik,
start period 15 menit, dan tiga retry. Probe menggunakan Python standard library
ke `http://127.0.0.1:8000/health`; probe gagal jika HTTP bukan success atau JSON
tidak menyatakan:

- `status=healthy`
- `model_loaded=true`
- `language_detector_loaded=true`

Healthcheck tidak menjalankan detection atau translation.

## Smoke test

Setelah container healthy:

```bash
python scripts/smoke_test_docker.py
```

Script memeriksa health, languages, Russian detection, manual English to
Indonesian translation, automatic Russian to Indonesian translation, dan
same-language unchanged behavior.

Long-text smoke test opsional memakai fixture opini:

```bash
python scripts/smoke_test_docker.py --include-long-text
```

Port atau timeout dapat diubah:

```bash
python scripts/smoke_test_docker.py --base-url http://127.0.0.1:8080 --timeout 600
```

## Troubleshooting

### Port 8000 already in use

Set di `.env`:

```text
APP_PORT=8080
```

Lalu jalankan kembali:

```bash
docker compose up -d
```

### Container unhealthy

```bash
docker compose ps
docker compose logs --tail=200 translation-service
```

First download atau model load dapat masih berjalan. Error download, cache
permission, kehabisan disk, atau konfigurasi model akan terlihat di server log.
Tidak ada fake healthy response atau external fallback.

### Model download failed

Periksa koneksi internet, DNS/proxy Docker, dan ruang disk. Jalankan ulang
setelah penyebabnya selesai. Jangan mengaktifkan local-only sampai cache lengkap.

### Cache volume was deleted

Container perlu mengunduh model lagi. Pantau log dan tunggu status healthy.

### Insufficient disk

Periksa pemakaian Docker:

```bash
docker system df
```

Hapus hanya resource yang memang tidak diperlukan. Jangan menghapus named
volume service jika cache masih ingin dipakai.

### API is slow

M2M100 generation di CPU adalah bagian utama latency. Long text menjalankan
beberapa inference secara berurutan. Menambah Uvicorn worker bukan solusi aman
untuk deployment ini karena setiap worker akan memuat satu copy model sendiri.
Tahap ini tidak memakai quantization atau mengganti model.

## Security and deployment boundary

Runtime memakai user `appuser` dengan UID/GID `10001:10001`. Directory `/models`
dimiliki user tersebut agar named volume kosong dapat diinisialisasi dan ditulis
tanpa root. Container tidak privileged, tidak memakai host network, dan tidak
memasang Docker socket. Build context mengabaikan `.git`, `.env`, `.venv`,
test, script, docs, local model/cache, log, wheel, serta weight file.

API saat ini belum mempunyai authentication, TLS, rate limiting, atau reverse
proxy. Karena itu deployment ini belum sepenuhnya production-hardened dan
sebaiknya tidak langsung diekspos ke internet. Tambahkan kontrol akses pada
lapisan deployment bila nanti dibutuhkan; jangan memasukkan credential ke
Dockerfile, Compose, atau Git.

Model dipisahkan dari image agar:

- source rebuild tidak membangun ulang layer model;
- image dan repository tidak membawa model weight;
- cache tetap ada antar-container;
- cache dapat dihapus secara eksplisit oleh operator.
