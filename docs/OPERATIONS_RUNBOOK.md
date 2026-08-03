# Operations runbook

Runbook ini mencakup operasi rutin untuk deployment Docker Compose CPU-only.
Jalankan command dari root repository.

## Quick start

```bash
docker compose up --build -d
```

Periksa status sampai health menjadi `healthy`:

```bash
docker compose ps
```

Main URLs:

```text
Health:  http://127.0.0.1:8000/health
Swagger: http://127.0.0.1:8000/docs
```

## Routine operations

Log 100 baris terakhir dan lanjutkan streaming:

```bash
docker compose logs -f --tail=100 translation-service
```

Stop tanpa menghapus model cache:

```bash
docker compose down
```

Restart process dengan image dan konfigurasi yang ada:

```bash
docker compose restart translation-service
```

Rebuild setelah source atau dependency berubah:

```bash
docker compose up --build -d
```

Pada shutdown, Docker init meneruskan SIGTERM, FastAPI melepas detector lalu
model, dan Uvicorn berhenti. `stop_grace_period` adalah dua menit.

## First startup and cached startup

Build image dapat memakan waktu karena dependency CPU PyTorch dan package lain
diunduh. Ketika named volume kosong, Hugging Face juga mengunduh model saat
application startup. Healthcheck memiliki start period 15 menit. Status
container `running` belum berarti siap; gunakan `docker compose ps` dan tunggu
`healthy`.

Pada mesin pengembangan, first model startup membutuhkan sekitar enam menit
sampai healthy. Setelah model cache tersedia, model loading berada pada kisaran
beberapa detik sampai belasan detik. Ini hanya contoh hasil development, bukan
jaminan: CPU, RAM, disk, network, Docker cache, dan load host dapat mengubah
waktu secara signifikan.

## Configuration

`.env` tidak wajib. Untuk override pada Windows Command Prompt:

```bat
copy .env.example .env
```

PowerShell equivalent:

```powershell
Copy-Item .env.example .env
```

Bash:

```bash
cp .env.example .env
```

Jangan commit `.env`. Parameter utama:

| Variable | Default Compose | Dampak |
|---|---:|---|
| `APP_PORT` | `8000` | Port host; port container tetap 8000 |
| `MODEL_LOCAL_FILES_ONLY` | `false` | Larang download dan hanya gunakan cache |
| `MODEL_MAX_INPUT_TOKENS` | `512` | Hard input limit engine |
| `MODEL_MAX_NEW_TOKENS` | `512` | Batas generation per chunk |
| `TRANSLATION_MAX_CONCURRENCY` | `1` | Inference request bersamaan per process |
| `LONG_TEXT_CHUNK_MAX_TOKENS` | `400` | Source token budget per chunk |
| `LONG_TEXT_MAX_CHUNKS` | `64` | Batas pekerjaan per request |
| `LANGUAGE_DETECTION_MIN_CONFIDENCE` | `0.30` | Confidence minimum |
| `LANGUAGE_DETECTION_MIN_RELATIVE_DISTANCE` | `0.05` | Margin minimum kandidat |
| `LANGUAGE_DETECTION_MIN_ALPHABETIC_CHARACTERS` | `3` | Konten minimum setelah cleaning |
| `LANGUAGE_DETECTION_MAX_CANDIDATES` | `3` | Candidate response maksimum |

`LONG_TEXT_CHUNK_MAX_TOKENS` tidak boleh melebihi
`MODEL_MAX_INPUT_TOKENS`. Compose menetapkan `MODEL_DEVICE=cpu` dan
`MODEL_CACHE_DIR=/models`. Pertahankan satu worker.

## Model cache

Compose memasang named volume `model-cache` ke `/models`. Nama aktual biasanya
mempunyai Compose project prefix. `docker compose down` mempertahankan volume,
sehingga container recreation dapat memakai cache yang sama.

`MODEL_LOCAL_FILES_ONLY=true` hanya dapat digunakan setelah seluruh checkpoint
dan tokenizer tersedia di cache. Jika cache kosong atau tidak lengkap, startup
akan gagal dan container tidak menjadi healthy.

> **Warning — destructive cache cleanup:** command berikut menghapus named
> model volume. Model harus diunduh kembali dan local-only startup akan gagal
> sampai cache diisi lagi.

```bash
docker compose down -v
```

Untuk stop biasa, selalu gunakan `docker compose down` tanpa `-v`.

## Updating source

1. Pastikan tidak ada perubahan lokal yang belum disimpan, lalu pull branch
   `main`.
2. Periksa commit dan perubahan konfigurasi/documentation.
3. Jalankan `docker compose up --build -d`.
4. Pantau `docker compose ps` dan log sampai healthy.
5. Jalankan smoke test pendek.

```bash
python scripts/smoke_test_docker.py
```

Optional long-text validation:

```bash
python scripts/smoke_test_docker.py --include-long-text
```

Long smoke test menggunakan CPU cukup lama; tidak perlu dijalankan pada setiap
restart rutin jika source/model/configuration tidak berubah.

## Backup and restore

Source of truth adalah private GitHub repository. Model cache bersifat
re-creatable dan dapat diunduh ulang dari upstream; karena itu tidak ada backup
application state khusus.

Backup Docker volume hanya opsional bila waktu download atau akses network
menjadi kendala. Prosedurnya harus mengikuti kebijakan backup Docker host dan
divalidasi oleh operator. Jangan menyalin isi volume saat container aktif, dan
jangan menjalankan cleanup volume sebelum backup diverifikasi.

## Troubleshooting

### Docker daemon tidak berjalan

`docker version` hanya menampilkan client atau gagal membuka named pipe/socket.
Mulai Docker Desktop/Engine, tunggu daemon siap, lalu ulangi
`docker compose config` dan `docker compose up -d`.

### Port conflict

Jika host port 8000 dipakai process lain, set `APP_PORT` di `.env`, misalnya
`APP_PORT=8080`, lalu gunakan URL dengan port tersebut.

### Container terus restarting

Jalankan `docker compose ps` dan `docker compose logs --tail=200
translation-service`. Penyebab umum adalah model/detector load error,
konfigurasi invalid, cache tidak lengkap dalam local-only mode, permission, atau
resource host.

### Container unhealthy atau masih dalam start period

Healthcheck menunggu model dan detector. First download dapat berlangsung lama.
Periksa log untuk progress/error dan bedakan status `starting` dari `unhealthy`.
Jangan restart berulang saat download normal masih berjalan.

### Permission `/models`

Container berjalan sebagai UID/GID `10001:10001`. Named volume baru disiapkan
oleh image. Jika volume berasal dari prosedur restore/manual, pastikan user
tersebut dapat membaca dan menulis. Hindari mengganti ke root sebagai solusi
permanen.

### Model download gagal atau cache hilang

Periksa DNS, proxy, firewall, akses Hugging Face, disk, dan log. Dengan
`MODEL_LOCAL_FILES_ONLY=false`, restart setelah koneksi pulih dapat melanjutkan
pengisian cache. Cache yang dihapus harus di-download ulang.

### Local-only startup gagal

Ubah sementara `MODEL_LOCAL_FILES_ONLY=false`, isi cache melalui startup dengan
network yang diizinkan, pastikan healthy, lalu aktifkan kembali local-only mode.

### Disk penuh

Periksa kapasitas Docker storage sebelum build/download. Jangan langsung
menghapus named volume model. Identifikasi image/build cache yang memang tidak
dipakai sesuai prosedur host; cleanup Docker bersifat destructive.

### API lambat atau request sangat panjang

M2M100 CPU generation adalah bottleneck. Long text menjalankan inference per
chunk secara sequential. Periksa `chunk_count`, panjang input, host CPU/RAM,
concurrent request, dan log duration. HTTP 413 berarti character, token, atau
chunk limit dilampaui; pecah workload di caller tanpa mengubah urutan data.

### Bahasa gagal dideteksi

Emoji-only, URL-only, mention-only, teks pendek, dan teks ambigu dapat ditolak.
Berikan `source_language` manual jika caller sudah mengetahui bahasanya.

### Indonesian terdeteksi sebagai Malay

Indonesian/Malay sangat berdekatan; “Selamat pagi” adalah contoh yang dapat
terdeteksi sebagai `ms`. Gunakan `source_language="id"` untuk konten Indonesia
yang sudah diketahui. Jangan menurunkan threshold tanpa evaluasi false positive.

### Console Windows bermasalah dengan Unicode

Gunakan PowerShell modern/Windows Terminal dan request JSON UTF-8. Untuk
PowerShell, buat object lalu `ConvertTo-Json` seperti contoh API reference agar
quoting dan encoding lebih aman.

### Cara membaca logs

Gunakan:

```bash
docker compose logs -f --tail=100 translation-service
```

Cari startup completed, model/detector load, healthcheck state, status
translation, duration, dan exception class. Application log sengaja hanya
mencatat language code, character count, chunk count, dan timing—bukan raw user
text.

## Safe cleanup

Stop dan hapus container/network Compose sambil mempertahankan model cache:

```bash
docker compose down
```

Hanya jika operator sengaja ingin menghapus model cache, baca warning pada
bagian Model cache sebelum menggunakan `docker compose down -v`.
