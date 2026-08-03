# Architecture

## System overview

Service ini menyediakan API HTTP untuk menerjemahkan teks dengan model lokal
`facebook/m2m100_418M`. FastAPI menangani kontrak dan lifecycle aplikasi,
sedangkan PyTorch menjalankan inference pada CPU dalam deployment Docker saat
ini. Tidak ada translation API eksternal dalam alur inference.

```text
Client atau scraper
→ FastAPI
→ request validation
→ optional Lingua language detection
→ M2M100 translation service
→ token-aware chunking jika diperlukan
→ JSON response
```

## Components

- **FastAPI application** (`app/main.py`) membuat aplikasi, mendaftarkan route
  dan exception handler, serta menyimpan service yang sudah dimuat di
  `app.state`.
- **Lifespan** mengatur load dan unload model serta detector. Request baru hanya
  dapat dilayani setelah startup selesai.
- **Translation service** (`app/services/translation.py`) memiliki tokenizer,
  model M2M100, daftar bahasa, validasi token, generation, decode, dan merge
  hasil chunk.
- **Language detection service** (`app/services/language_detection.py`)
  membangun satu detector Lingua dari intersection bahasa Lingua dan kode
  tokenizer M2M100. URL dan mention dibersihkan hanya pada salinan untuk
  detection; teks yang diterjemahkan tidak diubah.
- **Text chunking service** (`app/services/text_chunking.py`) memecah teks tanpa
  menghilangkan source span atau separator, menggunakan token counter dari
  tokenizer yang sudah dimuat.
- **API schemas** (`app/schemas/`) adalah kontrak Pydantic untuk request,
  response, health, dan error.
- **Exception handlers** (`app/api/error_handlers.py`) memetakan exception
  domain menjadi status HTTP dan envelope error yang aman. Traceback dicatat di
  server untuk kegagalan internal, bukan dikirim ke client.
- **Threadpool** Starlette menjalankan model load/unload, detection, dan
  translation synchronous di luar event loop.
- **Translation semaphore** membatasi jumlah request inference yang masuk
  bersamaan dalam satu process.
- **Internal `RLock`** melindungi model serta mutable `tokenizer.src_lang`
  selama load, unload, tokenisasi, generation, dan decode.
- **Docker container** memakai Linux image CPU-only, user non-root, dan satu
  Uvicorn worker.
- **Named model-cache volume** dipasang ke `/models` agar download model dapat
  dipakai kembali setelah container dibuat ulang.

## Startup lifecycle

1. Uvicorn memulai FastAPI dan masuk ke lifespan startup.
2. Tokenizer dan model M2M100 dimuat melalui threadpool.
3. Daftar supported translation languages dibaca dari tokenizer.
4. Detector Lingua dibangun dari intersection bahasa tersebut dengan bahasa
   yang didukung Lingua.
5. Semaphore translation process-local tersedia dengan nilai dari
   `TRANSLATION_MAX_CONCURRENCY`.
6. Startup dinyatakan selesai setelah model dan detector berhasil dimuat.
7. Docker healthcheck memanggil `/health`; container menjadi `healthy` setelah
   endpoint mengonfirmasi model dan detector loaded.

Model atau detector yang gagal dimuat membuat startup gagal. Aplikasi tidak
masuk kondisi siap sebagian.

## Shutdown lifecycle

1. Docker init meneruskan SIGTERM ke process Uvicorn.
2. FastAPI menjalankan bagian shutdown lifespan.
3. Referensi detector Lingua dilepas.
4. Referensi tokenizer/model M2M100 dilepas dan cache CUDA dikosongkan hanya
   jika device CUDA digunakan.
5. Uvicorn berhenti. Compose memberikan `stop_grace_period` dua menit.

## Translation flow

Manual source language:

```text
Request
→ validate source dan target
→ translate
→ response dengan source_language_mode="manual"
```

Automatic source language:

```text
Request
→ detect source once pada seluruh teks
→ validate detected source dan target
→ translate
→ response dengan source_language_mode="auto"
```

Jika source dan target sama, service mengembalikan teks asli dengan status
`unchanged` tanpa tokenisasi, chunking, atau model generation.

## Long-text flow

```text
Full text
→ detect language once
→ paragraph split
→ sentence split
→ Unicode-aware fallback
→ sequential chunk translation
→ ordered result merge
```

Chunker mempertahankan leading/trailing whitespace dan separator paragraf di
luar span yang dikirim ke model. Setiap hasil digabung sesuai urutan source.
Sebelum inference, source direkonstruksi dan harus identik dengan input. Setiap
tokenizer call memakai `truncation=False`; input tidak dipotong diam-diam.
Kegagalan satu chunk menggagalkan seluruh request dan tidak menghasilkan
partial translation response.

## Concurrency model

FastAPI semaphore dan `RLock` menyelesaikan masalah yang berbeda:

- Semaphore membatasi request M2M100 yang sedang inference dalam satu process.
- `RLock` menjaga mutable tokenizer state dan urutan operasi internal model.
- Threadpool mencegah operasi PyTorch atau Lingua synchronous berjalan langsung
  pada event loop.
- Seluruh chunk satu request diterjemahkan berurutan di dalam satu critical
  section; tidak ada parallel inference antar-chunk.
- Deployment menggunakan satu Uvicorn worker agar model tidak diduplikasi di
  memory. Menambah worker membuat satu copy model dan semaphore terpisah per
  process.

Semaphore bukan rate limiter dan tidak mengatur traffic lintas process atau
host.

## Storage

- Source application disalin ke image di `/service/app`.
- Model tidak dibundel ke image dan tidak disimpan dalam Git.
- Cache Hugging Face, Torch, tokenizer, dan model berada pada named volume yang
  dipasang ke `/models`.
- `.env` tidak disimpan di Git atau disalin ke image.
- `.dockerignore` mengecualikan Git metadata, tests, scripts, docs, cache, model,
  virtual environment, dan log dari build context.
- `docker compose down` mempertahankan cache; container berikutnya dapat
  menggunakan file yang sama. `docker compose down -v` menghapus cache.

## Failure behavior

| Kondisi | Perilaku |
|---|---|
| Model gagal load | Startup aplikasi gagal |
| Detector gagal load | Startup aplikasi gagal |
| Detection tidak cukup yakin | HTTP 422 `language_detection_uncertain` |
| Input tidak cocok untuk detection | HTTP 422 `invalid_detection_input` |
| Character limit terlampaui | HTTP 413 `text_too_large` |
| Hard token limit terlampaui | HTTP 413 `input_too_long` |
| Terlalu banyak chunk | HTTP 413 `too_many_chunks` |
| Chunk preparation gagal | HTTP 500 `text_chunking_failed` |
| Output mencapai limit tanpa EOS | HTTP 500 `translation_output_truncated` |
| Inference gagal | HTTP 500 `translation_failed` |

API tidak mengirim partial response. Error internal memakai pesan generik dan
tidak mengembalikan traceback atau source text.

## Security boundary

Container berjalan sebagai `appuser` dengan UID/GID `10001:10001`, tidak
privileged, tidak memasang Docker socket, dan tidak memakai host network.
Repository serta image tidak menyediakan secret bawaan.

Boundary ini belum cukup untuk paparan internet publik: API belum mempunyai
authentication, authorization, TLS, atau rate limiting. Tempatkan service di
internal network atau di belakang reverse proxy/gateway yang menyediakan
kontrol tersebut. Teks request diproses lokal, tetapi operator tetap harus
memperlakukan input dan log operasional sesuai kebijakan data deployment.
