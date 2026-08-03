# API reference

Base URL default adalah `http://127.0.0.1:8000`. Request dan response memakai
JSON, kecuali Swagger/ReDoc yang berupa HTML. Daftar kode bahasa runtime yang
valid tersedia melalui `GET /languages`.

## GET `/`

Mengembalikan identitas dasar process yang sedang berjalan. Tidak membutuhkan
request body.

```bash
curl http://127.0.0.1:8000/
```

```json
{
  "service": "M2M100 Translation Service",
  "version": "0.1.0",
  "status": "running",
  "docs": "/docs"
}
```

Endpoint ini bukan readiness check. Gunakan `/health` untuk memastikan model
dan detector loaded.

## GET `/health`

Mengembalikan status application, M2M100, Lingua, device, dan konfigurasi
chunking. Tidak menjalankan inference.

```bash
curl http://127.0.0.1:8000/health
```

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
```

Success: HTTP 200 dengan `status="healthy"`, `model_loaded=true`, dan
`language_detector_loaded=true`. Jika dependency lifespan tidak tersedia,
endpoint dapat mengembalikan HTTP 503.

## GET `/languages`

Mengembalikan kode bahasa dari service yang sudah dimuat.

```bash
curl http://127.0.0.1:8000/languages
```

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/languages"
```

Field utama:

- `languages`: seluruh kode source/target yang didukung tokenizer M2M100.
- `auto_detectable_languages`: subset yang juga dapat dipetakan ke Lingua.
- `count` dan `auto_detectable_count`: jumlah kedua daftar.
- `model_name` dan `language_detector`: implementation identifier.

Bahasa yang hanya ada pada `languages` tetap dapat dipakai dengan manual
`source_language`, tetapi tidak dapat dipilih oleh automatic detection.
Error umum: HTTP 503 jika model atau detector belum tersedia.

## POST `/detect-language`

Mendeteksi satu dominant language secara lokal dengan Lingua. Field `text`
wajib berupa string non-kosong dan maksimal mengikuti
`API_MAX_TEXT_CHARACTERS`.

Request:

```json
{
  "text": "Сегодня хорошая погода"
}
```

```bash
curl -X POST http://127.0.0.1:8000/detect-language \
  -H "Content-Type: application/json" \
  -d '{"text":"Сегодня хорошая погода"}'
```

```powershell
$body = @{ text = "Сегодня хорошая погода" } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/detect-language" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

Success: HTTP 200.

```json
{
  "language": "ru",
  "confidence": 0.8865,
  "confidence_margin": 0.8331,
  "detector": "lingua",
  "status": "detected",
  "candidates": [
    {"language": "ru", "confidence": 0.8865},
    {"language": "uk", "confidence": 0.0535}
  ]
}
```

`confidence_margin` adalah selisih kandidat pertama dan kedua; daftar
`candidates` dibatasi konfigurasi. Jika confidence atau margin tidak mencapai
threshold, API mengembalikan HTTP 422 `language_detection_uncertain`. Input
emoji-only, URL-only, mention-only, atau input dengan terlalu sedikit karakter
alfabet dapat ditolak sebagai `invalid_detection_input`. Confidence ini bukan
translation quality confidence.

Error umum: HTTP 413 untuk character limit, 422 untuk input invalid/uncertain,
500 untuk failure detector, dan 503 jika detector belum loaded.

## POST `/translate`

Menerjemahkan teks dengan M2M100. `text` dan `target_language` wajib.
`source_language` opsional dan default-nya `auto`; target tidak boleh `auto`.

Manual source request:

```json
{
  "text": "Good morning",
  "source_language": "en",
  "target_language": "id"
}
```

Automatic source request:

```json
{
  "text": "Сегодня хорошая погода",
  "target_language": "id"
}
```

```bash
curl -X POST http://127.0.0.1:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Good morning","source_language":"en","target_language":"id"}'
```

```powershell
$body = @{
  text = "Good morning"
  source_language = "en"
  target_language = "id"
} | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/translate" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

Success: HTTP 200.

```json
{
  "original_text": "Good morning",
  "translated_text": "Selamat pagi",
  "source_language": "en",
  "source_language_mode": "manual",
  "detected_language": null,
  "detection_confidence": null,
  "detection_confidence_margin": null,
  "target_language": "id",
  "model_name": "facebook/m2m100_418M",
  "device": "cpu",
  "status": "translated",
  "chunked": false,
  "chunk_count": 1,
  "chunk_token_limit": 400
}
```

Response fields:

| Field | Arti |
|---|---|
| `original_text` | Teks request yang tidak diubah |
| `translated_text` | Hasil merge translation atau teks asli untuk same-language |
| `source_language` | Kode source yang dipakai model |
| `source_language_mode` | `manual` atau `auto` |
| `detected_language` | Hasil Lingua pada mode auto; `null` pada manual |
| `detection_confidence` | Confidence Lingua; `null` pada manual |
| `detection_confidence_margin` | Margin Lingua; `null` pada manual |
| `target_language` | Kode target |
| `model_name` | Checkpoint yang digunakan |
| `device` | `cpu` atau `cuda` dari runtime |
| `status` | `translated` atau `unchanged` |
| `chunked` | `true` jika lebih dari satu chunk diterjemahkan |
| `chunk_count` | Jumlah chunk inference; same-language bernilai `0` |
| `chunk_token_limit` | Budget source token per chunk |

Automatic detection dijalankan sekali pada seluruh teks, bukan per chunk.
Jika source dan target sama, `status` menjadi `unchanged` dan original text
dikembalikan. Long text dipecah otomatis berdasarkan paragraph, sentence, dan
token budget. Tidak ada silent truncation dan tidak ada partial response jika
salah satu chunk gagal.

Error umum: HTTP 413 untuk character/token/chunk limit, 422 untuk validation,
unsupported language, atau uncertain detection, 500 untuk preparation atau
inference failure, dan 503 jika model/detector belum tersedia.

## GET `/docs` dan GET `/redoc`

`/docs` menyediakan Swagger UI dan `/redoc` menyediakan ReDoc dari OpenAPI
schema aplikasi. Keduanya tidak membutuhkan request body.

```bash
curl -I http://127.0.0.1:8000/docs
curl -I http://127.0.0.1:8000/redoc
```

Pada deployment tanpa authentication saat ini, UI ini juga tidak dilindungi.

## Error format

Semua application error memakai envelope berikut; field `details` hanya ada
pada error tertentu.

```json
{
  "error": {
    "code": "unsupported_language",
    "message": "Language code 'xx' is not supported."
  }
}
```

| Error code | HTTP | Kondisi utama |
|---|---:|---|
| `request_validation_error` | 422 | Body atau field tidak valid |
| `invalid_translation_input` | 422 | Input translation kosong/tidak valid di engine |
| `unsupported_language` | 422 | Kode source/target tidak didukung |
| `invalid_detection_input` | 422 | Konten alfabet tidak cukup untuk detection |
| `language_detection_uncertain` | 422 | Confidence atau margin kurang |
| `text_too_large` | 413 | Character limit API terlampaui |
| `input_too_long` | 413 | Hard input token limit terlampaui |
| `too_many_chunks` | 413 | Chunk count limit terlampaui |
| `model_not_loaded` | 503 | Model tidak tersedia |
| `language_detector_not_loaded` | 503 | Detector tidak tersedia |
| `translation_failed` | 500 | Inference translation gagal |
| `language_detection_failed` | 500 | Lingua gagal secara internal |
| `text_chunking_failed` | 500 | Source tidak dapat dipersiapkan secara lossless |
| `translation_output_truncated` | 500 | Generation mencapai limit tanpa EOS |
| `internal_server_error` | 500 | Error tidak terduga |

Startup juga mempunyai `model_load_failed`, `language_detector_load_failed`,
dan `device_configuration_error` (HTTP 503). Client tidak menerima traceback,
path lokal, atau raw source text melalui error internal.
