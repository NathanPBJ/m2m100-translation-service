# Senior handover

Dokumen ini adalah titik masuk singkat untuk menerima service. Detail kontrak
dan operasi tersedia pada dokumen yang ditautkan dari README.

## What was delivered

- FastAPI translation service.
- Local translation dengan `facebook/m2m100_418M`.
- Local language detection dengan Lingua.
- Manual dan automatic source language.
- Token-aware long-text chunking tanpa silent truncation.
- CPU-only Docker Compose deployment dengan persistent model cache.
- Unit/integration tests, smoke scripts, dan handover documentation.

## Main command

```bash
docker compose up --build -d
```

## Main URLs

```text
API: http://127.0.0.1:8000
Swagger: http://127.0.0.1:8000/docs
Health: http://127.0.0.1:8000/health
```

## Integration contract

Aplikasi utama atau scraper memanggil `POST /translate` dengan JSON. Source
language default ke `auto`, sehingga request minimal adalah:

```json
{
  "text": "Bonjour tout le monde",
  "target_language": "id"
}
```

Jika caller sudah mengetahui bahasa source, kirim `source_language` secara
manual untuk menghindari ambiguity detection. Lihat
[API reference](API_REFERENCE.md) untuk seluruh response field dan error.

## Important operational notes

- First startup mengunduh model jika named volume masih kosong.
- Pertahankan named volume model; jangan memakai `docker compose down -v`
  kecuali memang ingin menghapus cache dan mengunduh ulang model.
- Jalankan satu Uvicorn worker. Multiple workers menduplikasi model di RAM.
- Docker image saat ini CPU-only dan `torch.cuda.is_available()` bernilai
  `false`.
- Long text dapat lambat karena inference dijalankan sekali per chunk secara
  berurutan.
- Jangan memakai `--reload` untuk deployment dan jangan menambah worker tanpa
  perencanaan memory serta concurrency.
- Jangan expose service langsung ke internet tanpa authentication dan network
  protection. Service belum memiliki rate limiting atau TLS.
- Model dan model cache tidak berada dalam repository atau source archive.

## Actual development performance

Angka berikut adalah **development measurements pada mesin pengembangan, bukan
SLA atau benchmark universal**:

| Skenario | Hasil pengukuran |
|---|---:|
| Short text | sekitar 4,43 detik |
| Opinion fixture, 4 chunks | sekitar 136 detik |
| Product review, 10 chunks | sekitar 252 detik |
| Chronology, 20 chunks | sekitar 493 detik |
| Same-language long text | sekitar 0,05 detik |

Mesin deployment dapat lebih cepat atau lebih lambat karena CPU, RAM, load,
disk, dan konfigurasi berbeda. Lingua detection bukan bottleneck utama dalam
pengukuran ini. M2M100 generation pada CPU adalah bottleneck utama, dan long
text menjalankan generation secara sequential untuk setiap chunk.

## Known language detection limitation

Indonesian dan Malay dapat ambigu, terutama pada teks pendek. Pada pengujian
aktual, “Selamat pagi” dapat terdeteksi sebagai Malay (`ms`). Teks Indonesia
yang lebih panjang juga dapat ditolak jika confidence atau margin tidak
mencapai threshold. Threshold sengaja tidak dilonggarkan hanya untuk memaksa
hasil. Jika bahasa sudah diketahui, caller dapat mengirim
`source_language="id"`.

Detection confidence menunjukkan keyakinan Lingua terhadap bahasa, bukan
kualitas hasil translation.

## Current limitations

- Tidak ada authentication, rate limiting, atau TLS.
- Tidak ada batch endpoint atau streaming.
- Tidak ada translation cache, background queue, atau job status.
- Tidak ada GPU Docker image.
- Satu worker dan inference CPU membatasi throughput.
- Tidak ada SLA performa.
- Satu dominant language dipakai untuk seluruh request mixed-language.
- Konteks antar-chunk dapat berkurang pada long text.
- Service tidak cocok langsung diekspos ke public internet.

## Recommended next steps

Rekomendasi ini belum diimplementasikan dalam release ini:

1. Integrasikan service dengan scraper melalui internal network.
2. Tambahkan API authentication atau network restriction di layer deployment.
3. Jalankan benchmark pada server deployment yang sebenarnya.
4. Pertimbangkan queue jika scraper mengirim banyak post sekaligus.
5. Pertimbangkan optimasi model hanya setelah benchmark dan quality check.
6. Tambahkan monitoring jika service masuk operasi production.

## Handover acceptance checklist

- [ ] Repository private dapat diakses senior.
- [ ] `docker compose up --build -d` berhasil.
- [ ] Container mencapai status `healthy`.
- [ ] Swagger dapat dibuka.
- [ ] Manual translation berhasil.
- [ ] Automatic translation berhasil.
- [ ] Long translation berhasil.
- [ ] Named model volume tersedia dan dipertahankan.
- [ ] Log dapat dibaca tanpa raw user text.
- [ ] `.env` tidak di-commit.
- [ ] Model atau model weight tidak di-commit.
