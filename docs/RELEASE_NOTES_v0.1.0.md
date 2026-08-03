# v0.1.0 release notes

## Initial internal release

`v0.1.0` adalah initial internal release untuk local translation service yang
akan diintegrasikan dengan client atau scraper melalui HTTP. Release ini
menutup tahap implementasi dasar dan handover; release ini tidak diklaim fully
production-hardened.

## Highlights

- FastAPI API dengan health, languages, detection, dan translation endpoint.
- `facebook/m2m100_418M` berjalan lokal melalui PyTorch/Transformers.
- Lingua melakukan automatic dominant-language detection secara lokal.
- Caller tetap dapat memberikan manual source language.
- Token-aware long-text chunking menjaga urutan dan separator serta mencegah
  silent truncation.
- Docker Compose CPU-only menjalankan satu non-root Uvicorn worker.
- Named volume mempertahankan model cache antar-container recreation.
- Unit/integration test suite dan smoke/final validation scripts disertakan.

## Run

Requirement utama adalah Docker Engine/Desktop dengan Linux containers dan
Docker Compose v2.

```bash
docker compose up --build -d
docker compose ps
```

Swagger tersedia di `http://127.0.0.1:8000/docs`. First startup mengunduh model
ke named volume jika cache kosong, sehingga container dapat berada dalam status
`starting` cukup lama. Jangan menghapus volume dengan `docker compose down -v`
kecuali download ulang memang diinginkan.

## Main API

Integrasi utama menggunakan `POST /translate`:

```json
{
  "text": "Bonjour tout le monde",
  "target_language": "id"
}
```

Kode valid tersedia melalui `/languages`; gunakan `/health` sebagai readiness
check dan `/detect-language` jika hanya membutuhkan detection.

## Validation status

Baseline sebelum handover adalah 237 test lulus dan real Docker build/smoke test
lulus. Release candidate kemudian divalidasi kembali dengan full Pytest suite,
Ruff lint/format, compile check, lazy-import/OpenAPI check, Docker Compose
config/build/startup, short Docker smoke test, final validation script, serta
clean-clone verification. Hasil aktual final dicatat pada laporan handover
release, bukan dijadikan klaim performa lintas mesin di dokumen ini.

## Development performance

Pengukuran pada mesin development, bukan SLA:

- Short text sekitar 4,43 detik.
- Opinion fixture (4 chunks) sekitar 136 detik.
- Product review (10 chunks) sekitar 252 detik.
- Chronology (20 chunks) sekitar 493 detik.
- Same-language long text sekitar 0,05 detik.

CPU M2M100 generation adalah bottleneck utama dan long text menerjemahkan chunk
secara sequential. Hardware deployment lain dapat memberi hasil berbeda.

## Security and limitations

Container non-root dan tidak membawa secret atau model bawaan. API belum
mempunyai authentication, TLS, rate limiting, batch, streaming, queue,
translation cache, GPU image, monitoring stack, atau SLA. Jangan expose
service langsung ke public internet; gunakan internal network atau gateway yang
memberikan kontrol akses dan transport security.

Indonesian dan Malay dapat ambigu pada automatic detection. Caller sebaiknya
mengirim manual `source_language="id"` ketika bahasa sudah diketahui.

## Upgrade and release identity

Tidak ada upgrade path karena ini release pertama. Source release ditetapkan
oleh annotated Git tag `v0.1.0` dengan message
`rilis awal translation service`. Release commit adalah commit yang ditunjuk
tag tersebut dan dapat diperiksa dengan:

```bash
git rev-list -n 1 v0.1.0
git show --stat v0.1.0
```
