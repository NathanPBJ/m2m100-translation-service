# Changelog

Perubahan penting pada project dicatat di file ini.

## [0.1.0] - 2026-08-03

Initial internal release.

### Added

- FastAPI application foundation dan health endpoint.
- Local `facebook/m2m100_418M` translation engine.
- Manual multilingual translation dan supported-language endpoint.
- Automatic source-language detection dengan Lingua.
- Token-aware long-text chunking tanpa silent truncation.
- Centralized error handling dengan response envelope konsisten.
- CPU-only Dockerfile dan Docker Compose deployment.
- Persistent named volume untuk model cache.
- Unit/integration tests serta native dan Docker smoke scripts.
- Architecture, API, operations, release, dan senior handover documentation.

### Security

- Docker runtime menggunakan non-root user.
- Secret, `.env`, model, dan model cache tidak disimpan di Git atau application
  image.
- Authentication, TLS, dan rate limiting belum tersedia; service harus
  dilindungi sebelum digunakan di luar internal network.

### Known limitations

- M2M100 generation pada CPU mempunyai latency tinggi, terutama untuk long text.
- Satu Uvicorn worker digunakan agar model tidak diduplikasi.
- Tidak ada authentication atau rate limiting.
- Tidak ada batch endpoint, streaming, translation cache, atau background queue.
