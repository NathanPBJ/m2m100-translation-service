# Third-party notices

Repository ini berlisensi MIT sebagaimana tercantum dalam `LICENSE`. Package,
runtime, dan model di bawah adalah karya pihak ketiga dan tidak menjadi milik
repository ini. Daftar ini adalah ringkasan, bukan pengganti license text atau
legal advice. Pengguna deployment bertanggung jawab memeriksa serta mematuhi
license dan ketentuan upstream yang berlaku pada versi/artifact yang digunakan.

Versi package adalah versi yang terpasang pada environment development saat
audit 3 Agustus 2026. Range yang didukung project tetap tercatat pada
`pyproject.toml` dan `requirements.txt`.

| Item | Kegunaan | Versi/artifact audit | License terverifikasi | Sumber verifikasi |
|---|---|---|---|---|
| `facebook/m2m100_418M` | Tokenizer dan multilingual translation model | Hugging Face checkpoint `facebook/m2m100_418M` | MIT | [Hugging Face model card/repository](https://huggingface.co/facebook/m2m100_418M) menandai license `mit` |
| PyTorch (`torch`) | Tensor runtime dan model inference | `2.13.0+cpu` (`2.13.0` pada installed metadata) | BSD-style upstream; installed wheel menyatakan composite SPDX expression untuk bundled components | [PyTorch LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) dan installed `License-Expression`: `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` |
| Hugging Face Transformers | Implementasi M2M100 tokenizer/model | `4.57.6` | Apache-2.0 | Installed package metadata dan [Transformers repository](https://github.com/huggingface/transformers) |
| SentencePiece | Subword tokenization yang dibutuhkan M2M100 | `0.2.2` | Apache-2.0 | Installed `License-Expression` dan [SentencePiece repository](https://github.com/google/sentencepiece) |
| Lingua Language Detector | Offline dominant-language detection | `lingua-language-detector 2.2.0` | Apache-2.0 | Installed classifier dan [Lingua Rust repository](https://github.com/pemistahl/lingua-rs) |
| FastAPI | HTTP API framework dan OpenAPI | `0.115.14` | MIT | Installed classifier dan [FastAPI LICENSE](https://github.com/fastapi/fastapi/blob/master/LICENSE) |
| Uvicorn | ASGI server | `0.52.0` | BSD-3-Clause | Installed `License-Expression` dan package metadata |
| Pydantic | Request/response validation | `2.13.4` | MIT | Installed `License-Expression` dan [Pydantic LICENSE](https://github.com/pydantic/pydantic/blob/main/LICENSE) |
| Pydantic Settings | Environment-based configuration | `2.14.2` | MIT | Installed `License-Expression`/classifier dan [upstream repository](https://github.com/pydantic/pydantic-settings) |
| HTTPX | Test client untuk API tests | `0.28.1` | BSD-3-Clause | Installed metadata dan [HTTPX repository](https://github.com/encode/httpx) |
| Ruff | Lint dan format verification | `0.16.1` | MIT | Installed `License-Expression` dan [Ruff repository](https://github.com/astral-sh/ruff) |
| Pytest | Unit/integration test runner | `9.1.1` | MIT | Installed `License-Expression` dan [pytest repository](https://github.com/pytest-dev/pytest) |
| Python base Docker image | Linux/Python base untuk runtime image | `python:3.12-slim-bookworm` | Image packaging repository: MIT; Python, Debian, dan installed OS packages mempunyai license masing-masing | [Docker Official Image packaging](https://github.com/docker-library/python) dan license/artifact data dari image/upstream components |

## Notes on redistributed artifacts

- Model weight tidak berada di Git, source archive, atau application image.
  Model diunduh ke named volume saat runtime. License model tetap berlaku pada
  file yang diunduh.
- Docker image menggabungkan Python, Debian, CPU PyTorch wheel, dan transitive
  dependencies. Satu label license tidak menggantikan notices masing-masing
  component; operator yang mendistribusikan image harus meninjau license data
  image yang benar-benar dibangun.
- PyTorch wheel membawa third-party components. Composite expression pada
  installed metadata dicatat agar notice ini tidak menyederhanakannya menjadi
  satu license saja.
- Dependency transitive tidak ditulis satu per satu di ringkasan ini. Metadata
  distribution dan license files dalam artifact terpasang tetap menjadi sumber
  yang harus diperiksa ketika image didistribusikan.
