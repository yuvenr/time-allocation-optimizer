# Time Allocation Optimizer

Sistem cerdas untuk perencanaan rencana studi akademik dan optimasi alokasi waktu/beban belajar mahasiswa menggunakan pendekatan **Constraint Satisfaction**, **Association Rule Mining (ARM / FP-Growth)**, dan **Large Language Model (LLM)** untuk estimasi beban kognitif.

Sistem menyusun rekomendasi rencana studi mahasiswa per semester dari semester berjalan sampai lulus berdasarkan batasan kurikulum, dengan memadukan:

1. **Aturan Kurikulum & Batasan Akademik**: Memastikan pemenuhan batasan keras seperti prasyarat mata kuliah, semester penawaran (ganjil/genap), batas SKS menurut IP semester sebelumnya, paket semester awal, syarat kegiatan lapangan (PLP) dan Tugas Akhir (Capstone), pilihan tema, serta batas masa studi.
2. **Association Rule Mining (FP-Growth)**: Menambang pola pengambilan mata kuliah dari data riwayat KRS untuk memprioritaskan dan mengurutkan rekomendasi mata kuliah pilihan sesuai minat mahasiswa dan tren historis.
3. **Optimasi Beban Kognitif & Alokasi Waktu**: Mengevaluasi tingkat kompleksitas kognitif taksonomi Bloom revisi per Capaian Pembelajaran Mata Kuliah (CPMK) menggunakan LLM, yang digunakan sebagai ambang batas beban kognitif per semester (θ) pada mode *hybrid* agar mahasiswa tidak mengalami kelebihan beban di semester tertentu.

Mode `arm` (pembanding) dan `hybrid` memakai algoritma yang sama; perbedaan hasilnya hanya berasal dari batasan θ.

## Fitur

| Peran | Fitur |
|---|---|
| Mahasiswa | Profil & riwayat nilai, pilihan MK per tema, rencana per semester dengan alasan per MK, grafik beban kognitif, validasi 16 aturan akademik, geser MK antarsemester (*drag & drop*) dengan pemeriksaan aturan secara interaktif |
| DPA (berkata sandi) | Tinjauan rencana studi, perbandingan dengan rekomendasi sistem, validasi akhir & catatan bimbingan |
| Administrator (berkata sandi) | Daftar periksa kesiapan data dari nol, templat data, impor data akademik, preprocessing, ARM & analisis sensitivitas, penilaian Bloom oleh LLM, evaluasi LLM vs pakar (ICC), simulasi komparasi, ekspor CSV untuk JASP, paket data uji |

Panduan: [docs/PANDUAN_PENGGUNA.md](docs/PANDUAN_PENGGUNA.md) ·
[docs/PANDUAN_ADMINISTRATOR.md](docs/PANDUAN_ADMINISTRATOR.md) ·
deploy: [docs/DEPLOY_STREAMLIT_NEON.md](docs/DEPLOY_STREAMLIT_NEON.md).

## Menjalankan di komputer sendiri

Butuh Python 3.11 atau lebih baru.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                    # isi hash kata sandi peran: python -m src.auth
streamlit run app/streamlit_app.py
```

Tanpa `DATABASE_URL`, aplikasi memakai SQLite `data/crs.db` yang dibuat otomatis. Sistem awalnya kosong:
masuk sebagai Administrator, lalu ikuti tab **Mulai di sini** (paket data uji atau langkah demi langkah).
Tanpa data apa pun, rencana tetap dapat disusun dari struktur kurikulum dan aturan.

## Pengujian

```bash
python -m pytest            # seluruh tes luring; tidak memanggil API LLM, tidak menyentuh data/ aplikasi
```

Tes memakai data fiktif deterministik. `CRS_TEST_DATABASE_URL=postgresql://...` menjalankan seluruh tes di
PostgreSQL. GitHub Actions menjalankan keduanya pada setiap *push*.

## Struktur

```
app/                 antarmuka Streamlit (streamlit_app.py, tampilan.py)
src/                 rules.py (aturan), planner.py (penyusun rencana), arm.py, llm/ (penilaian Bloom),
                     preprocessing.py, simulate.py, metrics.py, statistik.py, export.py, db.py (SQLite/PostgreSQL),
                     penyesuaian.py, kesiapan.py, templat.py, paket_uji.py, berkas.py, auth.py
data/kurikulum_2024/ dataset Kurikulum 2024 sampel
tests/               pytest
docs/                panduan pengguna, administrator, dan deploy
config.yaml          seluruh ambang & parameter
```

## Data dan privasi

- Repositori **tidak** memuat data mahasiswa, NIM, basis data, maupun kunci API. Data akademik diimpor oleh
  administrator lewat aplikasi; NIM dianonimkan dengan *hash* bergaram sebelum disimpan.
- Yang dikirim ke LLM hanya teks kurikulum (deskripsi MK, CPMK, Sub-CPMK).
- `data/kurikulum_2024/` bersumber dari dokumen kurikulum Prodi PTI UB 2024–2029.

## Lisensi

- **Kode program** (seluruh isi repositori kecuali yang disebut di bawah): [MIT License](LICENSE),
  © 2026 Yuven Reksa Pratama.
- **Dataset Kurikulum 2024** (`data/kurikulum_2024/`) **tidak** termasuk lisensi MIT. Isinya diekstraksi dari
  dokumen kurikulum Program Studi S1 Pendidikan Teknologi Informasi FILKOM UB dan tetap menjadi hak Program
  Studi; disertakan sebagai sampel referensi agar aplikasi dapat dijalankan. Lihat
  [data/kurikulum_2024/README.md](data/kurikulum_2024/README.md).
