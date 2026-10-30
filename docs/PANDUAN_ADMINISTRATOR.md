# Panduan Administrator: Sistem Rekomendasi Rencana Studi PTI UB

Panduan ini ditujukan bagi administrator sistem dan dosen penguji. Isinya: cara menyiapkan sistem **dari
kondisi kosong** (belum ada data yang dimasukkan) sampai rencana studi mahasiswa divalidasi DPA, format setiap
berkas masukan beserta templatnya, serta skenario pengujian dengan hasil yang diharapkan.

Panduan untuk peran Mahasiswa dan DPA ada di [PANDUAN_PENGGUNA.md](PANDUAN_PENGGUNA.md). Cara memasang aplikasi
online (Streamlit Community Cloud + Neon) ada di
[DEPLOY_STREAMLIT_NEON.md](DEPLOY_STREAMLIT_NEON.md).

## Daftar isi

1. [Gambaran alur](#1-gambaran-alur)
2. [Masuk sebagai Administrator](#2-masuk-sebagai-administrator)
3. [Tab Mulai di sini](#3-tab-mulai-di-sini)
4. [Cara cepat: paket data uji](#4-cara-cepat-paket-data-uji)
5. [Menyiapkan kondisi kosong untuk pengujian](#5-menyiapkan-kondisi-kosong-untuk-pengujian)
6. [Langkah 1 sampai 12 dari nol](#6-langkah-1-sampai-12-dari-nol)
7. [Templat dan kamus data](#7-templat-dan-kamus-data)
8. [Skenario pengujian untuk dosen](#8-skenario-pengujian-untuk-dosen)
9. [Cadangan dan atur ulang](#9-cadangan-dan-atur-ulang)
10. [Privasi dan keamanan data](#10-privasi-dan-keamanan-data)
11. [Pemecahan masalah](#11-pemecahan-masalah)

## 1. Gambaran alur

```
 Data akademik (KRS + IP/IPK)  ─┐
 Riwayat KRS multisemester ─────┴─► Preprocessing ─► ARM (FP-Growth) ─┐
 CPMK Kurikulum 2024 (bawaan) ─► Penilaian Bloom oleh LLM ─► CL per MK ─► θ ─┤
                                                                           ├─► Rencana studi (Mahasiswa)
 Aturan kurikulum 2024 (bawaan) ───────────────────────────────────────────┘        └─► Validasi DPA
 Penilaian pakar (opsional) ─► Evaluasi RM1 (ICC LLM vs pakar)
 Simulasi komparasi mode ARM vs Hybrid ─► CSV untuk JASP
```

Yang **sudah tersedia** sejak awal (tidak perlu diunggah): dataset Kurikulum 2024 (72 MK, CPMK, prasyarat,
tema pilihan, struktur Jalur A sampai F, pita SKS). Dataset ini hanya dibaca.

Yang **harus dimasukkan** administrator: data akademik, riwayat KRS multisemester, penilaian Bloom (dihasilkan
LLM), dan secara opsional penilaian pakar.

Tanpa data KRS sekalipun sistem tetap dapat menyusun rencana dari struktur kurikulum dan aturan domain. ARM
belum aktif, dan hal ini disebutkan di asumsi setiap rencana. Kondisi ini sendiri termasuk bagian pengujian
(skenario U-02).

## 2. Masuk sebagai Administrator

1. Pilih **Administrator** di sidebar kiri, isi kata sandi, klik **Masuk**.
2. Kata sandi disimpan sebagai hash di `.env` (lokal) atau *Secrets* Streamlit Cloud (`CRS_HASH_ADMIN`).
   Untuk membuat hash baru jalankan `python -m src.auth`, lalu salin baris keluarannya.
3. Setelah 5 kali salah, peran terkunci untuk sesi itu. Muat ulang halaman untuk mencoba lagi. Setiap percobaan
   masuk tercatat di log akses (tanpa kata sandi).
4. Klik **Keluar** di sidebar untuk mengakhiri sesi.

Sidebar *Status sistem* menampilkan ringkasan: data KRS, jumlah aturan ARM, sumber skor CL, nilai θ, model LLM,
dan jenis basis data (SQLite lokal atau PostgreSQL di cloud).

## 3. Tab Mulai di sini

Tab pertama halaman Administrator. Bagian-bagiannya:

| Bagian | Fungsi |
|---|---|
| **Unduh panduan administrator (Markdown)** | Mengunduh dokumen ini. |
| Bilah kemajuan + tabel langkah | Status 12 langkah sejak sistem kosong: ✅ selesai, ⬜ belum, ⚠️ perlu perhatian (mis. hasil proses basi karena data berubah), ➖ opsional. Status dihitung otomatis dari isi basis data, tidak dapat diubah manual. |
| Kotak *Langkah berikutnya* | Langkah wajib pertama yang belum selesai beserta tempatnya di aplikasi. |
| **Templat data** | Tabel berkas templat + tombol unduh. Setiap templat berisi **nama kolom saja** (baris isian kosong). Tombol *Kolom ...* menampilkan kamus kolomnya (wajib/opsional dan keterangan). |
| **Unduh templat data akademik (xlsx)** | Templat ekspor sistem akademik (dua sheet). |
| **Unduh daftar kode MK Kurikulum 2024** | Acuan kode MK, nama, SKS, sifat, tema, dan semester Jalur A untuk mengisi templat. |
| **Unduh formulir penilaian pakar (terisi daftar CPMK)** | Formulir pakar yang sudah memuat seluruh 348 CPMK; pakar cukup mengisi `kode_pakar` dan `bloom`. |
| **Cadangan & atur ulang** | Lihat bagian 9. |

Tombol unduh templat yang sama juga tersedia langsung di dekat setiap tempat unggah (tab Data akademik, Beban
kognitif, dan halaman Mahasiswa).

## 4. Cara cepat: paket data uji

Untuk sesi pengujian bersama dosen, sistem dapat disiapkan dari kondisi kosong sampai siap pakai dengan **satu berkas
zip paket data uji** dalam satu kali proses (sekitar 15 detik sampai 1 menit, tanpa API berbayar). Hasilnya sama
persis dengan sistem asal paket: aturan ARM, θ, skor CL, hasil simulasi, dan validasi KRS identik.

| Berkas | Isi | Format |
|---|---|---|
| `paket_krs_akademik.csv` | KRS akademik anonim (tanpa NIM) | sama dengan `templat_krs_akademik.csv` |
| `paket_ip_semester.csv` | IP/IPK semester anonim | sama dengan `templat_ip_semester.csv` |
| `paket_rating_llm.csv` | Penilaian Bloom per CPMK oleh `gpt-6-luna`, 3 pengulangan | sama dengan `templat_rating_llm.csv` |
| `histori_krs.csv` + `SUMBER_DATA.txt` + `kalibrasi_sintetis.json` | Riwayat KRS multisemester beserta penanda asal datanya | disalin apa adanya (jangan diubah atau diganti nama) |

**Membuat paket** (sekali, oleh peneliti, dari sistem yang sudah berisi data): Mulai di sini → *Cadangan &
atur ulang* → **Unduh paket data uji (zip)**, atau `python -m src.paket_uji buat` (berkas
tersimpan di `data/paket_uji/`). Paket berisi data akademik anonim: serahkan langsung, jangan diunggah ke
repositori publik.

**Menyiapkan sistem dari paket** (saat pengujian): Mulai di sini → *Cara cepat: siapkan sistem dari paket data
uji*. Panel ini otomatis terbuka bila sistem masih kosong.
1. Pastikan baris 1 daftar periksa (konfigurasi) tidak berstatus ⬜. Kunci API LLM tidak diperlukan.
2. Pilih berkas zip paket pada **Berkas paket data uji** (atau seluruh isinya sekaligus). Tabel di bawahnya
   menunjukkan jenis setiap berkas yang dikenali. Nama tiga CSV paket boleh diganti; nama berkas riwayat tidak.
3. Klik **Siapkan sistem dari paket data uji**. Bilah kemajuan menampilkan tahapnya: impor data akademik,
   memasang riwayat multisemester, memulihkan rating LLM, lalu preprocessing, ARM, CL & θ, evaluasi,
   simulasi, dan validasi KRS.
4. Daftar periksa menunjukkan langkah 3 sampai 8 dan 11 ✅. Lanjutkan ke skenario U-09 sampai U-14.

Riwayat KRS multisemester dikemas apa adanya, byte per byte, bersama penanda asal datanya, karena hasil proses
dicocokkan dengan isi persis berkas itu. Jangan mengunggah berkas riwayat ini lewat *Unggah riwayat alumni*:
tombol itu menganggap riwayat sebagai data alumni asli. Paket tanpa berkas riwayat tetap dapat dipakai; sistem
lalu membangkitkan riwayat pelengkap baru dengan seed tetap (hasil ARM dan simulasi dapat berbeda dari sistem
asal paket).

## 5. Menyiapkan kondisi kosong untuk pengujian

Pilih salah satu cara:

**A. Atur ulang dari aplikasi (paling mudah).** Mulai di sini → *Cadangan & atur ulang*:
1. Klik **Unduh cadangan rating LLM** dan simpan berkasnya.
2. Tentukan **Pertahankan rating Bloom LLM**. Biarkan tercentang bila penilaian LLM tidak akan diuji ulang
   (menghemat biaya API). Hapus centang untuk pengujian benar-benar dari nol; rating dapat dipulihkan dari
   cadangan pada langkah 7.
3. Ketik `ATUR ULANG`, lalu klik **Atur ulang sistem ke kondisi awal**.
4. Tabel langkah di tab Mulai di sini kembali menunjukkan ⬜ untuk langkah 3 sampai 12.

**B. Basis data terpisah (tanpa menyentuh data yang sedang dipakai).**
- Lokal: jalankan aplikasi dengan folder data baru, misalnya
  `CRS_DB=data/uji.db CRS_RAW=data/uji_raw streamlit run app/streamlit_app.py`
  (PowerShell: `$env:CRS_DB="data/uji.db"; $env:CRS_RAW="data/uji_raw"; streamlit run app/streamlit_app.py`).
- Neon (online): buat *branch* basis data `uji-dosen` dan jalankan aplikasi uji yang memakai branch itu
  (DEPLOY_STREAMLIT_NEON.md bagian 6). Alternatif PostgreSQL lain: *secret* `DATABASE_SCHEMA=uji` membuat seluruh
  tabel kosong di skema `uji`; hapus untuk kembali ke data semula.

Siapkan juga berkas uji: data akademik anonim (CSV) atau xlsx sistem akademik, dan bila ada riwayat KRS
alumni bernilai. Semuanya mengikuti templat di bagian 7.

## 6. Langkah 1 sampai 12 dari nol

Nomor langkah sama dengan tabel di tab Mulai di sini. Bila memakai paket data uji (bagian 4), langkah 3 sampai 8
dan 11 sudah dikerjakan otomatis.

### Langkah 1. Konfigurasi `.env` / app settings

| Variabel | Fungsi | Bila kosong |
|---|---|---|
| `CRS_HASH_ADMIN`, `CRS_HASH_DPA` | Hash kata sandi peran | Peran tidak dapat dibuka |
| `ANON_SALT` | Salt anonimisasi NIM saat impor xlsx | Impor xlsx ditolak; impor CSV anonim tetap bisa |
| `OPENAI_API_KEY` (atau kunci provider lain) | Penilaian Bloom oleh LLM | Penilaian LLM gagal; pakai cadangan atau provider `mock` |
| `LLM_PROVIDER`, `LLM_MODEL` | Provider & model, mis. `openai` / `gpt-6-luna` | Memakai nilai di `config.yaml` |
| `DATABASE_URL` | PostgreSQL (cloud) | Memakai SQLite `data/crs.db` |

Gunakan `ANON_SALT` yang **sama** dengan data sebelumnya agar ID anonim mahasiswa konsisten antarperiode.

### Langkah 2. Dataset Kurikulum 2024

Tab **Kurikulum**. Periksa angka ringkas: 72 mata kuliah, 38 prasyarat, 9 tema pilihan, SKS Jalur A = 145,
SKS Jalur B sampai F = 147. Dataset divalidasi setiap kali dimuat (prasyarat asiklik, pita IP lengkap).

Di tab yang sama administrator dapat:
- melengkapi CPMK yang ambigu lewat **Simpan override CPMK** (rumusan asli tetap tersimpan; alasan wajib);
- menambah atau menghapus prasyarat lewat **Simpan override prasyarat** (ditolak bila membuat siklus).

Override disimpan terpisah; dataset tidak berubah. Setelah mengubah CPMK, nilai ulang Bloom untuk MK itu.

### Langkah 3. Impor data akademik (KRS + IP/IPK)

Tab **Data akademik** → *Impor data akademik*. Pilih format:

- **xlsx akademik**: berkas ekspor sistem akademik apa adanya (berisi NIM). Sheet KRS dikenali dari kolom
  `KODE MK`, sheet IP dari kolom `IP BEBAN`; periode dibaca dari nama sheet (harus memuat `Ganjil`/`Genap` dan
  tahun). NIM langsung di-*hash* di memori dan tidak pernah disimpan. Templat: `templat_data_akademik.xlsx`.
- **CSV anonim**: `templat_krs_akademik.csv` (wajib) dan `templat_ip_semester.csv` (opsional) yang sudah
  memakai `id_anon`. Berkas yang masih berkolom NIM ditolak.

Setelah berkas dipilih sistem menampilkan hasil validasi (jumlah baris, mahasiswa, periode, kode MK yang tidak
dikenal). Klik **Simpan ke basis data**. Periode yang sama diganti; periode lain dipertahankan. Kode Kurikulum
2020 disetarakan otomatis lewat tabel penyetaraan.

### Langkah 4. Riwayat KRS multisemester

Data akademik hanya memuat satu periode tanpa nilai per MK, sedangkan simulasi dan ARM semester genap butuh
riwayat bernilai lintas semester. Dua pilihan di tab **Data akademik**:

- **Bangkitkan riwayat multisemester**: riwayat pelengkap yang dikalibrasi dari data akademik (IPK, SKS per
  semester), menempuh kurikulum lewat aturan yang sama dengan aplikasi. Seed tetap sehingga hasilnya dapat
  diulang. Asal datanya tercatat di basis data dan log.
- **Unggah riwayat alumni** (*Simpan sebagai histori_krs.csv*): riwayat alumni asli, format
  `templat_histori_krs.csv`. Menggantikan riwayat yang ada.

### Langkah 5. Preprocessing

Tab **Preprocessing** → **Jalankan preprocessing**. Proses: anonimisasi, filter masa studi ≤ 8 semester,
normalisasi kode 2020 → 2024, status lulus, pembentukan transaksi (MK yang diambil per mahasiswa-semester), lalu
pembagian data latih:uji 80:20 per mahasiswa dengan seed tetap. Hasil yang diharapkan: jumlah transaksi latih
dan uji, transaksi per semester, proporsi baris terpetakan, dan daftar MK tanpa padanan.

Bila data di langkah 3 atau 4 berubah, halaman Administrator menampilkan peringatan *Data masukan berubah sejak
pipeline terakhir* dan langkah 5 sampai 11 harus dijalankan ulang.

### Langkah 6. ARM (FP-Growth)

Tab **ARM**. Atur `min_support`, `min_confidence`, `min_lift`, `max_len` (default dari `config.yaml`), centang
*Simpan parameter ini sebagai default* bila perlu, lalu **Jalankan FP-Growth**. Hasil: jumlah aturan per semester
dan 20 aturan teratas berdasarkan lift, ditampilkan dengan nama MK. Aturan yang konsekuennya MK wajib dibuang
agar pola trivial paket tidak mendominasi.

**Jalankan analisis sensitivitas** menghasilkan tabel jumlah aturan, rata-rata confidence, dan lift untuk grid
support × confidence, sebagai dasar pemilihan ambang.

### Langkah 7. Penilaian Bloom per CPMK oleh LLM

Tab **Beban kognitif (LLM)** → *Jalankan penilaian Bloom baru*. Pilih provider dan model, centang
*Lanjutkan* agar MK yang sudah dinilai dilewati, centang pernyataan biaya API, lalu **Jalankan penilaian
Bloom**. Satu pengulangan = satu permintaan berisi seluruh MK; dilakukan 3 pengulangan. Yang dikirim ke LLM
hanya nama, deskripsi, CPMK, dan Sub-CPMK (tanpa data mahasiswa).

Alternatif tanpa biaya:
- **Pulihkan rating LLM** dari cadangan (bagian 9): hasil `gpt-6-luna` yang sama dipakai lagi.
- Provider `mock`: heuristik kata kerja operasional, hanya untuk uji alur (ditandai *BUKAN hasil penelitian*).

Hasil yang diharapkan: tabel CL per MK, konsistensi antarpengulangan per CPMK (sepakat penuh, selisih ≤ 1
level), dan riwayat permintaan batch (token, durasi, status).

### Langkah 8. Skor CL dan θ

Klik **Hitung ulang CL & θ dari hasil tersimpan** (tidak memanggil API). CL per MK = agregasi level Bloom CPMK
kognitif; θ = persentil ke-75 beban semester KRS data latih. Nilai θ tampil di sidebar dan tab Parameter.
Langkah ini juga dijalankan otomatis setelah penilaian LLM atau pemulihan cadangan.

### Langkah 9 (opsional). Penilaian pakar

1. Unduh **formulir penilaian pakar (terisi daftar CPMK)**.
2. Setiap pakar mengisi `kode_pakar` (mis. P1) dan `bloom` (C1 sampai C6, A untuk afektif, kosong bila belum
   dinilai) secara independen, tanpa melihat hasil LLM.
3. Tab Beban kognitif → *Impor penilaian pakar* → pilih berkas → **Impor penilaian pakar**. Pakar dengan kode
   yang sama diganti seluruhnya.

### Langkah 10 (opsional). Evaluasi RM1

**Hitung ulang evaluasi** menampilkan ICC(A,1) LLM vs rata-rata pakar, antarpakar, antarpengulangan LLM, dan
baseline heuristik, beserta MAE, Bland-Altman, dan uji t. Hasil resmi dihitung di JASP dari
**Unduh rm1_skor_mk.csv**; **Unduh bloom_perbandingan.csv** untuk analisis tingkat CPMK.

### Langkah 11. Simulasi & validasi

Tab **Simulasi & validasi**:
- **Jalankan validasi KRS**: rekomendasi semester berjalan dibandingkan dengan KRS nyata mahasiswa data uji
  (Jaccard, presisi, proporsi KRS nyata di atas θ). **Unduh validasi_krs.csv** untuk JASP.
- **Jalankan simulasi komparasi**: riwayat data uji dipotong di semester k, lalu mode ARM dan Hybrid
  dibandingkan (SD beban, beban puncak, SD SKS, Jaccard, kepatuhan aturan). **Unduh simulasi_metrik.csv** berformat
  berpasangan (satu baris per profil) untuk uji berpasangan di JASP.

Hasil yang diharapkan: kedua mode lolos AD-02, AD-03, dan AD-04 pada semua profil, dan mode Hybrid tidak
pernah melampaui θ.

### Langkah 12. Uji alur Mahasiswa → DPA

1. Peran **Mahasiswa**: isi jalur, agama, riwayat nilai (atau unggah `templat_riwayat_nilai.csv` yang sudah
   diisi), semester berjalan, dan IP. Atur MK pilihan per tema bila perlu, lalu **Bangkitkan Rencana**.
2. Coba geser MK di papan *Sesuaikan rencana* (lihat skenario U-09 sampai U-11).
3. Peran **DPA**: pilih rencana, isi validasi akhir dan catatan, **Simpan validasi & catatan**.
4. Kembali ke peran Mahasiswa: *Cek status validasi DPA* dengan kode rencana.

### Parameter (kapan saja)

Tab **Parameter** → **Simpan parameter**: tafsiran prasyarat, mode penawaran MK pilihan, nilai lulus minimum,
θ manual atau otomatis, persentil θ, metode beban, sumber CL, dan batas SKS per IP. Perubahan disimpan terpisah
dari `config.yaml`; **Kembalikan ke config.yaml** membatalkannya.

### Log (kapan saja)

Tab **Log**: setiap proses (impor, preprocessing, ARM, penilaian LLM, simulasi, pemulihan, atur ulang) tercatat
dengan parameter, durasi, status, dan ringkasan.

## 7. Templat dan kamus data

Semua templat berisi baris nama kolom saja. Aturan umum:
- Simpan sebagai CSV. Berkas dari Excel berlocale Indonesia (pemisah `;`, koma desimal `3,25`) dikenali
  otomatis, begitu juga pemisah `,` dengan titik desimal.
- Jangan mengubah nama kolom. Kolom tambahan diabaikan, kecuali kolom NIM yang menyebabkan berkas ditolak.
- Kode MK boleh kode Kurikulum 2020 atau 2024 (disetarakan otomatis), kecuali formulir pakar dan cadangan
  rating yang memakai `id_cpmk` dataset 2024.

### templat_krs_akademik.csv (KRS akademik, CSV anonim)

Diunggah di: Data akademik → Impor data akademik → CSV anonim → *histori_krs.csv*.

| Kolom | Wajib | Keterangan |
|---|---|---|
| `id_anon` | ya | ID mahasiswa yang sudah dianonimkan (mis. M1A2B3C4D). Jangan isi NIM. |
| `angkatan` | ya | Tahun angkatan, mis. 2024. |
| `semester_ke` | ya | Semester mahasiswa saat mengambil MK (1, 2, 3, ...). |
| `periode` | ya | Periode KRS, mis. `Ganjil 2026/2027`. |
| `kode_mk` | ya | Kode MK. |
| `nama_mk` | opsional | Nama MK, hanya untuk dibaca. |
| `sks` | ya | SKS MK (bilangan bulat ≥ 1). |
| `kurikulum_kode` | opsional | Kode kurikulum dari sistem akademik. |

### templat_ip_semester.csv (IP/IPK semester, opsional)

Diunggah bersama KRS di tempat yang sama (*ip_semester.csv*). Semua kolom wajib terisi bila berkas diunggah.

| Kolom | Keterangan |
|---|---|
| `id_anon`, `angkatan`, `semester_ke` | Sama dengan KRS. |
| `periode` | Periode nilai, mis. `Genap 2025/2026`. |
| `ip_beban` | IP semester atas seluruh SKS yang diambil (0 sampai 4). |
| `sks_beban` | SKS yang diambil pada semester itu. |
| `ipk_beban` | IPK kumulatif atas seluruh SKS yang pernah diambil. |
| `sksk_beban` | SKS kumulatif yang pernah diambil. |
| `ip_lulus` | IP semester atas SKS yang lulus. |
| `sks_lulus` | SKS lulus pada semester itu. |
| `ipk_lulus` | IPK kumulatif atas SKS yang lulus. |
| `sksk_lulus` | SKS kumulatif yang lulus. |
| `sks_tidak_lulus` | `sks_beban` dikurangi `sks_lulus`. |

### templat_data_akademik.xlsx (ekspor sistem akademik, ber-NIM)

Dua sheet. Nama sheet harus memuat `Ganjil`/`Genap` dan tahun, karena periode dibaca dari nama sheet.
- Sheet `KRS Ganjil 2026`: `NIM`, `ANGKATAN`, `SEMESTER`, `KODE MK`, `NAMA MK`, `SKS`, `KURIKULUM`.
- Sheet `IP dan IPK Genap 2025`: `NIM`, `ANGKATAN`, `SEMESTER`, `IP BEBAN`, `SKS BEBAN`, `IPK BEBAN`,
  `SKSK BEBAN`, `IP LULUS`, `SKS LULUS`, `IPK LULUS`, `SKSK LULUS`.

Kolom lain (program studi, kelas) dibuang saat impor. Membutuhkan `ANON_SALT`.

### templat_histori_krs.csv (riwayat KRS multisemester alumni)

Diunggah di: Data akademik → *Unggah riwayat alumni*. Satu baris per MK per semester per alumni. Semua kolom
wajib.

| Kolom | Keterangan |
|---|---|
| `id_anon` | ID alumni anonim (boleh juga kolom `nim`, yang di-*hash* saat preprocessing). |
| `angkatan` | Tahun angkatan. |
| `semester_ke` | Semester saat MK diambil. |
| `kode_mk` | Kode MK (2020 atau 2024). |
| `sks` | SKS MK. |
| `nilai_huruf` | A, B+, B, C+, C, D+, D, atau E. |
| `ip_semester` | IP semester tersebut (0 sampai 4). |
| `kurikulum` | Kurikulum kode MK: 2020 atau 2024. |
| `masa_studi_semester` | Lama studi alumni sampai lulus. Hanya alumni ≤ 8 semester yang ditambang. |

### templat_penilaian_pakar.csv (penilaian Bloom oleh pakar)

Diunggah di: Beban kognitif (LLM) → *Impor penilaian pakar*. Lebih praktis memakai *formulir penilaian pakar
(terisi daftar CPMK)*; kolom tambahan `nama_mk` dan `teks` di formulir itu hanya bantuan membaca.

| Kolom | Wajib | Keterangan |
|---|---|---|
| `id_cpmk` | ya | ID CPMK, mis. `CIE62303-CPMK04`. |
| `kode_mk` | ya | Kode MK pemilik CPMK (harus cocok dengan `id_cpmk`). |
| `kode_pakar` | ya | Kode penilai, mis. P1 (tanpa nama). |
| `bloom` | ya (boleh kosong) | C1 sampai C6, A untuk afektif, kosong = belum dinilai. |
| `catatan` | opsional | Catatan pakar. |

### templat_rating_llm.csv (cadangan rating Bloom LLM)

Format berkas **Unduh cadangan rating LLM**; diunggah kembali di Mulai di sini → **Pulihkan rating LLM**.
Biasanya tidak diisi tangan.

| Kolom | Wajib | Keterangan |
|---|---|---|
| `sumber` | ya | `llm` (atau `llm_mock` untuk uji coba). |
| `id_cpmk`, `kode_mk` | ya | Seperti formulir pakar. |
| `penilai` | ya | Nomor pengulangan LLM (1, 2, 3). |
| `model` | ya | Nama model, mis. gpt-6-luna. |
| `versi_prompt` | ya | Versi prompt, mis. bloom-v3. |
| `bloom` | ya | C1 sampai C6 atau A. |
| `status` | ya | `ok` atau `perlu_tinjau_manual`. |
| `ranah` | opsional | kognitif, afektif, atau afektif+kognitif. |
| `timestamp` | opsional | Waktu penilaian (ISO 8601). |
| `penalaran`, `alasan`, `keyakinan`, `provider`, `model_pelayan`, `temperature`, `thinking`, `ringkasan_thinking`, `token_masuk`, `token_keluar`, `percobaan`, `catatan` | opsional | Metadata penilaian. |

### templat_riwayat_nilai.csv (riwayat nilai satu mahasiswa)

Diunggah di: Mahasiswa → *Isi otomatis dari berkas riwayat nilai*.

| Kolom | Keterangan |
|---|---|
| `kode_mk` | Kode MK yang sudah ditempuh (kode 2020 disetarakan otomatis). |
| `nilai_huruf` | A, B+, B, C+, C, D+, D, atau E. |

## 8. Skenario pengujian untuk dosen

Jalankan berurutan setelah kondisi kosong disiapkan (bagian 5). Kolom terakhir dapat dipakai sebagai lembar
centang pengujian.

| ID | Skenario | Langkah | Hasil yang diharapkan | ✔ |
|---|---|---|---|---|
| U-01 | Sistem kosong | Buka Administrator → Mulai di sini | Langkah 3 sampai 8, 11, dan 12 ⬜ (9 dan 10 ➖ opsional; 7 tetap ✅ bila rating LLM dipertahankan); kotak *Langkah berikutnya* menunjuk langkah 3 | |
| U-02 | Rencana tanpa data KRS | Mahasiswa: Jalur A, agama, semester 1, IP 3,20 → Bangkitkan Rencana | Rencana tersusun dari struktur kurikulum; asumsi menyebut ARM belum aktif; validasi lolos | |
| U-03 | Templat kosong | Unduh `templat_krs_akademik.csv`, buka di Excel | Hanya baris nama kolom | |
| U-04 | Berkas ber-NIM ditolak | Tambah kolom `nim` pada CSV KRS, unggah sebagai CSV anonim | Pesan *Ditolak: data masih memuat kolom NIM* | |
| U-05 | Kolom kurang ditolak | Hapus kolom `periode`, unggah | Pesan *KRS: kolom wajib tidak ada: ['periode']* | |
| U-06 | Impor data sah | Unggah berkas uji yang benar → Simpan ke basis data | Ringkasan periode & jumlah mahasiswa; langkah 3 ✅ | |
| U-07 | Pipeline dari nol | Langkah 4, 5, 6, 7 (atau pulihkan cadangan), 8 | Langkah 4 sampai 8 ✅; sidebar menampilkan jumlah aturan dan θ | |
| U-08 | Data berubah → basi | Impor ulang data akademik dengan isi berbeda | Peringatan *Data masukan berubah*; langkah 5 dan 6 ⚠️ sampai dijalankan ulang | |
| U-09 | Geser MK sah | Rencana U-02 (semester 1, tanpa riwayat): seret Tugas Akhir/Skripsi (UBU60001) dari S7 ke S8 | Diterima; rencana tersimpan; kotak *Disesuaikan mahasiswa* muncul | |
| U-10 | Geser MK melanggar penawaran | Rencana yang sama: seret PLP 1 (CIE61111) dari S7 ke S8 | Ditolak dengan alasan AD-04 (PLP hanya ditawarkan di semester ganjil; PLP 2 ikut dicoba dipindah); papan kembali | |
| U-11 | Geser prasyarat | Rencana yang sama: seret Jaringan Komputer (CIE61206, S3) ke S5 | Ditolak dengan alasan AD-03 pada Keamanan Siber (CSD60709, S4) yang membutuhkan Jaringan Komputer | |
| U-12 | Pilihan tema | Atur MK pilihan per tema: Tema A → Paradigma Berpikir Sistem dan Desain | Rencana memuat MK itu, bukan MK Tema A lainnya, pada kedua mode | |
| U-13 | Validasi DPA | DPA menyetujui rencana; mahasiswa cek dengan kode rencana | Status *disetujui* dan catatan DPA tampil | |
| U-14 | Simulasi | Jalankan simulasi komparasi | Kedua mode lolos AD-02/03/04; unduh CSV berpasangan | |
| U-15 | Penilaian pakar | Isi formulir pakar untuk beberapa CPMK, impor | Langkah 9 ✅; ICC tampil di RM1 setelah *Hitung ulang evaluasi* | |
| U-16 | Kata sandi | Masuk Administrator dengan sandi salah 5 kali | Peran terkunci untuk sesi itu; percobaan tercatat | |
| U-17 | Atur ulang | Unduh cadangan, ketik ATUR ULANG, atur ulang | Seperti U-01: langkah 3 sampai 8, 11, 12 kembali ⬜ (langkah 7 tetap ✅ bila rating LLM dipertahankan) | |
| U-18 | Pulihkan rating | Unggah cadangan → Pulihkan rating LLM | Langkah 7 dan 8 ✅ tanpa memanggil API | |
| U-19 | Siapkan dari paket | Setelah U-17 tanpa mempertahankan rating: unggah zip paket data uji → Siapkan sistem dari paket data uji | Langkah 3 sampai 8 dan 11 ✅ dalam sekitar 1 menit; θ = 75 dan 10 aturan ARM (sama dengan naskah); sidebar menampilkan θ dan jumlah aturan | |

**Urutan sesi pengujian singkat (± 20 menit) dengan paket data uji:** U-17 (atur ulang, tanpa mempertahankan
rating) → U-01 → U-02 → U-19 (siapkan dari paket) → U-09, U-10, U-11 → U-12 → U-13 → U-14. Skenario validasi
berkas (U-03 sampai U-06) dan pakar (U-15) dapat ditambahkan bila waktu memungkinkan.

## 9. Cadangan dan atur ulang

Mulai di sini → *Cadangan & atur ulang*:

| Tombol / isian | Fungsi |
|---|---|
| **Unduh paket data uji (zip)** | KRS akademik anonim, IP/IPK, rating LLM, dan riwayat multisemester (apa adanya) dari sistem saat ini, untuk Cara cepat (bagian 4). |
| **Unduh cadangan rating LLM** | Semua rating Bloom LLM tersimpan (semua model dan versi prompt) dalam satu CSV. |
| **Berkas cadangan rating LLM (CSV)** + **Pulihkan rating LLM** | Memvalidasi berkas (kolom, `id_cpmk`, label, baris ganda), mengganti rating dengan model dan versi prompt yang sama, lalu menghitung ulang CL dan θ. Model, versi prompt, dan waktu penilaian asli tetap tercatat. |
| **Pertahankan rating Bloom LLM** | Bila dicentang, rating LLM tidak ikut dihapus saat atur ulang (rating pakar tetap dihapus). |
| **Ketik ATUR ULANG untuk konfirmasi** + **Atur ulang sistem ke kondisi awal** | Mengosongkan data akademik, riwayat, transaksi, aturan, CL, θ, hasil simulasi, rencana, override, dan parameter UI. Dataset kurikulum, log proses, dan log akses tidak dihapus. Tidak dapat dibatalkan. |

Di cloud, Neon menyimpan riwayat perubahan basis data sehingga dapat dipulihkan ke titik waktu sebelumnya
(*point-in-time restore*, jangka waktunya sesuai paket) bila atur ulang dilakukan tanpa sengaja. Untuk sesi uji,
lebih aman memakai branch terpisah (bagian 5).

## 10. Privasi dan keamanan data

- NIM tidak pernah disimpan. xlsx ber-NIM dianonimkan di memori saat impor; CSV harus sudah anonim.
- Yang dikirim ke LLM hanya teks kurikulum (nama MK, deskripsi, CPMK, Sub-CPMK), tidak ada data mahasiswa.
- Rencana mahasiswa disimpan tanpa nama/NIM; label rencana jangan diisi data pribadi.
- Kunci API dan hash kata sandi hanya ada di `.env` atau *Secrets* Streamlit Cloud, tidak di repositori.

## 11. Pemecahan masalah

| Gejala | Penyebab & solusi |
|---|---|
| Impor xlsx: *ANON_SALT di .env belum diisi* | Isi `ANON_SALT` (langkah 1), atau pakai CSV anonim. |
| Impor xlsx: *periode tidak dapat dibaca dari nama sheet* | Ganti nama sheet, mis. `KRS Ganjil 2026`. |
| *Kode MK tidak dikenal Kurikulum 2024* | Kode tidak ada di tabel penyetaraan; periksa salah ketik, atau biarkan (dilaporkan, tidak dibuang dari impor). |
| CSV dari Excel ditolak karena kolom tidak ditemukan | Pastikan baris pertama tetap nama kolom templat dan tidak ada baris judul tambahan. |
| Langkah 6 ⚠️ setelah preprocessing | Jalankan ulang FP-Growth agar aturan sesuai data terbaru. |
| *θ belum tersedia* | Selesaikan langkah 7 dan 8. |
| Penilaian LLM gagal autentikasi | Periksa kunci API provider; atau pulihkan cadangan rating. |
| Tidak ada aturan ARM | Turunkan `min_support`/`min_confidence`; lihat analisis sensitivitas. |
