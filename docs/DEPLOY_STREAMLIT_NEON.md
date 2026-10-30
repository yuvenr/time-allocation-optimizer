# Deploy ke Streamlit Community Cloud + Neon (PostgreSQL)

Panduan ini memasang aplikasi dari repositori GitHub `crs_pti` ke Streamlit Community Cloud, dengan basis data
PostgreSQL di Neon. Kedua layanan memiliki paket gratis yang cukup untuk prototipe ini (basis data ± 5 MB).

```
GitHub crs_pti (privat) ──► Streamlit Community Cloud: app/streamlit_app.py (Python 3.12)
                                   │  Secrets: DATABASE_URL, hash kata sandi, ANON_SALT, (kunci API)
                                   ▼
                            Neon PostgreSQL: semua tabel + salinan berkas riwayat & parameter (tabel berkas)
```

Streamlit Community Cloud **tidak menyimpan berkas** yang ditulis aplikasi: setiap kali aplikasi dimulai ulang,
disk kembali seperti isi repositori. Karena itu riwayat KRS multisemester dan parameter UI disalin ke tabel
`berkas` di Neon dan dipulihkan otomatis saat aplikasi mulai (`src/berkas.py`). Log berkas (`logs/`) memang hilang
saat dimulai ulang; log proses utama tetap ada di tabel `log_proses`.

## 1. Neon

1. Daftar di neon.tech (bisa memakai akun GitHub) → **Create project**.
   - Nama: `crs-pti`; PostgreSQL 16 atau lebih baru.
   - **Region: AWS US East (N. Virginia)**. Server Streamlit Community Cloud berada di Amerika Serikat; region
     Asia akan memperlambat setiap halaman karena aplikasi menjalankan banyak kueri kecil.
2. Dashboard → **Connect** → matikan **Connection pooling**, lalu salin *connection string* **langsung** (host
   TANPA `-pooler`). Bentuknya:
   `postgresql://USER:PASSWORD@ep-xxxx.us-east-1.aws.neon.tech/neondb?sslmode=require`.
   Aplikasi memakai kumpulan koneksinya sendiri (koneksi dipakai ulang antarklik), sehingga koneksi langsung paling
   cepat. URL `-pooler` tetap didukung (aman dari kebocoran pengaturan antarklien), tetapi setiap pemanggilan
   basis data butuh satu perjalanan jaringan tambahan.
3. Simpan string itu di pengelola kata sandi. Jangan ditulis di berkas yang di-commit.

Catatan paket gratis: komputasi Neon tidur setelah beberapa menit tanpa kueri; kueri pertama sesudahnya butuh
beberapa detik.

## 2. Pindahkan data dari komputer lokal ke Neon

Jalankan di **folder pengembangan** (`D:\Code\CRS`), yang berisi `data/crs.db` dan `data/raw/`:

```powershell
# 1) salin riwayat multisemester + penanda asal data ke tabel berkas (SQLite lokal), byte identik
python -m src.berkas simpan
# 2) pindahkan seluruh tabel ke Neon; laporan jumlah baris per tabel, dibatalkan bila tidak cocok
$env:DATABASE_URL = "postgresql://...-pooler...neon.tech/neondb?sslmode=require"
python -m scripts.migrasi_sqlite_ke_postgres
# 3) periksa: daftar berkas yang tersimpan di Neon
python -m src.berkas daftar
Remove-Item Env:DATABASE_URL
```

Bila Neon sudah berisi data dan ingin diganti: tambahkan `--timpa` pada langkah 2.

## 3. GitHub

1. github.com → **New repository** → nama `crs_pti`, **Private**, tanpa README/gitignore (sudah ada).
2. Di `D:\Code\crs_pti`:
   ```powershell
   git remote add origin https://github.com/<akun>/crs_pti.git
   git push -u origin main
   ```
3. Tab **Actions** menjalankan seluruh tes otomatis setiap *push* (`.github/workflows/tests.yml`).

## 4. Streamlit Community Cloud

1. share.streamlit.io → masuk dengan GitHub → izinkan akses ke repositori privat bila diminta.
2. **Create app** → pilih sumber dari GitHub:
   - Repository `<akun>/crs_pti`, branch `main`, main file path `app/streamlit_app.py`.
   - App URL: mis. `crs-pti`.
   - **Advanced settings** → Python version **3.12**; **Secrets**: salin isi `.streamlit/secrets.toml.example`
     lalu isi nilainya (connection string Neon, hash kata sandi dan `ANON_SALT` dari `.env` lokal).
3. **Deploy**. Pemasangan pertama beberapa menit.
4. Akses aplikasi: *Settings → Sharing*. Aplikasi dari repositori privat dapat dibatasi hanya untuk orang yang
   diundang (alamat email); undang dosen pembimbing/penguji di sini. Paket gratis membatasi jumlah aplikasi
   privat; periksa batas terbaru di dokumentasi Streamlit Community Cloud. Halaman Administrator dan DPA tetap
   berkata sandi walaupun aplikasi dibuka untuk umum.

## 5. Verifikasi

1. Sidebar *Status sistem*: `Basis data: PostgreSQL · ep-…neon.tech…`, CL `llm`, θ = 75.
2. Masuk sebagai Administrator → tab **Mulai di sini**: langkah 2 sampai 8 dan 11 ✅, tidak ada peringatan
   *Data masukan berubah*.
3. **Reboot** aplikasi (menu ⋮ → Reboot), buka lagi: status tetap sama (riwayat dipulihkan dari Neon).
4. Peran Mahasiswa: bangkitkan rencana, lalu cek rencana muncul di halaman DPA.

## 6. Sesi pengujian dari nol tanpa mengganggu data utama

Gunakan **branch** Neon (salinan basis data instan):
1. Neon → **Branches** → *Create branch* `uji-dosen` dari `main` → salin *connection string*-nya.
2. Streamlit → buat aplikasi kedua dari repositori yang sama (mis. URL `crs-pti-uji`) dengan Secrets yang sama
   kecuali `DATABASE_URL` = string branch `uji-dosen`. Bila batas aplikasi privat tidak memungkinkan aplikasi
   kedua, ganti sementara `DATABASE_URL` aplikasi utama ke branch uji, lalu kembalikan setelah sesi.
3. Di aplikasi uji: Mulai di sini → atur ulang → Cara cepat dengan paket data uji (panduan administrator,
   bagian 4). Aplikasi utama tidak terpengaruh.
4. Setelah sesi, branch dapat dihapus atau di-*reset* dari `main`.

## 7. Memperbarui aplikasi

Semua pengembangan tetap di `D:\Code\CRS`. Untuk memperbarui versi online:

```powershell
cd D:\Code\CRS
python -m scripts.sinkron_deploy           # salin berkas yang diizinkan ke D:\Code\crs_pti
cd D:\Code\crs_pti
git add -A; git commit -m "..."; git push   # Streamlit Cloud memasang ulang otomatis
```

Bila skema basis data berubah, aplikasi memigrasikan Neon secara otomatis saat pertama kali dibuka.

## Pemecahan masalah

| Gejala | Penyebab & solusi |
|---|---|
| `could not connect to server` / timeout pertama | Neon sedang bangun; muat ulang. Periksa `DATABASE_URL` dan `sslmode=require`. |
| `prepared statement ... does not exist` | Versi lama aplikasi; pastikan kode terbaru (koneksi tanpa *prepared statement*). |
| ARM/θ "belum dikalibrasi" setelah reboot | Tabel `berkas` kosong: jalankan langkah 2.1 lalu `python -m src.berkas simpan` dengan `DATABASE_URL` Neon. |
| Halaman lambat | Pastikan `DATABASE_URL` memakai koneksi langsung (tanpa `-pooler`) dan region Neon US East. Klik pertama setelah Neon tidur (± 5 menit tanpa akses) memang lebih lama beberapa detik. |
| Aplikasi "is asleep" | Paket gratis menidurkan aplikasi yang lama tidak dibuka; klik *Wake up* dan tunggu ± 1 menit. Buka sebelum sesi dengan dosen. |
| Peran Administrator tidak bisa dibuka | `CRS_HASH_ADMIN` di Secrets kosong/salah salin. |
