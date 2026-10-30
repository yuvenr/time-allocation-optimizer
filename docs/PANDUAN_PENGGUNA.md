# Panduan Pengguna: Sistem Rekomendasi Rencana Studi PTI UB

Panduan ini menjelaskan setiap halaman, tombol, dan isian pada aplikasi. Pilih peran di **sidebar kiri**
(Mahasiswa, DPA, Administrator). Halaman DPA dan Administrator meminta kata sandi.

**Sidebar**
- **Peran**: memilih halaman yang dibuka.
- **Keluar**: muncul setelah masuk sebagai DPA/Administrator; mengakhiri sesi peran itu.
- **Status sistem**: ringkasan data yang sedang dipakai: periode KRS akademik, ketersediaan riwayat
  multisemester, jumlah aturan asosiasi, sumber skor beban kognitif (CL), nilai θ, dan model LLM.

## Mahasiswa: menyusun rencana studi

Tujuan: mendapatkan rancangan rencana studi dari semester berjalan sampai lulus, yang sudah diperiksa
terhadap aturan akademik. Rencana tetap harus divalidasi DPA.

**Panel di bagian atas**
| Isian / tombol | Fungsi |
|---|---|
| *Cek status validasi DPA* → **Kode rencana** | Tempel kode rencana yang Anda terima setelah membangkitkan rencana untuk melihat status (*disetujui* / *perlu revisi*) dan catatan DPA. |
| *Isi otomatis dari berkas riwayat nilai* → **Riwayat nilai** + **Terapkan riwayat** | Unggah CSV berkolom `kode_mk,nilai_huruf` (kode Kurikulum 2020 otomatis disetarakan). Tabel riwayat nilai terisi otomatis. Tidak perlu NIM/nama. |
| *Isi otomatis dari profil alumni data uji* → **Alumni**, **Potong riwayat di semester k**, **Terapkan** | Untuk uji coba/demo: mengisi riwayat dari satu alumni data uji sampai semester k, sehingga rencana dimulai di semester k+1. |
| *Simulasi mahasiswa berisiko studi molor (> 8 semester)* → **Skenario**, **Jalankan simulasi skenario** | Untuk demo: memilih profil fiktif yang bermasalah (MK prasyarat tidak lulus, IP rendah, tertinggal paket, tunggakan di semester akhir). Sistem membandingkan perkiraan semester lulus *pola konvensional* (mengambil paket semester berjalan dulu, MK tertunda menyusul bila SKS cukup) dengan rekomendasi sistem, dengan aturan akademik yang sama. Profil otomatis diisikan ke formulir; klik **Bangkitkan Rencana** untuk rencana lengkap. |
| **Tutup hasil simulasi** | Menyembunyikan kotak hasil simulasi skenario. |

**Profil akademik**
| Isian | Fungsi |
|---|---|
| **Jalur kurikulum** | Jalur A (reguler) atau B–F (merdeka belajar). Semester 1–5 sama untuk semua jalur. |
| **Riwayat nilai** (tabel) | Tambahkan baris: pilih mata kuliah dan nilai akhir terbaik. Sistem menghitung MK lulus, SKS lulus, IPK, dan SKS bernilai D/D+. MK yang belum lulus (E) akan diprioritaskan bila wajib. |
| **Agama** | Menentukan MK agama yang direncanakan, dan menyesuaikan baris MK agama pada tabel riwayat nilai (mis. riwayat berisi Agama Katolik, dipilih Islam → baris diganti Agama Islam dengan nilai yang sama; baris agama ganda digabung dengan nilai terbaik). Sistem tidak pernah menebak agama. |
| **Semester berjalan** | Semester pertama yang akan direncanakan. |
| **IP semester sebelumnya** | Menentukan batas SKS per semester (pita IP) dan diasumsikan tetap untuk semester berikutnya. |
| **IPK** (opsional) | Muncul bila IPK tidak dapat dihitung dari tabel riwayat; dipakai untuk syarat Tugas Akhir (IPK ≥ 2,00). |
| **Tema pilihan yang dipenuhi lewat konversi**, **SKS konversi** | Hanya untuk Jalur B–F. Kosongkan tema bila belum tahu, sistem memakai asumsi yang dicatat. |
| **Atur MK pilihan per tema** | Membuka panel (*drawer*) berisi sembilan tema pilihan A–I. Untuk tiap tema pilih satu MK yang Anda minati atau biarkan *Rekomendasi sistem*. Setiap opsi menampilkan semester penawaran, prasyarat, dan skor CL. Nama tema ditampilkan dalam bahasa Indonesia (terjemahan dari nama resmi berbahasa Inggris). Tema yang sudah terpenuhi dari riwayat nilai ditampilkan tanpa pilihan. Pilihan berlaku sama untuk mode Hybrid dan ARM murni; bila MK yang dipilih tidak dapat dijadwalkan (penawaran/prasyarat/kapasitas SKS), rencana memberi peringatan. |
| **Simpan pilihan tema** | Menutup panel dan menyimpan pilihan; ringkasannya tampil di samping tombol. Klik **Bangkitkan Rencana** agar rencana memakai pilihan baru. |
| **Serahkan semua ke sistem** | Menghapus seluruh pilihan sehingga semua tema kembali dipilihkan sistem. |
| **Mode** | *Hybrid* (pola mata kuliah + batas beban kognitif θ), *ARM murni* (tanpa batas beban), atau *Bandingkan keduanya*. |
| **Label rencana** | Penanda bebas, mis. "rencana-A". Jangan isi nama/NIM. |
| **Bangkitkan Rencana** | Menyusun dan menyimpan rencana, lalu menampilkan kode rencana. Simpan kode itu untuk cek status DPA. |

**Membaca hasil**
- Angka ringkas: SKS direncanakan, total SKS akhir (harus ≥ 145), beban puncak, θ, dan status validasi.
- Kotak kuning = peringatan (mis. agama belum diisi, rencana melewati semester 8).
- **Asumsi rencana**: semua asumsi yang dipakai (IP tetap, MK dianggap lulus, sumber data pola, cara θ dihitung).
- **Grafik beban kognitif**: beban tiap semester; garis putus-putus = θ.
- **Tabel per semester** + *Alasan, sumber & jadwal per MK*: mengapa setiap MK direkomendasikan (prioritas,
  popularitas/aturan asosiasi, skor CL dan level Bloom per CPMK, jadwal PLP/PKL).
- **Sesuaikan rencana: geser MK antarsemester** berupa papan *drag & drop*. Seret MK (mis. PLP, PKM, Tugas Akhir yang
  sering direkomendasikan di semester 7) ke semester lain, termasuk satu semester tambahan sesudah semester
  terakhir. Setiap perubahan langsung diperiksa dengan aturan yang sama dengan penyusun rencana: prasyarat (MK
  prasyarat tidak boleh digeser melewati MK yang membutuhkannya), semester penawaran ganjil/genap, batas SKS pita
  IP, syarat Tugas Akhir (≥ 120 SKS & IPK ≥ 2,00), syarat PLP, paket semester 1–2, dan masa studi maksimal 14
  semester. Pemindahan yang melanggar **ditolak** dan papan kembali ke susunan sebelumnya beserta alasannya.
  PLP 1 dan PLP 2 selalu dipindahkan bersama. Pemindahan yang hanya menyimpang dari anjuran (mis. MK wajib
  tertunda tidak diambil di kesempatan pertama, atau beban semester melebihi θ pada mode Hybrid) tetap diterima
  dengan catatan untuk DPA. Rencana tersimpan otomatis dan status validasi DPA kembali *belum ditinjau*.
- *Pindahkan lewat menu* → **Mata kuliah**, **Ke semester**, **Pindahkan**: cara alternatif tanpa menyeret
  (mis. dengan papan ketik); daftar *Ke semester* hanya berisi semester yang memenuhi seluruh aturan.
- **Kembalikan ke rekomendasi sistem**: muncul setelah ada pemindahan; membatalkan semua pemindahan.
- *Disesuaikan mahasiswa*: daftar pemindahan dan perbandingan susunan rekomendasi sistem vs setelah
  disesuaikan (juga terlihat oleh DPA).
- **Status validasi aturan domain**: 16 aturan (AD-01..AD-16): ✅ lolos, ❌ dilanggar, ⚠️ perlu verifikasi manual DPA.
- **Unduh rencana (JSON)**: menyimpan salinan rencana lengkap.
- Mode *Bandingkan keduanya* menampilkan tab **Perbandingan** (selisih beban puncak, SD beban, SD SKS).

## DPA: meninjau dan memvalidasi rencana

| Isian / tombol | Fungsi |
|---|---|
| Tabel daftar rencana | Semua rencana tersimpan: waktu, mode, jalur, semester mulai, IP, lolos aturan, jumlah butir *perlu verifikasi*, jumlah pemindahan oleh mahasiswa (*Disesuaikan mahasiswa*), status DPA. |
| **Pilih rencana** | Membuka detail: IPK, SKS lulus, MK yang telah diambil beserta nilainya, rencana per semester, dan hasil pemeriksaan aturan. |
| **Validasi akhir DPA** | *disetujui* atau *perlu revisi*. |
| **Catatan bimbingan** | Masukan untuk mahasiswa (mis. verifikasi syarat PLP, tunda MK tertentu). |
| **Simpan validasi & catatan** | Menyimpan status dan catatan; mahasiswa dapat melihatnya lewat kode rencana. |

Rencana yang digeser mahasiswa menampilkan kotak *Disesuaikan mahasiswa*: daftar pemindahan dan susunan asli
rekomendasi sistem, sehingga DPA dapat menilai alasan penyimpangannya.

Butir berlabel ⚠️ *perlu verifikasi manual* adalah syarat yang isinya tidak ada di dokumen kurikulum
(pedoman PLP/PKL/Capstone/Tugas Akhir/PKM, konversi merdeka belajar), perlu dicek DPA.

## Administrator

Panduan lengkap Administrator (menyiapkan sistem dari nol, templat data, cadangan & atur ulang, skenario
pengujian) ada di [PANDUAN_ADMINISTRATOR.md](PANDUAN_ADMINISTRATOR.md). Ringkasan per tab di bawah ini.

## Administrator: tab Kurikulum

| Isian / tombol | Fungsi |
|---|---|
| **Berkas** | Menampilkan isi berkas dataset Kurikulum 2024 (hanya dibaca). |
| *CPMK yang perlu ditinjau* → **CPMK**, **Rumusan CPMK yang dipakai sistem**, **Alasan perubahan**, **Simpan override CPMK** | Melengkapi rumusan CPMK yang ambigu tanpa mengubah dataset (teks asli tetap tersimpan). Setelah itu jalankan ulang penilaian Bloom untuk MK tersebut. |
| **Hapus override CPMK** + **Hapus override CPMK terpilih** | Mengembalikan CPMK ke rumusan dataset. |
| *Override prasyarat* → **MK**, **MK prasyarat**, **Aksi** (tambah/hapus), **Alasan**, **Simpan override prasyarat** | Menambah/menghapus relasi prasyarat. Ditolak bila kode tidak dikenal, relasi tidak ada, atau membuat siklus. |
| **Hapus override prasyarat terpilih** | Membatalkan override prasyarat. |

## Administrator: tab Data akademik

| Isian / tombol | Fungsi |
|---|---|
| Ringkasan & *Pratinjau data anonim* | Periode KRS/IP, jumlah mahasiswa, mahasiswa per semester. Data yang tersimpan tidak memuat NIM. |
| *Impor data akademik* → **Format** | *xlsx akademik* (berkas dari akademik, NIM di-hash otomatis dengan `ANON_SALT`) atau *CSV anonim* (`histori_krs.csv` + `ip_semester.csv`). |
| **Berkas xlsx** / **histori_krs.csv**, **ip_semester.csv** | Unggah berkas; sistem memvalidasi kolom, rentang IP, dan kode MK sebelum menyimpan. |
| **Simpan ke basis data** | Menyimpan data (periode yang sama diganti). Setelah itu jalankan Preprocessing & ARM. |
| **Bangkitkan riwayat multisemester** | Membuat riwayat KRS multisemester untuk bagian yang belum ada di data akademik (semester genap, simulasi), dikalibrasi dari data akademik. Tidak pernah menimpa riwayat alumni yang diunggah. |
| *Unggah riwayat alumni* → **CSV histori KRS**, **Simpan sebagai histori_krs.csv** | Mengganti riwayat multisemester dengan riwayat alumni dari akademik. |

## Administrator: tab Preprocessing

| Tombol | Fungsi |
|---|---|
| **Jalankan preprocessing** | Mengubah data KRS menjadi transaksi (MK yang diambil per mahasiswa per semester), menyetarakan kode Kurikulum 2020 → 2024, membuang mahasiswa/alumni di atas 8 semester, dan membagi data latih:uji 80:20. Tabel *Transaksi per semester & sumber* menunjukkan asal data tiap semester. |

## Administrator: tab ARM

| Isian / tombol | Fungsi |
|---|---|
| **min_support**, **min_confidence**, **min_lift**, **max_len** | Ambang penambangan aturan asosiasi (FP-Growth). |
| **Simpan parameter ini sebagai default** | Menyimpan ambang ke `config.override.yaml`. |
| **Jalankan FP-Growth** | Menambang aturan dari transaksi latih. Hasil: ringkasan aturan per semester dan 20 aturan teratas menurut lift. |
| **Jalankan analisis sensitivitas** | Mencoba kombinasi support × confidence dan menampilkan jumlah aturan, rata-rata confidence, dan lift. |

## Administrator: tab Beban kognitif (LLM)

Beban kognitif (CL) tiap MK berasal dari level Bloom setiap CPMK yang dinilai LLM. **Hasil penilaian
tersimpan di basis data dan dipakai ulang**: tidak perlu memanggil API lagi kecuali CPMK berubah.

| Isian / tombol | Fungsi |
|---|---|
| Kotak hijau *Hasil LLM tersimpan dipakai* | Model, versi prompt, jumlah MK/CPMK/pengulangan, dan waktu penilaian yang sedang dipakai. |
| **Hitung ulang CL & θ dari hasil tersimpan** | Tanpa API: menyusun ulang konsensus Bloom → CL per MK → θ (mis. setelah mengubah parameter agregasi). |
| *Jalankan penilaian Bloom baru* → **Provider**, **Model** | Penyedia LLM (OpenAI/Claude/mock luring) dan nama model. |
| **Lanjutkan (lewati yang sudah ada)** | Hanya menilai (MK, pengulangan) yang belum ada untuk model & versi prompt yang sama. Matikan untuk menilai ulang semuanya. |
| **Saya paham ini memanggil API berbayar** | Konfirmasi wajib sebelum tombol penilaian aktif (kecuali mock). |
| **Jalankan penilaian Bloom** | Mengirim **satu permintaan berisi seluruh MK** per pengulangan (3 pengulangan = 3 permintaan). Jawaban JSON dipisah per MK; MK yang tidak lengkap dikirim ulang sendiri. Setelah selesai, CL, θ, dan evaluasi dihitung ulang. |
| Tabel **CL per MK** | Skor CL dan keterangannya (mis. MK yang seluruh CPMK-nya afektif memakai penilaian kognitif individual). MK tanpa CPMK (UBU60029/UBU60030) tidak pernah direkomendasikan. |
| *Konsistensi antarpengulangan* + **Tampilkan hanya CPMK yang tidak sepakat** | Seberapa sering ketiga pengulangan memberi label Bloom yang sama per CPMK; daftar CPMK yang berbeda untuk dibahas. |
| **Unduh konsistensi per CPMK (CSV)** | Tabel label tiap pengulangan per CPMK. |
| *Riwayat permintaan batch* | Satu baris per permintaan API: jumlah MK, status, token. |
| *Impor penilaian pakar* → **CSV penilaian pakar**, **Impor penilaian pakar** | Memasukkan formulir pakar yang sudah diisi (UC-10). |
| **Hitung ulang evaluasi** | Menghitung ulang ICC LLM vs pakar (RM1) dan kappa per CPMK. |
| **Buat templat penilaian pakar** | Membuat formulir isian pakar (satu baris per CPMK). |
| **Unduh bloom_perbandingan.csv**, **Unduh rm1_skor_mk.csv** | Data untuk analisis di JASP. |

## Administrator: tab Parameter

| Isian / tombol | Fungsi |
|---|---|
| **Tafsiran literal prasyarat '≥ E'** | Aktif = prasyarat cukup pernah ditempuh; nonaktif (bawaan) = harus lulus. |
| **Mode penawaran MK pilihan** | *paritas* (bawaan) atau *tabel11_ketat*. |
| **Nilai lulus minimum** | Batas nilai lulus (bawaan D). |
| **MK yang disarankan 'tambah' oleh 09 …** | Tafsiran penawaran Tugas Akhir/PKM/Capstone di semester berparitas lain. |
| **Penentuan θ**, **θ manual**, **Persentil otomatis** | θ dihitung otomatis dari persentil beban semester KRS, atau diisi manual. |
| **Metode beban semester** | *jumlah* (Σ CL × SKS) atau *rata_tertimbang*. |
| **Sumber CL**, **Agregasi CL per MK** | Sumber penilaian Bloom (llm, pakar, heuristik) dan cara menggabungkan level CPMK menjadi CL MK. |
| **Kecualikan PLP/PKL/PKM/Capstone dari beban** | Kegiatan lapangan tidak dihitung dalam beban kognitif semester. |
| Tabel batas SKS per IP | Menimpa pita SKS dataset per rentang IP (kosongkan = pakai dataset). |
| **Simpan parameter** | Menyimpan ke `config.override.yaml`. |
| **Kembalikan ke config.yaml** | Menghapus semua ubahan parameter dari UI. |

## Administrator: tab Simulasi & validasi

| Isian / tombol | Fungsi |
|---|---|
| **Jalankan validasi KRS** | Membandingkan rekomendasi semester berjalan (mode ARM dan hybrid) dengan KRS nyata mahasiswa data uji: Jaccard, presisi, recall, kepatuhan aturan, dan proporsi KRS nyata yang melampaui θ. |
| **Unduh validasi_krs.csv** | Data berpasangan untuk uji statistik di JASP. |
| **Semester potong k** | Titik potong riwayat data uji untuk simulasi. |
| **Jalankan simulasi komparasi** | Membandingkan kedua mode pada profil yang sama (SD beban, beban puncak, SD SKS, Jaccard, kepatuhan). |
| **Unduh simulasi_metrik.csv** | Data berpasangan `_arm`/`_hybrid` untuk JASP. |

## Administrator: tab Log

Menampilkan setiap proses (impor, preprocessing, ARM, penilaian LLM, simulasi) beserta parameter, durasi,
status, dan sumber data. Percobaan masuk peran tercatat terpisah di tabel `log_akses`.
