# Fixture tes buatan tangan

Folder ini berisi data kecil yang **ditulis tangan** untuk memverifikasi kode, dan hanya dipakai untuk itu.
Isinya **bukan data penelitian** dan tidak boleh dilaporkan sebagai temuan.

## `mini/raw/`
- `histori_krs.csv`: 6 mahasiswa **fiktif** (M01–M06), 54 baris KRS berkode **Kurikulum 2020**
  (kode lama diambil dari `data/kurikulum_2024/11_penyetaraan_mk_2020_2024.csv`). Kasus yang sengaja dimasukkan:
  - M06 bermasa studi 10 semester, sehingga harus tersaring.
  - `CIE60044` berjenis `dihapus`, sehingga harus terbuang dan tercatat.
  - M02 mendapat nilai D di `CIE61010`.
  - `COM60014` berubah dari 4 menjadi 5 SKS.
  - `CSD60004`, `CIE61010`, dan `CIE62020` berubah status wajib/pilihan.
- `SUMBER_DATA.txt`: penanda `fixture`. Dengan penanda ini, `status_data_krs` melaporkan data
  sebagai fixture, dan setiap rencana diberi catatan "BUKAN temuan penelitian".

Kurikulum yang dipakai adalah dataset asli `data/kurikulum_2024` (hanya dibaca). Hasil yang diharapkan
(jumlah baris per jenis perubahan, support, confidence, lift, θ, dan sebagainya) dihitung manual dan ditulis
di docstring tes yang memakainya. Tes-tes itu adalah fungsi bernama `*mini*` di
`tests/test_preprocessing.py`, `tests/test_arm.py`, `tests/test_planner.py`, dan
`tests/test_metrics_simulate.py`, dengan fixture `cfg_mini` di `tests/conftest.py`.
