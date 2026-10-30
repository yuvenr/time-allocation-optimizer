"""Daftar periksa kesiapan sistem "dari nol" dan atur ulang (iterasi 3; panduan: docs/PANDUAN_ADMINISTRATOR.md).

``periksa_kesiapan`` membaca keadaan basis data & berkas lalu mengembalikan langkah-langkah berurutan yang
harus dilakukan Administrator sejak sistem kosong sampai rencana divalidasi DPA, masing-masing dengan status
dan tempat melakukannya di aplikasi. ``atur_ulang`` mengembalikan sistem ke kondisi awal (untuk pengujian
dari nol), dengan opsi mempertahankan rating LLM agar API berbayar tidak perlu dipanggil ulang.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src import auth, db
import src.config as konfigurasi
from src.config import Config
from src.data_loader import load_kurikulum, sidik_raw, status_data_krs, transaksi_segar
from src.logutil import catat_proses

SELESAI, BELUM, OPSIONAL, PERHATIAN = "selesai", "belum", "opsional", "perlu perhatian"
IKON = {SELESAI: "✅", BELUM: "⬜", OPSIONAL: "➖", PERHATIAN: "⚠️"}
BERKAS_RAW = ("histori_krs.csv", "SUMBER_DATA.txt", "kalibrasi_sintetis.json")
KUNCI_API = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


@dataclass
class Langkah:
    no: int
    judul: str
    status: str
    keterangan: str
    tempat: str
    wajib: bool = True


def _jumlah(cfg: Config, tabel: str, where: str = "", params: tuple = ()) -> int:
    return db.jumlah(cfg, tabel, where, params) if db.ada_db(cfg) else 0


def _ringkasan(cfg: Config, nama: str) -> dict | None:
    return db.baca_ringkasan(cfg, nama) if db.ada_db(cfg) else None


def periksa_kesiapan(cfg: Config) -> list[Langkah]:
    """Langkah 1–12 sejak sistem kosong; status dihitung dari keadaan saat ini (tanpa mengubah apa pun)."""
    L: list[Langkah] = []

    def tambah(judul, status, ket, tempat, wajib=True):
        L.append(Langkah(len(L) + 1, judul, status, ket, tempat, wajib))

    # 1. konfigurasi rahasia
    kurang = [v for p, v in auth.VAR_HASH.items() if auth.hash_peran(p) is None]
    var_api = KUNCI_API.get(cfg.llm.provider)
    if var_api and not os.getenv(var_api, "").strip():
        kurang.append(var_api)
    tanpa_salt = not os.getenv("ANON_SALT", "").strip()
    if kurang:
        status, ket = BELUM, f"belum diisi: {', '.join(kurang)}"
    elif tanpa_salt:
        status, ket = PERHATIAN, ("ANON_SALT kosong: impor xlsx ber-NIM akan ditolak (impor CSV anonim tetap bisa). "
                                  "Isi dengan salt yang sama seperti data sebelumnya")
    else:
        status, ket = SELESAI, f"kata sandi peran, salt anonimisasi, dan kunci API ({cfg.llm.provider}) terisi"
    tambah("Konfigurasi .env / app settings", status, ket,
           ".env (lokal) atau Azure App Service → Environment variables")

    # 2. kurikulum
    try:
        kur = load_kurikulum(cfg)
        tambah("Dataset Kurikulum 2024 terbaca", SELESAI,
               f"{len(kur.mk)} MK, {len(kur.prasyarat)} prasyarat, {len(kur.tema)} tema (bawaan aplikasi, hanya dibaca)",
               "Administrator → Kurikulum")
    except Exception as e:     # dataset rusak/hilang: semua langkah lain bergantung padanya
        tambah("Dataset Kurikulum 2024 terbaca", PERHATIAN, f"gagal dimuat: {e}", "Administrator → Kurikulum")

    # 3–4. data KRS
    n_krs = _jumlah(cfg, "krs_asli")
    ak = _ringkasan(cfg, "akademik") or {}
    tambah("Impor data akademik (KRS + IP/IPK)", SELESAI if n_krs else BELUM,
           f"{n_krs:,} baris KRS · periode {', '.join(ak.get('periode_krs', [])) or '-'} · "
           f"{ak.get('n_mahasiswa_krs', 0)} mahasiswa" if n_krs else
           "unggah xlsx sistem akademik atau CSV anonim (templat tersedia)",
           "Administrator → Data akademik → Impor data akademik")
    st_riwayat = status_data_krs(cfg)
    tambah("Riwayat KRS multisemester", BELUM if st_riwayat == "belum_tersedia" else SELESAI,
           "unggah riwayat alumni (templat tersedia) atau bangkitkan riwayat pelengkap"
           if st_riwayat == "belum_tersedia" else "histori_krs.csv tersedia",
           "Administrator → Data akademik")

    # 5–6. preprocessing & ARM
    ada_krs = bool(n_krs) or st_riwayat != "belum_tersedia"
    _, alasan = transaksi_segar(cfg, "latih") if db.ada_db(cfg) else ([], "belum ada")
    pre = _ringkasan(cfg, "preprocessing")
    if pre and alasan is None:
        tambah("Preprocessing", SELESAI, f"{pre.get('n_transaksi_latih', '-')} transaksi latih, "
                                          f"{pre.get('n_transaksi_uji', '-')} transaksi uji",
               "Administrator → Preprocessing")
    else:
        tambah("Preprocessing", PERHATIAN if pre else BELUM,
               "data masukan berubah sejak preprocessing terakhir, jalankan ulang" if pre else
               ("jalankan preprocessing" if ada_krs else "butuh langkah 3 atau 4"), "Administrator → Preprocessing")
    arm = _ringkasan(cfg, "arm")
    arm_segar = bool(arm) and bool(pre) and alasan is None and arm.get("run_id_preprocessing", pre.get("run_id")) \
        == pre.get("run_id")
    tambah("Penambangan aturan asosiasi (ARM)", SELESAI if arm_segar else (PERHATIAN if arm else BELUM),
           f"{arm.get('n_aturan', 0)} aturan" if arm_segar else
           ("jalankan ulang FP-Growth setelah preprocessing terbaru" if arm else "jalankan FP-Growth"),
           "Administrator → ARM")

    # 7–8. beban kognitif
    n_llm = _jumlah(cfg, "rating_bloom", "sumber = 'llm'")
    n_mock = _jumlah(cfg, "rating_bloom", "sumber = 'llm_mock'")
    sumber = "llm" if n_llm else "llm_mock"
    n_mk = int(db.baca(cfg, "SELECT COUNT(DISTINCT kode_mk) AS n FROM rating_bloom WHERE sumber = ?", (sumber,))
               ["n"].iloc[0]) if n_llm or n_mock else 0
    if n_llm:
        status, ket = SELESAI, f"{n_llm:,} penilaian untuk {n_mk} MK"
    elif n_mock:
        status, ket = PERHATIAN, (f"hanya hasil provider mock ({n_mock:,} penilaian, {n_mk} MK): untuk uji alur, "
                                  f"BUKAN hasil penelitian")
    else:
        status, ket = BELUM, "jalankan penilaian (API berbayar) atau pulihkan cadangan rating LLM"
    tambah("Penilaian Bloom per CPMK oleh LLM", status, ket,
           "Administrator → Beban kognitif (LLM) / Mulai di sini → Cadangan & atur ulang")
    cl, th = _ringkasan(cfg, "cl"), _ringkasan(cfg, "theta")
    tambah("Skor CL per MK dan ambang θ", SELESAI if cl and th else BELUM,
           f"θ = {th.get('theta')}" if cl and th and th.get("theta") is not None else
           "klik Hitung ulang CL & θ dari hasil tersimpan", "Administrator → Beban kognitif (LLM)")

    # 9–10. pakar & evaluasi (opsional: RM1)
    n_pakar = int(db.baca(cfg, "SELECT COUNT(DISTINCT penilai) AS n FROM rating_bloom WHERE sumber = 'pakar'")["n"]
                  .iloc[0]) if _jumlah(cfg, "rating_bloom", "sumber = 'pakar'") else 0
    tambah("Impor penilaian pakar (RM1)", SELESAI if n_pakar else OPSIONAL,
           f"{n_pakar} pakar" if n_pakar else "unduh formulir pakar terisi CPMK, isi, lalu impor",
           "Administrator → Beban kognitif (LLM) → Impor penilaian pakar", wajib=False)
    tambah("Evaluasi RM1 (ICC LLM vs pakar)", SELESAI if _ringkasan(cfg, "evaluasi") and n_pakar else OPSIONAL,
           "pratinjau tersedia; unduh rm1_skor_mk.csv untuk JASP" if n_pakar else "butuh langkah 9",
           "Administrator → Beban kognitif (LLM)", wajib=False)

    # 11. simulasi
    sim, val = _ringkasan(cfg, "simulasi"), _ringkasan(cfg, "validasi_krs")
    tambah("Simulasi komparasi & validasi KRS", SELESAI if sim and sim.get("n_profil") else BELUM,
           f"{sim.get('n_profil')} profil simulasi" + (f", {val.get('n_profil', 0)} profil validasi KRS" if val else "")
           if sim and sim.get("n_profil") else "jalankan simulasi komparasi (dan validasi KRS bila ada data akademik)",
           "Administrator → Simulasi & validasi")

    # 12. alur pengguna
    n_rencana = _jumlah(cfg, "rencana")
    n_ditinjau = _jumlah(cfg, "rencana", "status_dpa != 'belum_ditinjau'")
    tambah("Uji alur Mahasiswa → DPA", SELESAI if n_ditinjau else (PERHATIAN if n_rencana else BELUM),
           f"{n_rencana} rencana tersimpan, {n_ditinjau} sudah divalidasi DPA" if n_rencana else
           "bangkitkan rencana di halaman Mahasiswa, lalu validasi di halaman DPA", "Peran Mahasiswa, lalu peran DPA")
    return L


def atur_ulang(cfg: Config, pertahankan_rating_llm: bool = True) -> dict:
    """Kembalikan sistem ke kondisi awal untuk pengujian dari nol: kosongkan basis data (``db.kosongkan_data``),
    hapus riwayat multisemester di ``paths.raw`` dan parameter UI (``config.override.yaml``). Log proses & log
    akses dipertahankan sebagai jejak audit; dataset kurikulum tidak disentuh."""
    tetap = ("rating_bloom", "batch_llm") if pertahankan_rating_llm else ()
    with catat_proses(cfg, "atur_ulang", {"pertahankan_rating_llm": pertahankan_rating_llm}) as log:
        if pertahankan_rating_llm:
            # rating pakar ikut terhapus walau rating LLM dipertahankan (uji impor pakar dari nol)
            with db.koneksi(cfg) as c:
                n_pakar = int(c.execute("SELECT COUNT(*) FROM rating_bloom WHERE sumber = 'pakar'").fetchone()[0])
                c.execute("DELETE FROM rating_bloom WHERE sumber = 'pakar'")
        hapus = db.kosongkan_data(cfg, pertahankan=tetap)
        if pertahankan_rating_llm:
            hapus["rating_bloom (pakar)"] = n_pakar
        berkas = []
        for f in BERKAS_RAW:
            p = cfg.raw_dir / f
            if p.exists():
                p.unlink()
                berkas.append(f)
        override = konfigurasi.OVERRIDE_PATH
        if override.exists():
            override.unlink()
            berkas.append(override.name)
        hasil = {"baris_dihapus": {k: v for k, v in hapus.items() if v}, "berkas_dihapus": berkas}
        log.ringkasan.update(hasil)
    return hasil
