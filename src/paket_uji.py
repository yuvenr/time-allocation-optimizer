"""Paket data uji (iterasi 3): menyiapkan sistem dari kondisi kosong sampai siap pakai dengan 3 berkas CSV.

Untuk pengujian bersama dosen (UAT, RM3) agar tidak perlu menjalankan 12 langkah satu per satu:

| Berkas                    | Isi                                                        | Asal                    |
|---------------------------|------------------------------------------------------------|-------------------------|
| paket_krs_akademik.csv    | KRS akademik anonim (format ``KOLOM_KRS``)                 | tabel krs_asli          |
| paket_ip_semester.csv     | IP/IPK semester anonim (format ``KOLOM_IP``)               | tabel ip_asli           |
| paket_rating_llm.csv      | Rating Bloom per CPMK oleh LLM, 3 pengulangan              | tabel rating_bloom (llm)|

Riwayat KRS multisemester ikut dikemas APA ADANYA (byte identik) bersama penanda asal datanya
(``histori_krs.csv``, ``SUMBER_DATA.txt``, ``kalibrasi_sintetis.json``; keputusan 20 Okt 2026: riwayat lama
dipertahankan agar hasil sama dengan naskah). Berkas itu ditulis langsung ke ``paths.raw`` (bukan lewat unggah
riwayat alumni yang menganggapnya asli), sehingga asal datanya tetap tercatat. Bila paket tidak memuat riwayat,
riwayat pelengkap dibangkitkan (``src.sintetis.generate``, seed tetap). Tidak ada API LLM yang dipanggil.

Berkas paket memuat data akademik (anonim, tanpa NIM) → dibuat di ``data/paket_uji/`` (diabaikan git) atau
diunduh dari UI, lalu diserahkan langsung; jangan di-commit ke repositori publik.

    python -m src.paket_uji buat [folder]       # ekspor paket dari basis data saat ini
    python -m src.paket_uji siapkan <folder>    # siapkan sistem kosong dari paket
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Callable

import pandas as pd

from src.akademik import KOLOM_IP, KOLOM_KRS, impor, muat_ip, muat_krs
from src.berkas import BERKAS_RAW
from src.config import Config, load_config
from src.data_loader import status_data_krs
from src.llm.beban import KOLOM_RATING, ekspor_rating_llm, impor_rating_llm
from src.logutil import catat_proses

BERKAS = {"krs": "paket_krs_akademik.csv", "ip": "paket_ip_semester.csv", "rating_llm": "paket_rating_llm.csv"}
JUDUL = {"krs": "KRS akademik", "ip": "IP/IPK semester", "rating_llm": "Rating Bloom LLM"}
FOLDER_BAWAAN = "data/paket_uji"


def buat_paket(cfg: Config) -> dict[str, pd.DataFrame]:
    """Ekspor isi basis data saat ini menjadi paket. Hanya rating LLM sungguhan (bukan mock) yang dikemas."""
    rating = ekspor_rating_llm(cfg)
    rating = rating[rating["sumber"] == "llm"]
    paket = {"krs": muat_krs(cfg).reindex(columns=KOLOM_KRS), "ip": muat_ip(cfg).reindex(columns=KOLOM_IP),
             "rating_llm": rating}
    kosong = [JUDUL[k] for k, df in paket.items() if df.empty and k != "ip"]
    if kosong:
        raise ValueError(f"basis data belum berisi {', '.join(kosong)}; paket tidak dapat dibuat")
    return paket


def raw_paket(cfg: Config) -> dict[str, bytes]:
    """Berkas riwayat multisemester + penanda asal data di ``paths.raw``, byte per byte."""
    return {f: (cfg.raw_dir / f).read_bytes() for f in BERKAS_RAW if (cfg.raw_dir / f).exists()}


def zip_paket(paket: dict[str, pd.DataFrame], raw: dict[str, bytes] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k, df in paket.items():
            z.writestr(BERKAS[k], df.to_csv(index=False).encode("utf-8-sig"))
        for nama, isi in (raw or {}).items():
            z.writestr(nama, isi)
    return buf.getvalue()


def kenali(df: pd.DataFrame) -> str | None:
    """Jenis berkas paket dari kolomnya (nama berkas boleh diganti pengguna)."""
    kolom = set(df.columns)
    if {"sumber", "id_cpmk", "model", "versi_prompt", "bloom"} <= kolom:
        return "rating_llm"
    if {"ip_beban", "ipk_beban"} <= kolom:
        return "ip"
    if {"id_anon", "periode", "kode_mk", "sks"} <= kolom:
        return "krs"
    return None


def siapkan_dari_paket(cfg: Config, berkas: dict[str, pd.DataFrame], raw: dict[str, bytes] | None = None,
                       progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Kondisi kosong → siap pakai: impor data akademik → riwayat multisemester (dari paket apa adanya, atau
    dibangkitkan) → pulihkan rating LLM → preprocessing, ARM, CL & θ, evaluasi, simulasi & validasi KRS
    (``src.pipeline.jalankan_semua``, tanpa API)."""
    from src.pipeline import jalankan_semua
    from src.sintetis import generate

    if "krs" not in berkas:
        raise ValueError(f"berkas {BERKAS['krs']} wajib ada di paket")
    lapor = progress or (lambda i, n, t: None)
    n = 4
    hasil: dict = {}
    with catat_proses(cfg, "siapkan_paket_uji", {"berkas": sorted(berkas)}) as log:
        lapor(0, n, "Memvalidasi dan mengimpor data akademik…")
        hasil["akademik"] = impor(cfg, berkas["krs"], berkas.get("ip", pd.DataFrame(columns=KOLOM_IP)),
                                  asal="paket data uji")
        lapor(1, n, "Menyiapkan riwayat KRS multisemester…")
        raw = {k: v for k, v in (raw or {}).items() if k in BERKAS_RAW}
        if status_data_krs(cfg) not in ("belum_tersedia", "sintetis"):
            hasil["riwayat"] = "riwayat alumni asli sudah ada, tidak ditimpa"
        elif "histori_krs.csv" in raw:
            from src.berkas import simpan as simpan_berkas

            cfg.raw_dir.mkdir(parents=True, exist_ok=True)
            for f in BERKAS_RAW:       # penanda asal data mengikuti paket persis (tanpa penanda = riwayat asli)
                (cfg.raw_dir / f).unlink(missing_ok=True)
                if f in raw:
                    (cfg.raw_dir / f).write_bytes(raw[f])
            simpan_berkas(cfg)
            hasil["riwayat"] = {"dari_paket": sorted(raw)}
        else:
            hasil["riwayat"] = {k: v for k, v in generate(cfg).items() if k in ("n_baris", "n_alumni")}
        lapor(2, n, "Memulihkan rating Bloom LLM…")
        if "rating_llm" in berkas:
            hasil["rating_llm"] = impor_rating_llm(cfg, berkas["rating_llm"])
        lapor(3, n, "Preprocessing, ARM, CL & θ, evaluasi, simulasi, dan validasi KRS…")
        r = jalankan_semua(cfg, nilai_bloom=False)
        hasil["theta"] = r.get("theta", {}).get("theta")
        hasil["n_aturan"] = r.get("arm", {}).get("n_aturan")
        lapor(4, n, "Selesai")
        log.ringkasan.update({k: v for k, v in hasil.items() if k in ("theta", "n_aturan")})
    return hasil


def baca_folder(folder: Path) -> tuple[dict[str, pd.DataFrame], dict[str, bytes]]:
    from src.templat import baca_csv_unggahan

    hasil, raw = {}, {}
    for f in sorted(Path(folder).iterdir()):
        if f.name in BERKAS_RAW:
            raw[f.name] = f.read_bytes()
        elif f.suffix.lower() == ".csv":
            df = baca_csv_unggahan(f, dtype={"id_anon": str, "kode_mk": str, "penilai": str})
            if (k := kenali(df)) is not None:
                hasil[k] = df
    return hasil, raw


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("buat", "siapkan"):
        print(__doc__)
        return
    cfg = load_config()
    folder = Path(argv[1] if len(argv) > 1 else FOLDER_BAWAAN)
    if argv[0] == "buat":
        folder.mkdir(parents=True, exist_ok=True)
        for k, df in buat_paket(cfg).items():
            df.to_csv(folder / BERKAS[k], index=False, encoding="utf-8-sig")
            print(f"{folder / BERKAS[k]}: {len(df):,} baris")
        for nama, isi in raw_paket(cfg).items():
            (folder / nama).write_bytes(isi)
            print(f"{folder / nama}: {len(isi):,} bita (apa adanya)")
    else:
        berkas, raw = baca_folder(folder)
        r = siapkan_dari_paket(cfg, berkas, raw, progress=lambda i, n, t: print(f"[{i}/{n}] {t}"))
        print(f"Selesai: θ = {r['theta']}, {r['n_aturan']} aturan ARM")


if __name__ == "__main__":
    main()
