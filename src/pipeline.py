"""Orkestrasi pipeline: [preprocessing → ARM] → penilaian Bloom → CL → θ → evaluasi → [simulasi & validasi].

Langkah dalam kurung siku hanya berjalan bila ada data KRS (data akademik di basis data dan/atau
riwayat ``paths.raw/histori_krs.csv``; ``label_sumber_data``). Dipakai UI dan CLI:
    python -m src.pipeline [--tanpa-simulasi] [--bloom]

``--bloom`` menjalankan penilaian LLM (berbayar); tanpanya rating Bloom yang tersimpan dipakai ulang.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable

from src import db
from src.config import Config, load_config
from src.data_loader import label_sumber_data, sidik_raw

TAHAP = ("preprocessing", "arm", "cl", "theta", "evaluasi", "simulasi", "validasi_krs")   # nama ringkasan di basis data


def hitung_dan_simpan_theta(cfg: Config) -> dict:
    """Hitung θ (manual → alumni → struktur) + skor CL terkini lalu simpan ringkasannya."""
    from src.planner import KonteksPlanner

    ctx = KonteksPlanner.muat(cfg)
    info = dict(ctx.info_theta)
    info["dihitung_pada"] = datetime.now().isoformat(timespec="seconds")
    info["mk_tanpa_cl"] = ctx.mk_tanpa_cl
    info["sumber_cl"] = ctx.info_cl
    db.simpan_ringkasan(cfg, "theta", info)
    return info


def status_pipeline(cfg: Config) -> dict:
    """Artefak mana yang sudah ada dan apakah data masukan berubah sejak pipeline terakhir."""
    ada = {tahap: db.baca_ringkasan(cfg, tahap) is not None for tahap in TAHAP}
    state = db.baca_ringkasan(cfg, "pipeline") or {}
    return {
        "data_krs": label_sumber_data(cfg),
        "artefak": ada,
        "basi": bool(state) and state.get("sidik_raw") != sidik_raw(cfg),
        "terakhir": state.get("selesai"),
    }


def jalankan_semua(cfg: Config, provider: str | None = None, nilai_bloom: bool = False,
                   sertakan_simulasi: bool = True, progress: Callable[[str], None] | None = None) -> dict:
    """Jalankan seluruh pipeline secara berurutan."""
    from src.arm import jalankan_arm
    from src.export import ekspor_evaluasi
    from src.llm.beban import jalankan_cl, jalankan_penilaian_bloom
    from src.preprocessing import jalankan_preprocessing

    lapor = progress or (lambda _t: None)
    hasil: dict = {}
    krs = label_sumber_data(cfg) != "belum_tersedia"
    if krs:
        lapor("Preprocessing & normalisasi KRS (UC-07)…")
        hasil["preprocessing"] = jalankan_preprocessing(cfg)
        lapor("Penambangan aturan asosiasi FP-Growth (UC-08)…")
        hasil["arm"] = jalankan_arm(cfg)
    else:
        lapor("Data KRS belum tersedia, preprocessing & ARM dilewati.")
    if nilai_bloom:      # memanggil API berbayar — hanya bila diminta eksplisit (hasil tersimpan dipakai ulang)
        lapor("Penilaian Bloom per CPMK berbatch (UC-09)…")
        hasil["bloom"] = jalankan_penilaian_bloom(cfg, provider=provider, lanjutkan=True)
    lapor("Menghitung CL per MK dan θ…")
    hasil["cl"] = jalankan_cl(cfg)
    hasil["theta"] = hitung_dan_simpan_theta(cfg)
    lapor("Evaluasi kesepakatan Bloom (heuristik / LLM / pakar)…")
    hasil["evaluasi"] = ekspor_evaluasi(cfg)
    if krs and sertakan_simulasi:
        from src.data_loader import sumber_data
        from src.simulate import jalankan_simulasi, jalankan_validasi_krs

        sumber = sumber_data(cfg)
        if sumber["riwayat"]:
            lapor("Simulasi komparasi mode arm vs hybrid (UC-12)…")
            hasil["simulasi"] = jalankan_simulasi(cfg)
        if sumber["krs_akademik"]:
            lapor("Validasi rekomendasi terhadap KRS akademik (data uji)…")
            hasil["validasi_krs"] = jalankan_validasi_krs(cfg)
    db.simpan_ringkasan(cfg, "pipeline", {"sidik_raw": sidik_raw(cfg),
                                        "selesai": datetime.now().isoformat(timespec="seconds"),
                                        "tahap": list(hasil)})
    return hasil


if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    r = jalankan_semua(load_config(), nilai_bloom="--bloom" in sys.argv,
                       sertakan_simulasi="--tanpa-simulasi" not in sys.argv, progress=print)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))}
                      if isinstance(v, dict) else v for k, v in r.items()},
                     indent=2, ensure_ascii=False, default=str))
