"""Metrik evaluasi rencana studi (UC-12).

Keseimbangan beban:
- ``sd_beban``      : SD populasi B_s antarsemester rencana.
- ``beban_puncak``  : max B_s.
- ``sd_sks``        : SD populasi SKS antarsemester.
Relevansi:
- ``jaccard_rata``  : Pers. 2.12 — rata-rata J(A_t, B_t) atas T semester yang DIREKOMENDASIKAN (A_t ≠ ∅),
                      B_t = MK aktual alumni pada semester t.
Kepatuhan (dari ``validate_plan``, identik kedua mode):
- ``lolos_prasyarat`` (AD-03), ``lolos_sks`` (AD-02), ``lolos_paritas`` (AD-04), ``lolos_semua``;
- ``lolos_145``     : total minimal (AT01, 145 SKS) tercapai DALAM horizon (semester 8);
- ``lolos_theta``   : semua semester reguler (bukan paket/konversi) B_s ≤ θ (dihitung untuk kedua mode).
Lama studi:
- ``semester_selesai`` : semester terakhir yang berisi MK dalam rencana (K5: semakin kecil semakin baik).

Dengan ``metrik.abaikan_semester_kosong`` (default), SD dihitung hanya atas
semester yang berisi MK, sehingga rencana yang selesai lebih awal tidak
dihukum oleh semester kosong di akhir horizon.
"""
from __future__ import annotations

import numpy as np

from src.config import Config
from src.schemas import Plan


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def metrik_rencana(plan: Plan, cfg: Config, aktual: dict[int, set[str]] | None = None,
                   theta: float | None = None) -> dict:
    """Hitung metrik satu rencana. ``aktual`` = MK yang diambil alumni per semester."""
    sems = plan.semester
    if cfg.metrik.abaikan_semester_kosong:
        terisi = [s for s in sems if s.items]
    else:
        terisi = list(sems)
    beban = np.array([s.total_beban for s in terisi], dtype=float)
    sks = np.array([s.total_sks for s in terisi], dtype=float)
    theta = plan.theta if theta is None else theta

    hasil = {
        "sd_beban": round(float(beban.std(ddof=0)), 4) if len(beban) else 0.0,
        "beban_puncak": round(float(beban.max()), 4) if len(beban) else 0.0,
        "rata_beban": round(float(beban.mean()), 4) if len(beban) else 0.0,
        "sd_sks": round(float(sks.std(ddof=0)), 4) if len(sks) else 0.0,
        "total_sks_rencana": int(sum(s.total_sks for s in sems)),
        "n_semester_terisi": int(sum(1 for s in sems if s.items)),
        "semester_selesai": max((s.semester for s in sems if s.items), default=None),
    }
    if aktual is not None:
        nilai = [jaccard(a, aktual.get(s.semester, set())) for s in sems if (a := {i.kode for i in s.items})]
        hasil["jaccard_rata"] = round(float(np.mean(nilai)), 4) if nilai else None
    horizon = cfg.kurikulum.horizon_semester
    sks_horizon = plan.sks_lulus_awal + sum(s.total_sks for s in sems if s.semester <= horizon)

    v = plan.validasi
    per = v.per_aturan if v else {}
    hasil.update({
        "lolos_semua": bool(v.lolos) if v else False,
        "lolos_prasyarat": bool(per.get("AD-03", False)),
        "lolos_sks": bool(per.get("AD-02", False)),
        "lolos_paritas": bool(per.get("AD-04", False)),
        "lolos_145": bool(per.get("AD-01", False)) and sks_horizon >= plan.sks_minimal_lulus,
        "lolos_theta": bool(theta is not None and all(s.total_beban <= theta + 1e-9 for s in sems
                                                      if s.jenis == "reguler")),
    })
    return hasil


METRIK_NUMERIK = ["sd_beban", "beban_puncak", "rata_beban", "sd_sks", "jaccard_rata", "total_sks_rencana",
                  "n_semester_terisi", "semester_selesai"]
METRIK_BOOLEAN = ["lolos_semua", "lolos_prasyarat", "lolos_sks", "lolos_paritas", "lolos_145", "lolos_theta"]
