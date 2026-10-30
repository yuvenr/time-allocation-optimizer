"""Simulasi skenario mahasiswa berisiko studi molor (src/skenario.py) pada dataset Kurikulum 2024."""
from __future__ import annotations

import pytest

from src.config import load_config
from src.planner import KonteksPlanner
from src.rules import Keadaan, batas_sks, boleh_diambil, cek_mk
from src.skenario import SKENARIO, jalankan_skenario, profil_skenario


@pytest.fixture(scope="module")
def ctx():
    return KonteksPlanner.muat(load_config())


def test_profil_skenario_sesuai_definisi(ctx):
    kur = ctx.kurikulum
    for sk in SKENARIO:
        p = profil_skenario(sk, kur, ctx.cfg)
        assert p.semester_saat_ini == sk.semester and p.ip_terakhir == sk.ip and p.jalur == "A"
        assert set(sk.gagal) == p.mk_pernah_gagal
        assert not {k for k in sk.belum if k in kur} & p.pernah_menempuh
        assert all(kur.semester_jalur(k, "A") is None or kur.semester_jalur(k, "A") < sk.semester
                   for k in p.pernah_menempuh)


@pytest.mark.parametrize("sk", SKENARIO, ids=lambda s: s.id)
def test_skenario_molor_dan_sistem_tidak_lebih_lambat(sk, ctx):
    r = jalankan_skenario(sk, ctx)
    konv = r["lulus"]["konvensional"]
    assert konv is not None and konv > ctx.cfg.kurikulum.horizon_semester     # memang berisiko molor
    for mode, plan in r["rencana"].items():
        assert r["lulus"][mode] is not None and r["lulus"][mode] <= konv
        assert all(plan.validasi.per_aturan[a] for a in ("AD-02", "AD-03", "AD-04")), plan.validasi.pelanggaran


def test_pola_konvensional_patuh_aturan(ctx):
    """Pembanding memakai constraint engine yang sama: prasyarat/penawaran dan pita SKS tiap semester."""
    kur, cfg = ctx.kurikulum, ctx.cfg
    for sk in SKENARIO:
        r = jalankan_skenario(sk, ctx)
        kd = Keadaan.dari_profil(r["profil"], kur, cfg)
        for s, kode in sorted(r["konvensional"].semester.items()):
            assert kur.total_sks(kode) <= batas_sks(s, "A", sk.ip, kur)[1]
            assert all(boleh_diambil(cek_mk(k, s, kd, kur, cfg)) for k in kode), (sk.id, s)
            kd = kd.tambah(kode)


def test_sistem_lebih_cepat_minimal_satu_skenario(ctx):
    hasil = [jalankan_skenario(sk, ctx)["lulus"] for sk in SKENARIO]
    assert any(h["arm"] < h["konvensional"] for h in hasil)
