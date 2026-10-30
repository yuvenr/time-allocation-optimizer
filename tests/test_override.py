"""Tes override kurikulum administrator (src/override.py; KF-01..KF-03, KF-11).
Dataset CSV tidak pernah diubah; override tersimpan terpisah, wajib beralasan, dan tervalidasi."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src import db
from src.data_loader import load_konteks, load_kurikulum
from src.llm.beban import bangun_prompt_batch, rng_batch
from src.llm.konteks import SumberKonteks
from src.llm.prompts import muat_prompt
from src.override import (daftar_override_prasyarat, hapus_override_cpmk, hapus_override_prasyarat,
                          simpan_override_cpmk, simpan_override_prasyarat)
from src.rules import Keadaan, batas_sks, cek_mk
from tests.conftest import buat_cfg


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "raw").mkdir()
    return buat_cfg(tmp_path)


def _ad03(kode, lulus, kur, cfg, s=4):
    kd = Keadaan("A", None, frozenset(lulus), frozenset(lulus))
    return {h.status for h in cek_mk(kode, s, kd, kur, cfg) if h.kode == "AD-03"}


def test_override_cpmk_dipakai_penilaian_bloom(cfg):
    cpmk = load_konteks(cfg, "cpmk")
    asli = cpmk.loc[cpmk["id_cpmk"] == "COM60014-CPMK01", "teks"].iloc[0]
    baru = "Mahasiswa mampu merancang program terstruktur untuk menyelesaikan masalah komputasi sederhana."
    simpan_override_cpmk(cfg, "COM60014-CPMK01", baru, "rumusan asli ambigu (KKO ganda)", cpmk)
    ovr = load_konteks(cfg, "cpmk").set_index("id_cpmk").loc["COM60014-CPMK01"]
    assert ovr["teks"] == baru and ovr["teks_dataset"] == asli and bool(ovr["override"])
    _, user, _ = bangun_prompt_batch([SumberKonteks.muat(cfg).mk("COM60014")], rng_batch(1, 1),
                                     muat_prompt("bloom", cfg.llm.versi_prompt_bloom))
    assert baru in user and asli not in user
    with pytest.raises(ValueError, match="tidak dikenal"):
        simpan_override_cpmk(cfg, "XXX-CPMK01", baru, "x", cpmk)
    with pytest.raises(ValueError, match="alasan"):
        simpan_override_cpmk(cfg, "COM60014-CPMK01", baru, " ", cpmk)
    with pytest.raises(ValueError, match="pendek"):
        simpan_override_cpmk(cfg, "COM60014-CPMK01", "abc", "x", cpmk)
    hapus_override_cpmk(cfg, "COM60014-CPMK01")
    assert load_konteks(cfg, "cpmk").set_index("id_cpmk").loc["COM60014-CPMK01", "teks"] == asli


def test_override_prasyarat_hapus_dan_tambah(cfg):
    """Dataset: CIE62209 ← CIE62204 (PR001). Hapus → CIE62209 tanpa prasyarat. Tambah CIE62201 → CIE62209."""
    kur = load_kurikulum(cfg)
    assert _ad03("CIE62209", set(), kur, cfg) == {"gagal"}
    simpan_override_prasyarat(cfg, "CIE62209", "CIE62204", "hapus", "keputusan rapat prodi (uji)")
    kur = load_kurikulum(cfg)
    assert _ad03("CIE62209", set(), kur, cfg) == set()
    simpan_override_prasyarat(cfg, "CIE62209", "CIE62201", "tambah", "uji")
    kur = load_kurikulum(cfg)
    assert _ad03("CIE62209", set(), kur, cfg) == {"gagal"}
    assert _ad03("CIE62209", {"CIE62201"}, kur, cfg) == {"lolos"}
    h = next(h for h in cek_mk("CIE62209", 4, Keadaan("A", None, frozenset(), frozenset()), kur, cfg)
             if h.kode == "AD-03")
    assert "OVR-CIE62209-CIE62201" in h.sumber and "override_prasyarat" in h.sumber
    assert len(daftar_override_prasyarat(cfg)) == 2
    assert len(load_kurikulum(cfg, override=False).prasyarat) == len(kur.prasyarat)   # −1 hapus, +1 tambah
    hapus_override_prasyarat(cfg, "CIE62209", "CIE62201")
    hapus_override_prasyarat(cfg, "CIE62209", "CIE62204")
    assert _ad03("CIE62209", set(), load_kurikulum(cfg), cfg) == {"gagal"}


@pytest.mark.parametrize("args,pesan", [
    (("CIE62204", "CIE62209", "tambah", "uji"), "bersiklus"),          # kebalikan PR001 → siklus
    (("CIE62209", "XXX00000", "tambah", "uji"), "tidak dikenal"),
    (("CIE62209", "CIE62204", "tambah", "uji"), "sudah ada"),
    (("CIE62209", "CIE62201", "hapus", "uji"), "tidak ada di dataset"),
    (("CIE62209", "CIE62209", "tambah", "uji"), "dirinya sendiri"),
    (("CIE62209", "CIE62201", "tambah", ""), "alasan"),
    (("CIE62209", "CIE62201", "ubah", "uji"), "aksi"),
])
def test_override_prasyarat_ditolak(cfg, args, pesan):
    with pytest.raises(ValueError, match=pesan):
        simpan_override_prasyarat(cfg, *args)
    assert daftar_override_prasyarat(cfg).empty                         # tidak tersimpan bila tidak sah


def test_override_mengubah_sidik_konteks(cfg):
    awal = db.sidik_konteks(cfg)
    simpan_override_prasyarat(cfg, "CIE62209", "CIE62201", "tambah", "uji")
    assert db.sidik_konteks(cfg) != awal


def test_batas_sks_ip_dari_config(tmp_path):
    """KF-11: IP ≥ 3,00 → [20, 22] menimpa pita 09 (22–24) di semua semester; pita lain tetap."""
    (tmp_path / "raw").mkdir()
    cfg = buat_cfg(tmp_path, aturan={"BATAS_SKS_IP": {3.0: [20, 22]}})
    kur = load_kurikulum(cfg)
    assert batas_sks(3, "A", 3.5, kur)[:2] == (20, 22) and batas_sks(6, "B", 3.2, kur)[:2] == (20, 22)
    assert batas_sks(3, "A", 2.7, kur)[:2] == (19, 21)
    assert "KF-11" in kur.pita("A", 3, 3.5).catatan
    assert batas_sks(3, "A", 3.5, load_kurikulum(cfg, override=False))[:2] == (22, 24)
    with pytest.raises(ValueError, match="tidak ada di 09"):
        load_kurikulum(buat_cfg(tmp_path, aturan={"BATAS_SKS_IP": {3.1: [20, 22]}}))
    with pytest.raises(ValidationError, match="sks_min"):
        buat_cfg(tmp_path, aturan={"BATAS_SKS_IP": {3.0: [24, 20]}})
