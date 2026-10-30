"""Tes paket data uji (``src/paket_uji.py``): dari sistem kosong menjadi siap pakai dengan 3 CSV, reprodusibel,
tanpa API. Data akademik FIKTIF (helper tes akademik); rating = keluaran provider mock yang diberi label llm
khusus untuk tes ini."""
from __future__ import annotations

import filecmp
import io
import zipfile

import pandas as pd
import pytest

from src import db
from src.kesiapan import SELESAI, periksa_kesiapan
from src.llm.beban import ekspor_rating_llm, jalankan_penilaian_bloom
from src.paket_uji import BERKAS, buat_paket, kenali, raw_paket, siapkan_dari_paket, zip_paket
from src.planner import KonteksPlanner
from tests.conftest import buat_cfg
from tests.test_akademik import _ip, _krs

LANGKAH_SIAP = ("Impor data akademik (KRS + IP/IPK)", "Riwayat KRS multisemester", "Preprocessing",
                "Penambangan aturan asosiasi (ARM)", "Penilaian Bloom per CPMK oleh LLM", "Skor CL per MK dan ambang θ",
                "Simulasi komparasi & validasi KRS")


def _cfg(base):
    (base / "raw").mkdir(parents=True)
    return buat_cfg(base, arm={"min_transaksi_asli": 10}, generator={"n_alumni": 60})


@pytest.fixture(scope="module")
def paket(tmp_path_factory) -> dict[str, pd.DataFrame]:
    cfg = _cfg(tmp_path_factory.mktemp("rating"))
    jalankan_penilaian_bloom(cfg, provider="mock")
    rating = ekspor_rating_llm(cfg).assign(sumber="llm", model="model-uji")
    return {"krs": _krs(30, 30), "ip": _ip(30), "rating_llm": rating}


def test_dari_kosong_menjadi_siap(tmp_path, paket):
    cfg = _cfg(tmp_path)
    assert all(x.status != SELESAI for x in periksa_kesiapan(cfg)[2:])
    r = siapkan_dari_paket(cfg, paket)
    assert r["theta"] is not None and r["rating_llm"]["n_pengulangan"] == 3
    status = {x.judul: x.status for x in periksa_kesiapan(cfg)}
    assert all(status[j] == SELESAI for j in LANGKAH_SIAP), status
    # riwayat pelengkap dibangkitkan (bukan diunggah) → asal datanya tetap tercatat
    assert (cfg.raw_dir / "SUMBER_DATA.txt").read_text(encoding="utf-8").startswith("sintetis")
    assert db.jumlah(cfg, "log_proses", "proses = 'siapkan_paket_uji' AND status = 'ok'") == 1


def test_reprodusibel(tmp_path, paket):
    """Dua sistem kosong yang disiapkan dari paket yang sama menghasilkan riwayat, θ, CL, dan aturan identik."""
    a, b = _cfg(tmp_path / "a"), _cfg(tmp_path / "b")
    siapkan_dari_paket(a, paket)
    siapkan_dari_paket(b, paket)
    assert filecmp.cmp(a.raw_dir / "histori_krs.csv", b.raw_dir / "histori_krs.csv", shallow=False)
    ka, kb = KonteksPlanner.muat(a), KonteksPlanner.muat(b)
    assert ka.theta == kb.theta and ka.cl == kb.cl and ka.arm.per_konsekuen.keys() == kb.arm.per_konsekuen.keys()


def test_buat_paket_dan_zip_pulang_pergi(tmp_path, paket):
    cfg = _cfg(tmp_path)
    siapkan_dari_paket(cfg, paket)
    keluar = buat_paket(cfg)
    assert len(keluar["krs"]) == len(paket["krs"]) and len(keluar["rating_llm"]) == len(paket["rating_llm"])
    with zipfile.ZipFile(io.BytesIO(zip_paket(keluar))) as z:
        assert sorted(z.namelist()) == sorted(BERKAS.values())
        jenis = {kenali(pd.read_csv(io.BytesIO(z.read(n)), encoding="utf-8-sig")): n for n in z.namelist()}
    assert {k: BERKAS[k] for k in BERKAS} == jenis


def test_paket_tidak_lengkap_ditolak(tmp_path, paket):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="belum berisi"):
        buat_paket(cfg)
    with pytest.raises(ValueError, match="wajib"):
        siapkan_dari_paket(cfg, {"rating_llm": paket["rating_llm"]})
    assert kenali(pd.DataFrame(columns=["a", "b"])) is None


def test_riwayat_dari_paket_dipasang_apa_adanya(tmp_path, paket):
    """Keputusan 20 Okt 2026: riwayat lama ikut paket byte identik (bukan dibangkitkan ulang); penanda asal data
    ikut, sehingga riwayat tetap tercatat sebagai pelengkap. Tanpa penanda → dianggap riwayat asli."""
    sumber = _cfg(tmp_path / "sumber")
    siapkan_dari_paket(sumber, paket)                       # membangkitkan riwayat
    raw = raw_paket(sumber)
    assert set(raw) == {"histori_krs.csv", "SUMBER_DATA.txt", "kalibrasi_sintetis.json"}

    tujuan = _cfg(tmp_path / "tujuan")
    r = siapkan_dari_paket(tujuan, paket, raw)
    assert r["riwayat"] == {"dari_paket": sorted(raw)}
    assert all((tujuan.raw_dir / f).read_bytes() == isi for f, isi in raw.items())
    assert KonteksPlanner.muat(tujuan).theta == KonteksPlanner.muat(sumber).theta
    assert db.jumlah(tujuan, "berkas") == 3                 # salinan untuk hosting tanpa disk persisten
    with zipfile.ZipFile(io.BytesIO(zip_paket(buat_paket(tujuan), raw))) as z:
        assert set(raw) <= set(z.namelist())

    asli = _cfg(tmp_path / "asli")
    siapkan_dari_paket(asli, paket, {"histori_krs.csv": raw["histori_krs.csv"]})
    assert not (asli.raw_dir / "SUMBER_DATA.txt").exists()
