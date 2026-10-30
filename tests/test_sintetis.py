"""Tes generator riwayat KRS pelengkap (``src/sintetis.py``).

Yang diuji: label & provenance (selalu ``sintetis``), determinisme, perlindungan riwayat asli, konsistensi
setiap riwayat terhadap constraint engine Kurikulum 2024 (diputar ulang semester demi semester lewat
``src/rules.py``), dan kalibrasi IPK terhadap data akademik di basis data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.akademik import impor
from src.data_loader import KOLOM_KRS_WAJIB, load_kurikulum, status_data_krs, transaksi_segar
from src.preprocessing import jalankan_preprocessing, normalisasi_kode
from src.rules import Keadaan, aturan_korequisit, aturan_paket, batas_sks, cek_mk, semester_paket, tema_terpenuhi
from src.schemas import KRS
from src.simulate import bentuk_profil, kelompokkan
from src.sintetis import generate
from tests.conftest import buat_cfg

N_ALUMNI = 40


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    base = tmp_path_factory.mktemp("demo")
    cfg = buat_cfg(base, generator={"n_alumni": N_ALUMNI})
    ringkasan = generate(cfg, cfg.raw_dir)
    return cfg, load_kurikulum(cfg), pd.read_csv(cfg.raw_dir / "histori_krs.csv", dtype={"kode_mk": str}), ringkasan


def test_label_sintetis_dan_laporan_kalibrasi(demo):
    cfg, _, _, r = demo
    assert (cfg.raw_dir / "SUMBER_DATA.txt").read_text(encoding="utf-8").startswith("sintetis")
    assert (cfg.raw_dir / "kalibrasi_sintetis.json").exists()
    assert status_data_krs(cfg) == "sintetis"
    assert r["sumber_kalibrasi"].startswith("bawaan")          # tanpa data akademik di basis data
    assert r["sintetis"]["n_mahasiswa"] == N_ALUMNI


def test_deterministik(demo, tmp_path):
    cfg, _, _, _ = demo
    generate(cfg, tmp_path / "ulang")
    assert (tmp_path / "ulang" / "histori_krs.csv").read_bytes() == (cfg.raw_dir / "histori_krs.csv").read_bytes()


def test_tidak_menimpa_riwayat_asli(demo, tmp_path):
    cfg, _, _, _ = demo
    asli = tmp_path / "asli"
    asli.mkdir()
    (asli / "histori_krs.csv").write_text("nim,angkatan\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generate(cfg, asli)
    assert (asli / "histori_krs.csv").read_text(encoding="utf-8") == "nim,angkatan\n"


def test_skema_dan_kode_kurikulum_2024(demo):
    _, kur, df, _ = demo
    assert list(df.columns) == ["id_anon"] + KOLOM_KRS_WAJIB
    for rec in df.to_dict("records"):
        KRS.model_validate(rec)
    assert df["id_anon"].str.startswith("SIN").all()                # tidak tertukar dengan ID akademik (M…)
    assert df.groupby("id_anon")["masa_studi_semester"].nunique().eq(1).all()
    assert all(normalisasi_kode(k, kur)[0] == [k] for k in df["kode_mk"].unique())


def _putar_ulang(g: pd.DataFrame, kur, cfg):
    """Riwayat satu alumni → daftar (semester, keadaan awal, kode, ip_prev) + MK lulus akhir."""
    agama = next(iter(kur.grup("AGAMA") & set(g["kode_mk"])), None)
    lulus: set[str] = set()
    pernah: set[str] = set()
    ip_prev = None
    langkah = []
    for s, gs in g.groupby("semester_ke", sort=True):
        langkah.append((int(s), Keadaan("A", agama, frozenset(lulus), frozenset(pernah), ipk=4.0),
                        list(gs["kode_mk"]), ip_prev))
        pernah |= set(gs["kode_mk"])
        lulus |= {k for k, n in zip(gs["kode_mk"], gs["nilai_huruf"]) if cfg.aturan.lulus(n)}
        ip_prev = float(gs["ip_semester"].iloc[0])
    return langkah, lulus


def test_riwayat_patuh_constraint_engine(demo):
    """Setiap MK lolos aturan per MK pada keadaan awal semester; paket 1–2 (AD-08), korequisit PLP,
    pita SKS 09 dari IP semester sebelumnya (AD-02)."""
    cfg, kur, df, _ = demo
    paket = semester_paket(kur)
    for id_anon, g in df.groupby("id_anon"):
        langkah, _ = _putar_ulang(g, kur, cfg)
        for s, kd, kode, ip_prev in langkah:
            assert len(kode) == len(set(kode)), (id_anon, s)
            for k in kode:
                gagal = [h for h in cek_mk(k, s, kd, kur, cfg) if h.melanggar]
                assert not gagal, (id_anon, s, k, [h.alasan for h in gagal])
            assert not [h for h in aturan_korequisit(s, kode, kd, kur, cfg) if h.melanggar], (id_anon, s)
            if s in paket:
                assert not [h for h in aturan_paket(s, kode, kd, kur, cfg) if h.melanggar], (id_anon, s)
            else:
                assert kur.total_sks(kode) <= batas_sks(s, "A", ip_prev, kur)[1], (id_anon, s, ip_prev)


def test_alumni_lulus_sesuai_kurikulum_2024(demo):
    cfg, kur, df, _ = demo
    agama = kur.grup("AGAMA")
    for id_anon, g in df.groupby("id_anon"):
        _, lulus = _putar_ulang(g, kur, cfg)
        assert kur.wajib - agama <= lulus, (id_anon, sorted(kur.wajib - agama - lulus))
        assert len(agama & lulus) == 1
        assert tema_terpenuhi(lulus, [], kur) == set(kur.tema)
        assert kur.total_sks(lulus) >= 145
        assert g["semester_ke"].max() == g["masa_studi_semester"].iloc[0]


def test_ip_konsisten_dan_variasi(demo):
    cfg, _, df, r = demo
    bobot = cfg.aturan.BOBOT_NILAI
    for _, gs in df.groupby(["id_anon", "semester_ke"]):
        ip = sum(bobot[n] * s for n, s in zip(gs["nilai_huruf"], gs["sks"])) / gs["sks"].sum()
        assert gs["ip_semester"].nunique() == 1 and abs(gs["ip_semester"].iloc[0] - ip) <= 0.005 + 1e-9
    assert (~df["nilai_huruf"].map(cfg.aturan.lulus)).any()               # ada MK yang diulang
    assert 0.3 <= r["sintetis"]["proporsi_masa_studi_le_8"] <= 1.0
    assert r["sintetis"]["ipk_median"] >= 3.0                             # bawaan IPK wajar (bukan "jelek")


def test_kalibrasi_dari_data_akademik(tmp_path):
    """IPK sintetis mengikuti distribusi IPK akademik di basis data (dipotong di generator.ipk_min)."""
    cfg = buat_cfg(tmp_path, generator={"n_alumni": 30, "ipk_min": 2.75})
    rng = np.random.default_rng(1)
    ipk = np.round(rng.uniform(2.9, 3.5, 60), 2)
    ip = pd.DataFrame({"id_anon": [f"M{i:08X}" for i in range(60)], "angkatan": 2024, "semester_ke": 4,
                       "periode": "Genap 2025/2026", "ip_beban": ipk, "sks_beban": 24, "ipk_beban": ipk,
                       "sksk_beban": 88, "ip_lulus": ipk, "sks_lulus": 24, "ipk_lulus": ipk, "sksk_lulus": 88,
                       "sks_tidak_lulus": 0})
    krs = pd.DataFrame({"id_anon": ip["id_anon"], "angkatan": 2024, "semester_ke": 5, "periode": "Ganjil 2026/2027",
                        "kode_mk": "CIE61108", "nama_mk": "x", "sks": 3, "kurikulum_kode": "2024"})
    impor(cfg, krs, ip, asal="tes")
    r = generate(cfg)
    assert r["sumber_kalibrasi"].startswith("data akademik (60 mahasiswa")
    assert r["proporsi_sks_tidak_lulus"] == 0 and r["nyata"]["ipk_median"] == pytest.approx(float(np.median(ipk)), abs=0.01)
    assert abs(r["sintetis"]["ipk_median"] - float(np.median(ipk))) <= 0.15


def test_ujung_ke_ujung_profil_alumni_uji(demo):
    """Setelah preprocessing: transaksi uji segar berlabel sintetis, kode dikenal, dan profil yang dipotong di k
    memuat MK lulus semester 1..k."""
    cfg, kur, _, _ = demo
    ringkasan = jalankan_preprocessing(cfg)
    assert ringkasan["mk_tidak_dikenal"] == {} and ringkasan["n_transaksi_asli"] == 0
    uji, alasan = transaksi_segar(cfg, "uji")
    assert alasan is None and uji and {t.sumber for t in uji} == {"sintetis"}
    assert set().union(*(t.kode for t in uji)) <= set(kur.kode)
    for riwayat in kelompokkan(uji).values():
        prof, _ = bentuk_profil(riwayat, 4, cfg.kurikulum.horizon_semester, set(kur.kode))
        diambil = set().union(*(t.kode for t in riwayat if t.semester_ke <= 4))
        gagal_akhir = diambil - prof.mk_lulus
        assert prof.mk_pernah_gagal == gagal_akhir and prof.semester_saat_ini == 5


def test_generator_tidak_bergantung_pythonhashseed(tmp_path):
    """Reprodusibel (KNF-09): seed tetap → riwayat identik walau urutan iterasi set berbeda antarproses."""
    import hashlib
    import subprocess
    import sys
    from pathlib import Path

    akar = Path(__file__).resolve().parents[1]
    kode = ("import sys, pathlib; sys.path.insert(0, sys.argv[1]); from tests.conftest import buat_cfg; "
            "from src.sintetis import generate; t = pathlib.Path(sys.argv[2]); "
            "generate(buat_cfg(t, generator={'n_alumni': 5}), t / 'o')")
    sidik = []
    for seed in ("0", "3"):
        out = tmp_path / seed
        out.mkdir()
        subprocess.run([sys.executable, "-c", kode, str(akar), str(out)], check=True, capture_output=True,
                       env={**__import__("os").environ, "PYTHONHASHSEED": seed})
        sidik.append(hashlib.md5((out / "o" / "histori_krs.csv").read_bytes()).hexdigest())
    assert sidik[0] == sidik[1]
