"""Tes ARM (UC-08; CLAUDE.md §6): hitungan manual, pemeriksaan silang terhadap oracle brute-force,
kasus tepi ambang, determinisme, filter konsekuen wajib, dan model untuk planner."""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from mlxtend.frequent_patterns import fpgrowth

from src import db
from src.arm import (ModelARM, _one_hot, jalankan_arm, jumlah_minimum, sensitivity_analysis, support_item,
                     tambang)
from src.config import load_config
from src.data_loader import load_kurikulum, load_transaksi
from src.preprocessing import jalankan_preprocessing
from src.schemas import Transaksi

CFG = load_config(use_override_file=False, use_env=False)
KUR = load_kurikulum(CFG)


def _tr(semester, *kode_list):
    return [Transaksi(id_anon=f"x{i}", angkatan=2020, semester_ke=semester, ip_semester=3.0,
                      items=[f"{k}@S{semester}" for k in kode]) for i, kode in enumerate(kode_list)]


# ---- oracle: enumerasi brute-force murni Python, aritmetika pecahan eksak ----------------------


def oracle(item_lists: list[list[str]], min_support, min_conf, min_lift, max_len) -> dict:
    """{(anteseden, konsekuen): (support, confidence, lift)} — definisi buku teks, tanpa mlxtend."""
    n = len(item_lists)
    if n == 0:
        return {}
    ts = [frozenset(t) for t in item_lists]
    items = sorted(set().union(*ts))
    hitung: dict[frozenset, int] = {}
    for r in range(1, (max_len or len(items)) + 1):
        for combo in itertools.combinations(items, r):
            c = sum(1 for t in ts if set(combo) <= t)
            if Fraction(c, n) >= Fraction(repr(float(min_support))):
                hitung[frozenset(combo)] = c
    hasil = {}
    for itemset, c in hitung.items():
        for r in range(1, len(itemset)):
            for ant in itertools.combinations(sorted(itemset), r):
                a = frozenset(ant)
                k = itemset - a
                conf = Fraction(c, hitung[a])
                lift = Fraction(c * n, hitung[a] * hitung[k])
                if conf >= Fraction(repr(float(min_conf))) and lift >= Fraction(repr(float(min_lift))):
                    hasil[(";".join(sorted(a)), ";".join(sorted(k)))] = (c / n, float(conf), float(lift))
    return hasil


def _sebagai_dict(aturan: pd.DataFrame) -> dict:
    return {(r.anteseden, r.konsekuen): (r.support, r.confidence, r.lift) for r in aturan.itertuples()}


def _cocok(aturan: pd.DataFrame, harap: dict):
    got = _sebagai_dict(aturan)
    assert set(got) == set(harap), (sorted(set(got) - set(harap))[:5], sorted(set(harap) - set(got))[:5])
    for k, v in harap.items():
        assert got[k] == pytest.approx(v, abs=1e-6), k


def _transaksi_acak(rng, n, n_item=6, semester=3):
    p = rng.uniform(0.1, 0.8, n_item)
    kode = [f"CIE6{i:04d}" for i in range(n_item)]
    return [Transaksi(id_anon=f"a{j}", angkatan=2020, semester_ke=semester, ip_semester=3.0,
                      items=[f"{k}@S{semester}" for k, x in zip(kode, rng.random(n_item) < p) if x])
            for j in range(n)]


@pytest.mark.parametrize("seed", range(12))
def test_tambang_identik_dengan_oracle(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.choice([7, 10, 13, 20, 37]))
    tr = _transaksi_acak(rng, n)
    for sup, conf, lift, max_len in [(0.1, 0.5, 1.0, 3), (0.2, 0.65, 1.0, 3), (0.3, 0.8, 0.0, 2),
                                     (0.15, 0.5, 1.1, None)]:
        aturan, _ = tambang(tr, sup, conf, lift, max_len, partisi=True)
        _cocok(aturan, oracle([t.items for t in tr], sup, conf, lift, max_len))


def test_ambang_support_tepat_di_batas():
    """Regresi: mlxtend mentah (ceil(0,07 × 100) = 8) kehilangan itemset dengan support tepat 0,07."""
    rng = np.random.default_rng(0)
    selisih_mentah = 0
    for _ in range(40):
        tr = _transaksi_acak(rng, 100, n_item=5)
        aturan, _ = tambang(tr, 0.07, 0.1, 0.0, 3)
        harap = oracle([t.items for t in tr], 0.07, 0.1, 0.0, 3)
        _cocok(aturan, harap)
        df = _one_hot([t.items for t in tr])
        mentah = {frozenset(s) for s in fpgrowth(df, min_support=0.07, use_colnames=True, max_len=3).itemsets}
        eksak = {frozenset(a.split(";")) | frozenset(k.split(";")) for a, k in harap}
        selisih_mentah += bool(eksak - mentah)
    assert selisih_mentah > 0, "kasus uji harus memuat itemset tepat di ambang yang terlewat mlxtend mentah"
    assert jumlah_minimum(0.07, 100) == 7 and jumlah_minimum(0.3, 10) == 3 and jumlah_minimum(0.001, 5) == 1


def test_ambang_confidence_dan_lift_inklusif():
    """n=10: A=5, B=4, AB=3 → A→B conf 3/5 = 0,6 tepat di ambang 0,6 → lolos; ambang 0,6000001 → tidak."""
    tr = _tr(3, "AB", "AB", "AB", "A", "A", "B", "C", "C", "C", "C")
    d = _sebagai_dict(tambang(tr, 0.3, 0.6, 1.0, 2)[0])
    assert d[("A@S3", "B@S3")] == pytest.approx((0.3, 0.6, 1.5))
    assert ("A@S3", "B@S3") not in _sebagai_dict(tambang(tr, 0.3, 0.6000001, 1.0, 2)[0])
    # lift tepat 1,5 di ambang 1,5 → lolos bila inklusif; dibuang bila ketat (Gambar 2.4: lift > ambang)
    assert ("A@S3", "B@S3") in _sebagai_dict(tambang(tr, 0.3, 0.1, 1.5, 2)[0])
    assert ("A@S3", "B@S3") not in _sebagai_dict(tambang(tr, 0.3, 0.1, 1.5, 2, lift_ketat=True)[0])
    assert ("A@S3", "B@S3") in _sebagai_dict(tambang(tr, 0.3, 0.1, 1.4999, 2, lift_ketat=True)[0])


def test_lift_tepat_satu_dibuang_bila_ketat():
    """n=4: A=2, B=2, AB=1 → A→B conf 0,5, lift 0,5/0,5 = 1,0 (independen). Default config (lift_ketat) membuangnya."""
    tr = _tr(3, "AB", "A", "B", "C")
    assert ("A@S3", "B@S3") in _sebagai_dict(tambang(tr, 0.25, 0.5, 1.0, 2)[0])
    assert ("A@S3", "B@S3") not in _sebagai_dict(tambang(tr, 0.25, 0.5, 1.0, 2, lift_ketat=True)[0])
    assert load_config(use_override_file=False, use_env=False).arm.lift_ketat


# ---- kasus tepi ----------------------------------------------------------------------------


def test_kasus_tepi_kosong_dan_item_tunggal():
    aturan, _ = tambang([], 0.2, 0.5)
    assert aturan.empty and list(aturan.columns) == ["anteseden", "konsekuen", "support", "confidence", "lift",
                                                     "sumber", "run_id"]
    assert support_item([]).empty
    kosong = [Transaksi(id_anon=f"k{i}", angkatan=2020, semester_ke=3, ip_semester=0.0, items=[]) for i in range(3)]
    aturan, info = tambang(kosong, 0.2, 0.5)
    assert aturan.empty and info["S3"]["n_transaksi"] == 3
    assert tambang(_tr(3, "A", "A", "B"), 0.1, 0.1)[0].empty


def test_transaksi_kosong_ikut_dihitung_dalam_support():
    """Mahasiswa yang tidak lulus satu MK pun di semester itu tetap bagian populasi semester."""
    tr = _tr(3, "AB", "AB") + [Transaksi(id_anon="k", angkatan=2020, semester_ke=3, ip_semester=0.0, items=[])]
    assert support_item(tr).set_index("item").loc["A@S3", "support"] == pytest.approx(2 / 3)
    d = _sebagai_dict(tambang(tr, 0.5, 0.5, 1.0)[0])
    assert d[("A@S3", "B@S3")] == pytest.approx((2 / 3, 1.0, 1.5))


def test_item_support_satu_dan_max_len():
    """Item paket (support 1) → lift = 1; max_len membatasi ukuran itemset aturan."""
    tr = _tr(1, "PAB", "PAB", "PA", "PB", "P")
    d = _sebagai_dict(tambang(tr, 0.2, 0.1, 1.0, 3)[0])
    assert d[("A@S1", "P@S1")][2] == pytest.approx(1.0)
    assert any(len(a.split(";")) + len(k.split(";")) == 3 for a, k in d)
    d2 = _sebagai_dict(tambang(tr, 0.2, 0.1, 1.0, 2)[0])
    assert d2 and all(len(a.split(";")) + len(k.split(";")) <= 2 for a, k in d2)


def test_partisi_semester_dan_tanpa_partisi():
    tr = _tr(3, "AB", "AB", "AB", "AC", "BC", "C") + _tr(4, "DE", "DE", "D", "E")
    sp = support_item(tr, partisi=True).set_index("item")
    sg = support_item(tr, partisi=False).set_index("item")
    assert sp.loc["A@S3", "support"] == pytest.approx(4 / 6) and sg.loc["A@S3", "support"] == pytest.approx(4 / 10)
    aturan, _ = tambang(tr, 0.2, 0.1, 0.0, 3, partisi=True)
    for r in aturan.itertuples():   # aturan tidak pernah melintasi semester
        assert len({i.split("@")[1] for i in (r.anteseden + ";" + r.konsekuen).split(";")}) == 1
    for s in (3, 4):
        bagian = aturan[aturan.anteseden.str.endswith(f"S{s}")]
        _cocok(bagian, oracle([t.items for t in tr if t.semester_ke == s], 0.2, 0.1, 0.0, 3))
    global_, _ = tambang(tr, 0.2, 0.1, 0.0, 3, partisi=False)
    _cocok(global_, oracle([t.items for t in tr], 0.2, 0.1, 0.0, 3))


def test_deterministik():
    rng = np.random.default_rng(7)
    tr = _transaksi_acak(rng, 50) + _transaksi_acak(rng, 30, semester=5)
    a1, _ = tambang(tr, 0.1, 0.3, 0.0, 3, run_id="x")
    a2, _ = tambang(list(reversed(tr)), 0.1, 0.3, 0.0, 3, run_id="x")
    pd.testing.assert_frame_equal(a1, a2)


def test_filter_konsekuen_mk_wajib():
    """§6: aturan berkonsekuen MK wajib (paket) dibuang; konsekuen pilihan dipertahankan."""
    tr = _tr(5, ["CIE61108", "CIE61402"], ["CIE61108", "CIE61402"], ["CIE61108", "CIE61404"], ["CIE61108"])
    semua, _ = tambang(tr, 0.2, 0.1, 0.0, 2)
    tersaring, info = tambang(tr, 0.2, 0.1, 0.0, 2, kurikulum=KUR, buang_konsekuen_wajib=True)
    assert any(k == "CIE61108@S5" for k in semua["konsekuen"])
    assert not any(k.startswith("CIE61108") for k in tersaring["konsekuen"])
    assert ("CIE61108@S5", "CIE61402@S5") in _sebagai_dict(tersaring)
    assert info["_filter"]["n_aturan_konsekuen_wajib_dibuang"] == len(semua) - len(tersaring) > 0


def test_sensitivitas_cocok_dengan_oracle_dan_monoton():
    rng = np.random.default_rng(3)
    tr = _transaksi_acak(rng, 25)
    tabel = sensitivity_analysis(tr, CFG)
    a = CFG.arm
    assert len(tabel) == len(a.sensitivitas.support) * len(a.sensitivitas.confidence)
    for r in tabel.itertuples():
        harap = oracle([t.items for t in tr], r.min_support, r.min_confidence, a.min_lift, a.max_len)
        assert r.n_aturan == len(harap), (r.min_support, r.min_confidence)
    for _, g in tabel.groupby("min_confidence"):
        assert g.sort_values("min_support")["n_aturan"].is_monotonic_decreasing
    for _, g in tabel.groupby("min_support"):
        assert g.sort_values("min_confidence")["n_aturan"].is_monotonic_decreasing


# ---- hitungan manual fixture mini --------------------------------------------------------------


def _transaksi_mini(cfg):
    jalankan_preprocessing(cfg)
    return load_transaksi(cfg, "latih") + load_transaksi(cfg, "uji")


def test_arm_mini_hitungan_manual(cfg_mini):
    """Hitungan manual atas 20 transaksi fixture mini (5 per semester, partisi per semester; item = MK yang
    diambil, termasuk CIE61402 yang tidak lulus pada M02).

    S3: CIE61206 5/5=1.0, CIE61208 4/5=0.8, CIE61402 3/5=0.6; {61208,61402} 3/5=0.6
        CIE61402 → CIE61208: conf 0.6/0.6=1.0, lift 1.0/0.8=1.25
        CIE61208 → CIE61402: conf 0.6/0.8=0.75, lift 0.75/0.6=1.25
    S4: CIE62209 1.0, CSD60709 0.8, CSD60711 0.6, CIE61208 0.2; {60709,60711} 0.4
        CSD60711 → CSD60709: conf 0.4/0.6=0.667, lift 0.667/0.8=0.833 < 1 → dibuang oleh min_lift
        CIE61208 → CSD60709: supp 0.2 (= ambang), conf 1.0, lift 1.0/0.8=1.25
    """
    tr = _transaksi_mini(cfg_mini)
    assert len(tr) == 20
    aturan, info = tambang(tr, min_support=0.2, min_confidence=0.5, min_lift=1.0, max_len=3)
    assert {k: v["n_transaksi"] for k, v in info.items() if not k.startswith("_")} == {
        "S1": 5, "S2": 5, "S3": 5, "S4": 5}
    d = _sebagai_dict(aturan)
    assert d[("CIE61402@S3", "CIE61208@S3")] == pytest.approx((0.6, 1.0, 1.25))
    assert d[("CIE61208@S3", "CIE61402@S3")] == pytest.approx((0.6, 0.75, 1.25))
    assert set(aturan["sumber"]) == {"fixture"}
    assert d[("CIE61208@S4", "CSD60709@S4")] == pytest.approx((0.2, 1.0, 1.25))
    assert ("CSD60711@S4", "CSD60709@S4") not in d
    for s in (1, 2, 3, 4):
        _cocok(aturan[aturan.anteseden.str.endswith(f"S{s}")],
               oracle([t.items for t in tr if t.semester_ke == s], 0.2, 0.5, 1.0, 3))
    # dengan filter §6 hanya aturan berkonsekuen MK pilihan (CIE61402, CSD60711) yang tersisa
    tersaring, _ = tambang(tr, 0.2, 0.5, 1.0, 3, kurikulum=KUR, buang_konsekuen_wajib=True)
    assert {k.split("@")[0] for k in tersaring["konsekuen"]} <= {"CIE61402", "CSD60711"}
    assert ("CIE61208@S3", "CIE61402@S3") in _sebagai_dict(tersaring)

    sup = support_item(tr).set_index("item")
    for item, (s, n) in {"CIE61206@S3": (1.0, 5), "CIE61208@S3": (0.8, 4), "CIE61402@S3": (0.6, 3),
                         "CIE62209@S4": (1.0, 5), "CSD60709@S4": (0.8, 4), "CSD60711@S4": (0.6, 3),
                         "CIE61208@S4": (0.2, 1), "COM60014@S1": (1.0, 5)}.items():
        assert sup.loc[item, "support"] == pytest.approx(s) and sup.loc[item, "jumlah"] == n, item


def test_jalankan_arm_hanya_data_latih_dan_deteksi_basi(cfg_mini):
    jalankan_preprocessing(cfg_mini)
    r = jalankan_arm(cfg_mini)
    latih = load_transaksi(cfg_mini, "latih")
    uji_ids = {t.id_anon for t in load_transaksi(cfg_mini, "uji")}
    assert r["n_transaksi_latih"] == len(latih) == 16 and uji_ids and not uji_ids & {t.id_anon for t in latih}
    a = cfg_mini.arm
    ulang, info = tambang(latih, a.min_support, a.min_confidence, a.min_lift, a.max_len, kurikulum=KUR,
                          buang_konsekuen_wajib=True, lift_ketat=a.lift_ketat)
    aturan = db.baca(cfg_mini, "SELECT * FROM aturan ORDER BY lift DESC, confidence DESC, support DESC, "
                              "anteseden, konsekuen")
    assert list(aturan.columns) == ["anteseden", "konsekuen", "support", "confidence", "lift", "sumber", "run_id"]
    kunci = ["anteseden", "konsekuen", "support", "confidence", "lift"]
    pd.testing.assert_frame_equal(aturan[kunci].reset_index(drop=True), ulang[kunci].reset_index(drop=True),
                                  check_dtype=False)
    assert r["n_aturan_konsekuen_wajib_dibuang"] == info["_filter"]["n_aturan_konsekuen_wajib_dibuang"] > 0
    assert r["sidik_data"]
    m = ModelARM.dari_cfg(cfg_mini, KUR)
    assert not m.kosong and m.support["CIE61206@S3"] == pytest.approx(1.0)
    # data KRS berubah setelah penambangan → hasil dianggap basi dan tidak dipakai planner
    path = cfg_mini.raw_dir / "histori_krs.csv"
    path.write_text(path.read_text(encoding="utf-8") + "M07,Mahasiswa Fiktif 07,2021,1,COM60014,4,A,4.00,K2020,8\n",
                    encoding="utf-8")
    basi = ModelARM.dari_cfg(cfg_mini, KUR)
    assert basi.kosong and "basi" in basi.catatan


# ---- model untuk planner -----------------------------------------------------------------


def test_model_arm_skor_aturan():
    aturan = pd.DataFrame({
        "anteseden": ["CIE61101@S3", "CIE61101@S3;CIE61102@S3", "CIE61101@S3"],
        "konsekuen": ["CIE61402@S3", "CIE61402@S3", "CIE61402@S3;CIE61403@S3"],
        "support": [.3, .2, .2], "confidence": [.8, .9, .9], "lift": [1.2, 2.0, 3.0], "run_id": "t",
    })
    sup = pd.DataFrame({"item": ["CIE61402@S3"], "support": [0.4]})
    m = ModelARM.dari_df(aturan, sup)
    assert m.support["CIE61402@S3"] == 0.4
    assert m.skor_aturan("CIE61402@S3", set())[0] == 0.0
    assert m.skor_aturan("CIE61402@S3", {"CIE61101@S3"})[0] == pytest.approx(0.96)
    skor, teks = m.skor_aturan("CIE61402@S3", {"CIE61101@S3", "CIE61102@S3"})
    assert skor == pytest.approx(1.8) and "CIE61102@S3" in teks
    assert "CIE61402@S3;CIE61403@S3" not in m.per_konsekuen   # konsekuen ganda diabaikan planner
    assert ModelARM.dari_df(None, None).kosong


def test_model_arm_mengabaikan_kode_di_luar_kurikulum():
    aturan = pd.DataFrame({"anteseden": ["OLD:CIE60044@S3", "CIE61101@S3"], "konsekuen": ["CIE61402@S3"] * 2,
                           "support": [.3, .3], "confidence": [.9, .8], "lift": [2.0, 1.5], "run_id": "t"})
    sup = pd.DataFrame({"item": ["OLD:CIE60044@S3", "CIE61402@S3"], "support": [0.5, 0.4]})
    m = ModelARM.dari_df(aturan, sup, KUR)
    assert set(m.support) == {"CIE61402@S3"}
    assert [t for _, _, _, t in m.per_konsekuen["CIE61402@S3"]] == [
        "{CIE61101@S3} → CIE61402@S3 (conf 0.80, lift 1.50)"]


def test_model_arm_tanpa_hasil_penambangan(cfg_kosong):
    m = ModelARM.dari_cfg(cfg_kosong, KUR)
    assert m.kosong and m.catatan
