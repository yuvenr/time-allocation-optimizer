"""Tes iterasi 3: preferensi MK pilihan per tema (planner, kedua mode) dan penyesuaian manual rencana
(``src/penyesuaian.py``: geser MK antarsemester dengan validasi aturan keras)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.penyesuaian import (kembalikan_rekomendasi, pindahkan, semester_papan, semester_tujuan_sah,
                             sesuaikan_rencana, susunan_rencana)
from src.planner import susun_rencana
from tests.test_planner import ATURAN_KERAS, ctx, ctx_dasar, kode_rencana, profil, semester_dari  # noqa: F401

SKRIPSI, PLP1, PLP2 = "UBU60001", "CIE61111", "CIE61112"


@pytest.fixture(scope="module")
def rencana_s5(ctx):
    """Jalur A ideal sampai S4, IP 3,20: S7 berisi PLP 1+2, PKM, Skripsi (lihat struktur 08)."""
    return susun_rencana(profil(ctx.kurikulum, "A", 5, 3.2), "arm", ctx)


def _keras(plan) -> list[str]:
    return [p for p in plan.validasi.pelanggaran if p.split(":")[0] in ATURAN_KERAS]


# ---- preferensi MK pilihan per tema ----------------------------------------------------------


@pytest.mark.parametrize("mode", ["arm", "hybrid"])
def test_preferensi_tema_dipakai_kedua_mode(ctx, mode):
    kur = ctx.kurikulum
    tanpa = susun_rencana(profil(kur, "A", 3, 3.2), mode, ctx)
    assert "CIE61402" in kode_rencana(tanpa) and "CIE61403" not in kode_rencana(tanpa)   # default sistem Tema A
    p = susun_rencana(profil(kur, "A", 3, 3.2, pilihan_tema={"A": "CIE61403"}), mode, ctx)
    kode = kode_rencana(p)
    assert "CIE61403" in kode and "CIE61402" not in kode
    item = next(i for s in p.semester for i in s.items if i.kode == "CIE61403")
    assert any("Dipilih mahasiswa untuk Tema A" in a for a in item.alasan)
    assert not _keras(p)


def test_preferensi_setiap_opsi_dijadwalkan_atau_diperingatkan(ctx):
    """Properti: untuk setiap tema & setiap MK opsinya, MK pilihan mahasiswa dijadwalkan — atau rencana
    memberi peringatan — dan MK lain dari tema itu tidak pernah direkomendasikan; aturan keras tetap lolos."""
    kur = ctx.kurikulum
    for tema in sorted(kur.tema):
        for kode in sorted(kur.mk_tema(tema)):
            p = susun_rencana(profil(kur, "A", 3, 3.2, pilihan_tema={tema: kode}), "arm", ctx)
            rencana = set(kode_rencana(p))
            dijadwalkan = kode in rencana
            diperingatkan = any(w.startswith(f"Tema {tema}: MK pilihan yang dipilih") for w in p.peringatan)
            assert dijadwalkan or diperingatkan, (tema, kode)
            assert not (rencana & (kur.mk_tema(tema) - {kode})), (tema, kode)
            assert not _keras(p), (tema, kode, _keras(p))


def test_preferensi_tidak_sah_diabaikan(ctx):
    kur = ctx.kurikulum
    p = susun_rencana(profil(kur, "A", 3, 3.2, pilihan_tema={"A": "CIE61404", "Z": "CIE61402"}), "arm", ctx)
    assert sum("diabaikan" in w for w in p.peringatan) == 2
    assert "CIE61402" in kode_rencana(p)        # Tema A kembali ke rekomendasi sistem


# ---- penyesuaian manual ---------------------------------------------------------------------


def test_struktur_rencana_uji(rencana_s5):
    assert {SKRIPSI, PLP1, PLP2} <= set(susunan_rencana(rencana_s5)[7])
    assert rencana_s5.validasi.lolos


def test_skripsi_digeser_ke_semester_8_diterima(ctx, rencana_s5):
    r = pindahkan(rencana_s5, SKRIPSI, 8, ctx)
    assert r.diterima and not r.ditolak
    p = r.plan
    assert semester_dari(p, SKRIPSI) == 8 and p.validasi.lolos
    s7, s8 = (next(s for s in p.semester if s.semester == x) for x in (7, 8))
    lama7 = next(s for s in rencana_s5.semester if s.semester == 7)
    assert s7.total_sks == lama7.total_sks - 6 and s8.total_sks == 6
    assert [(x.kode, x.dari, x.ke, x.otomatis) for x in p.penyesuaian] == [(SKRIPSI, 7, 8, False)]
    assert p.disesuaikan and susunan_rencana(rencana_s5) == {s.semester: [i.kode for i in s.items]
                                                             for s in p.semester_sistem if s.jenis == "reguler"}
    item = next(i for i in s8.items if i.kode == SKRIPSI)
    assert item.alasan[0] == "Digeser manual oleh mahasiswa: rekomendasi sistem S7 → S8"
    assert all("S8" in t for t in item.sumber + item.perlu_verifikasi if t.startswith("AD-"))
    assert not rencana_s5.disesuaikan            # rencana asli tidak berubah (salinan)


def test_plp_berpasangan_dan_paritas(ctx, rencana_s5):
    r = pindahkan(rencana_s5, PLP1, 8, ctx)      # PLP ditawarkan di semester ganjil saja
    assert not r.diterima and any("AD-04: S8" in t for t in r.ditolak)
    assert r.plan is rencana_s5
    r = pindahkan(rencana_s5, PLP1, 9, ctx)
    assert r.diterima, r.ditolak
    assert semester_dari(r.plan, PLP1) == semester_dari(r.plan, PLP2) == 9
    assert {(x.kode, x.otomatis) for x in r.pemindahan} == {(PLP1, False), (PLP2, True)}
    assert any("korequisit" in c for c in r.catatan)
    assert any(w.startswith("Rencana melewati semester 8") for w in r.plan.peringatan)


def test_batas_sks_ditolak(ctx, rencana_s5):
    penuh = max((s for s in rencana_s5.semester if s.jenis == "reguler" and s.semester % 2 == 1),
                key=lambda s: s.total_sks)
    sisa = penuh.batas_sks - penuh.total_sks
    sumber = next(s for s in rencana_s5.semester if s.semester % 2 == 1 and s.semester != penuh.semester
                  and s.jenis == "reguler")
    kode = next(i.kode for i in sumber.items if i.sks > sisa and i.kode not in (PLP1, PLP2))
    r = pindahkan(rencana_s5, kode, penuh.semester, ctx)
    assert not r.diterima and any(f"AD-02: S{penuh.semester}" in t for t in r.ditolak)


def test_prasyarat_tidak_boleh_melewati_mk_yang_membutuhkan(ctx):
    kur = ctx.kurikulum
    plan = susun_rencana(profil(kur, "A", 3, 3.2), "arm", ctx)
    sem = {k: s for s, ks in susunan_rencana(plan).items() for k in ks}
    pasangan = [(p.kode_syarat, k) for k in sem for p in kur.prasyarat_nilai(k)
                if p.kode_syarat in sem and sem[p.kode_syarat] < sem[k]]
    assert pasangan, "rencana uji harus memuat prasyarat & MK yang membutuhkannya"
    for syarat, k in pasangan:
        r = pindahkan(plan, syarat, sem[k], ctx)
        assert not r.diterima and any(t.startswith(k) and "AD-03" in t for t in r.ditolak), (syarat, k, r.ditolak)


def test_semester_paket_terkunci(ctx):
    plan = susun_rencana(profil(ctx.kurikulum, "A", 2, 3.2), "arm", ctx)
    assert 2 not in semester_papan(plan, ctx)
    kode = susunan_rencana(plan)[4][0]
    r = pindahkan(plan, kode, 2, ctx)
    assert not r.diterima and "paket (AD-08)" in r.ditolak[0]
    assert 2 not in semester_tujuan_sah(plan, kode, ctx)


def test_semester_tujuan_sah_konsisten(ctx, rencana_s5):
    sah = semester_tujuan_sah(rencana_s5, SKRIPSI, ctx)
    assert 8 in sah and 7 not in sah and all(s >= 5 for s in sah)
    assert all(pindahkan(rencana_s5, SKRIPSI, s, ctx).diterima for s in sah)
    assert 8 not in semester_tujuan_sah(rencana_s5, PLP1, ctx)


def test_susunan_tidak_sah_ditolak_sebagai_galat(ctx, rencana_s5):
    sus = susunan_rencana(rencana_s5)
    sus[7] = sus[7][1:]
    with pytest.raises(ValueError):
        sesuaikan_rencana(rencana_s5, sus, ctx)


def test_tanpa_perubahan(ctx, rencana_s5):
    r = sesuaikan_rencana(rencana_s5, susunan_rencana(rencana_s5), ctx)
    assert r.diterima and not r.berubah and r.plan is rencana_s5


def test_kembalikan_rekomendasi(ctx, rencana_s5):
    p = pindahkan(rencana_s5, SKRIPSI, 8, ctx).plan
    p = pindahkan(p, SKRIPSI, 10, ctx).plan      # penyesuaian kedua: susunan sistem tetap yang asli
    assert [x.ke for x in p.penyesuaian] == [8, 10]
    item = next(i for s in p.semester for i in s.items if i.kode == SKRIPSI)
    assert [a for a in item.alasan if a.startswith("Digeser")] == ["Digeser manual oleh mahasiswa: rekomendasi "
                                                                    "sistem S7 → S10"]
    asli = kembalikan_rekomendasi(p, ctx)
    assert not asli.disesuaikan and not asli.penyesuaian
    assert susunan_rencana(asli) == susunan_rencana(rencana_s5)
    assert asli.validasi.pelanggaran == rencana_s5.validasi.pelanggaran
    assert not any(w.startswith("Rencana disesuaikan") for w in asli.peringatan)


def test_status_dpa_kembali_belum_ditinjau(ctx, rencana_s5):
    disetujui = rencana_s5.model_copy(update={"status_dpa": "disetujui"})
    r = pindahkan(disetujui, SKRIPSI, 8, ctx)
    assert r.plan.status_dpa == "belum_ditinjau" and any("DPA" in c for c in r.catatan)


def test_theta_hybrid_catatan_atau_ditolak(ctx, rencana_s5):
    """Beban semester > θ (mode hybrid) = catatan; ditolak bila penyesuaian.theta_keras."""
    s8 = semester_dari(rencana_s5, SKRIPSI)
    hybrid = rencana_s5.model_copy(update={"mode": "hybrid", "theta": 1.0})
    r = pindahkan(hybrid, SKRIPSI, s8 + 1, ctx)
    assert r.diterima and any("> θ 1.0" in c for c in r.catatan)
    cfg_keras = ctx.cfg.model_copy(update={"penyesuaian": ctx.cfg.penyesuaian.model_copy(update={"theta_keras": True})})
    r = pindahkan(hybrid, SKRIPSI, s8 + 1, replace(ctx, cfg=cfg_keras))
    assert not r.diterima and any("> θ 1.0" in t for t in r.ditolak)
