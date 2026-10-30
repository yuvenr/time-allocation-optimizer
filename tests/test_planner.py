"""Tes penyusun rencana (UC-02) pada Kurikulum 2024: dua mode, aturan keras §8, θ, jalur,
saran 09, AT08/AT09, PLP, agama, konversi merdeka belajar, pengaruh ARM."""
from __future__ import annotations

import itertools
from dataclasses import replace

import pytest

from src.arm import ModelARM
from src.planner import (KODE_KONVERSI, KonteksPlanner, _kesempatan_terakhir, daftar_rencana, hitung_beban, muat_rencana,
                         perbarui_catatan_dpa, perkiraan_lulus, simpan_rencana, susun_rencana, theta_struktur)
from src.rules import Keadaan, tema_terpenuhi
from src.schemas import Profile
from tests.conftest import buat_cfg

ATURAN_KERAS = ["AD-02", "AD-03", "AD-04", "AD-06", "AD-07", "AD-08", "AD-09", "AD-10", "AD-11", "AD-12", "AD-14",
                "AD-15"]
PILIHAN_A = {5: {"A": "CIE61402", "B": "CIE61404"},
             6: {"C": "CIE62305", "D": "CSD60702", "E": "CIE62116", "F": "CIE62118", "G": "CSD60716",
                 "H": "CSD60711", "I": "CIE61119"}}
PEDOMAN = {"CIE61111", "CIE61112", "UBU60002", "COM60061", "UBU60001", "UBU60005"}


@pytest.fixture(scope="module")
def ctx_dasar(tmp_path_factory) -> KonteksPlanner:
    """Kondisi saat ini: tanpa data KRS (ARM kosong); CL = heuristik KKO (baseline), θ dari struktur."""
    base = tmp_path_factory.mktemp("planner")
    (base / "raw").mkdir()
    return KonteksPlanner.muat(buat_cfg(base))


@pytest.fixture(scope="module")
def ctx(ctx_dasar) -> KonteksPlanner:
    """+ CL uji deterministik 1–6 (BUKAN hasil penelitian) dan θ dari struktur kurikulum."""
    cl = {k: float(1 + sum(map(ord, k)) % 6) for k in ctx_dasar.kurikulum.kode}
    c = replace(ctx_dasar, cl=cl, keterangan_cl={}, cl_default=3.5)
    c.info_theta = theta_struktur(c)
    c.theta = c.info_theta["theta"]
    return c


def riwayat_ideal(kur, jalur: str, s: int, agama: str = "Islam") -> set[str]:
    """MK lulus bila struktur jalur diikuti persis sampai semester s−1 (slot tema diisi PILIHAN_A)."""
    lulus = set()
    for sm in range(1, s):
        for slot in kur.slot(jalur, sm):
            if slot.jenis_slot == "MK_WAJIB":
                lulus.add(slot.kode_mk)
            elif slot.jenis_slot == "MK_WAJIB_PILIH_SATU":
                lulus.add(kur.mk_agama(agama))
            elif slot.jenis_slot == "PILIHAN_TEMA":
                lulus.add(PILIHAN_A[sm][slot.tema])
    return lulus


def profil(kur, jalur="A", s=1, ip=3.2, gagal=(), **kw) -> Profile:
    lulus = riwayat_ideal(kur, jalur, s) - set(gagal)
    sudah_konversi = jalur != "A" and s > 6
    tk = kw.pop("tema_konversi", {"C", "D", "E", "F", "G", "I"} if sudah_konversi else None)
    return Profile(semester_saat_ini=s, ip_terakhir=ip, agama=kw.pop("agama", "Islam"), jalur=jalur,
                   mk_lulus=lulus, mk_pernah_gagal=set(gagal), tema_konversi=tk,
                   sks_konversi=20 if sudah_konversi else 0, **kw)


def grid_profil(kur) -> list[Profile]:
    hasil = []
    for jalur, s, ip in itertools.product("ABF", (1, 3, 4, 5, 6, 7, 8), (3.6, 2.7, 2.2, 1.7, 1.2)):
        hasil.append(profil(kur, jalur, s, ip))
        if s >= 4:  # varian: satu MK wajib semester sebelumnya tidak lulus
            wajib = sorted(sl.kode_mk for sl in kur.slot(jalur, s - 1) if sl.jenis_slot == "MK_WAJIB")
            if wajib:
                hasil.append(profil(kur, jalur, s, ip, gagal={wajib[0]}))
    return hasil


def kode_rencana(plan) -> list[str]:
    return [i.kode for s in plan.semester for i in s.items if i.jenis == "mk"]


def semester_dari(plan, kode) -> int | None:
    return next((s.semester for s in plan.semester for i in s.items if i.kode == kode), None)


# ---- tes properti: aturan keras §8 untuk seluruh grid profil, kedua mode ----------------------


@pytest.mark.parametrize("mode", ["arm", "hybrid"])
def test_grid_profil_patuh_aturan_keras(ctx, mode):
    kur = ctx.kurikulum
    profil_list = grid_profil(kur)
    assert len(profil_list) > 150
    for prof in profil_list:
        plan = susun_rencana(prof, mode, ctx)
        v = plan.validasi
        for kode in ATURAN_KERAS:
            assert v.per_aturan[kode], (prof.jalur, prof.semester_saat_ini, prof.ip_terakhir, v.pelanggaran)
        semua = kode_rencana(plan)
        assert len(semua) == len(set(semua)), "MK direkomendasikan dua kali"
        assert not set(semua) & prof.mk_lulus, "MK yang sudah lulus direkomendasikan lagi"
        assert "UBU60030" not in semua and "UBU60029" not in semua
        for s in plan.semester:
            plp = {"CIE61111", "CIE61112"} & {i.kode for i in s.items}
            if len(plp) == 1:   # boleh sendiri hanya bila pasangannya sudah lulus sebelumnya
                assert ({"CIE61111", "CIE61112"} - plp) <= prof.mk_lulus, "PLP tanpa korequisit"
            assert s.batas_sks is not None and s.total_sks <= s.batas_sks, "melampaui pita SKS"
            if mode == "hybrid" and s.jenis == "reguler":
                assert s.total_beban <= ctx.theta + 1e-9
            for it in s.items:
                assert it.alasan and it.sumber, f"{it.kode}: alasan/sumber kosong"
                if it.jenis == "mk" and it.cl is not None and it.kode in ctx.cl:   # KF-17: justifikasi per CPMK
                    assert it.justifikasi_cpmk and all(j.startswith(it.kode) for j in it.justifikasi_cpmk)
                if it.kode in PEDOMAN:
                    assert any("AD-16" in x for x in it.perlu_verifikasi)
        if not (v.per_aturan["AD-01"] and v.per_aturan["AD-05"] and v.per_aturan["AD-13"]):
            assert any("Syarat lulus belum tercapai" in p for p in plan.peringatan)


def test_theta_tak_hingga_hybrid_identik_arm(ctx):
    """Perbedaan kedua mode HANYA berasal dari θ."""
    c = replace(ctx, theta=1e12)
    for prof in grid_profil(ctx.kurikulum)[::5]:
        a, h = susun_rencana(prof, "arm", c), susun_rencana(prof, "hybrid", c)
        assert [[i.kode for i in s.items] for s in a.semester] == [[i.kode for i in s.items] for s in h.semester]


def test_theta_kecil_melewati_kandidat(ctx):
    c = replace(ctx, theta=40.0)
    plan = susun_rencana(profil(ctx.kurikulum, "A", 3, 3.2), "hybrid", c)
    assert any(s.dilewati_theta for s in plan.semester)
    assert all(s.total_beban <= 40.0 + 1e-9 for s in plan.semester if s.jenis == "reguler")
    for kode in ATURAN_KERAS:
        assert plan.validasi.per_aturan[kode], plan.validasi.pelanggaran


def test_hybrid_tanpa_theta_ditolak(ctx_dasar, ctx):
    assert ctx_dasar.arm.kosong and ctx_dasar.theta is not None           # CL heuristik → θ struktur
    assert ctx_dasar.info_cl["sumber_rating"] == "heuristik" and not ctx_dasar.info_theta["dikalibrasi_data"]
    with pytest.raises(ValueError, match="θ"):
        susun_rencana(profil(ctx_dasar.kurikulum), "hybrid", replace(ctx_dasar, theta=None))
    with pytest.raises(ValueError):
        susun_rencana(profil(ctx.kurikulum), "lain", ctx)


# ---- perilaku spesifik terhadap dokumen ----------------------------------------------------


def test_mahasiswa_ideal_jalur_a(ctx_dasar):
    """IP 3,2 sejak S1: saran 09 'tambah' (Tema A S3, Tema I S4, Tema H S5, Capstone S6, Skripsi+PKM S7)
    dan AT08 (Tema C menggantikan Tema A di S5) → lulus 145 SKS di S7, semua aturan lolos."""
    kur = ctx_dasar.kurikulum
    plan = susun_rencana(profil(kur, "A", 1, 3.2), "arm", ctx_dasar)
    assert plan.validasi.lolos, plan.validasi.pelanggaran
    assert [s.semester for s in plan.semester] == list(range(1, 9))
    tema = {s.semester: {kur.mk[i.kode].tema for i in s.items if i.jenis == "mk" and kur.mk[i.kode].tema}
            for s in plan.semester}
    assert tema[3] == {"A"} and tema[4] == {"I"} and tema[5] == {"B", "C", "H"} and tema[6] == {"D", "E", "F", "G"}
    s5 = plan.semester[4]
    assert any("AT08" in c for c in s5.catatan)
    assert semester_dari(plan, "COM60061") == 6
    assert semester_dari(plan, "UBU60001") == 7 and semester_dari(plan, "UBU60005") == 7
    assert sum(s.total_sks for s in plan.semester) == 145
    assert [s.total_sks for s in plan.semester] == [20, 20, 24, 24, 24, 15, 18, 0]
    assert any("ARM belum aktif" in a for a in plan.asumsi)


def test_ip_rendah_mengikuti_saran_kurangi_09(ctx_dasar):
    """IP 2,2 di S3: 09 'kurangi CIE61208' → 18 SKS; CIE61208 tertunda diprioritaskan di S5 (AT09)."""
    kur = ctx_dasar.kurikulum
    plan = susun_rencana(profil(kur, "A", 3, 2.2), "arm", ctx_dasar)
    s3 = plan.semester[0]
    assert s3.total_sks == 18 and "CIE61208" not in {i.kode for i in s3.items}
    assert semester_dari(plan, "CIE61208") == 5
    item = next(i for s in plan.semester for i in s.items if i.kode == "CIE61208")
    assert "AT09" in (item.prioritas or "")
    assert plan.validasi.lolos, plan.validasi.pelanggaran


def test_ip_sangat_rendah_semester_3(ctx_dasar):
    """IP < 1,50 di S3: kurangi 4 MK; pita ≤ 12 SKS → CIE61207 (2 SKS) masih muat = 12 SKS (INK21)."""
    kur = ctx_dasar.kurikulum
    plan = susun_rencana(profil(kur, "A", 3, 1.2), "arm", ctx_dasar)
    s3 = plan.semester[0]
    assert {i.kode for i in s3.items} == {"CIE61101", "CIE61102", "CIE61103", "CIE61207"}
    assert s3.total_sks == 12 and any("INK21" in c for c in s3.catatan)
    assert plan.semester[-1].semester > 8 and any("melewati semester 8" in p for p in plan.peringatan)


@pytest.mark.parametrize("ip,harap", [(2.7, 7), (2.2, 8)])
def test_skripsi_mengikuti_saran_09(ctx_dasar, ip, harap):
    """09 Jalur A S7: IP ≥ 2,50 'tambah UBU60001;UBU60005'; IP < 2,50 'tetap' → Skripsi di S8."""
    plan = susun_rencana(profil(ctx_dasar.kurikulum, "A", 7, ip), "arm", ctx_dasar)
    assert semester_dari(plan, "UBU60001") == harap
    assert plan.validasi.lolos, plan.validasi.pelanggaran


def test_plp_menunggu_seluruh_mk_kependidikan(ctx_dasar):
    """CIE61110 (Kependidikan, ganjil) tidak lulus → diulang di S7; PLP baru boleh di S9."""
    plan = susun_rencana(profil(ctx_dasar.kurikulum, "A", 7, 3.2, gagal={"CIE61110"}), "arm", ctx_dasar)
    assert semester_dari(plan, "CIE61110") == 7
    assert semester_dari(plan, "CIE61111") == semester_dari(plan, "CIE61112") == 9
    assert plan.validasi.per_aturan["AD-11"] and plan.validasi.lolos, plan.validasi.pelanggaran
    assert any("melewati semester 8" in p and "CIE61111 (S9)" in p and "belum terjadwal hingga S8" in p
               for p in plan.peringatan)


def test_agama_tidak_disimpulkan(ctx_dasar):
    plan = susun_rencana(profil(ctx_dasar.kurikulum, "A", 1, 3.2, agama=None), "arm", ctx_dasar)
    semua = set(kode_rencana(plan))
    assert not semua & ctx_dasar.kurikulum.grup("AGAMA")
    assert any("Agama mahasiswa belum diisi" in p for p in plan.peringatan)
    assert any(h.kode == "AD-12" and h.status == "perlu_verifikasi_manual" for h in plan.validasi.rincian)
    plan_k = susun_rencana(profil(ctx_dasar.kurikulum, "A", 1, 3.2, agama="Katholik"), "arm", ctx_dasar)
    assert set(kode_rencana(plan_k)) & ctx_dasar.kurikulum.grup("AGAMA") == {"MPK60002"}


def test_jalur_b_konversi_dan_sisa_tema(ctx_dasar):
    """Jalur B, IP 2,7: konversi S6 mencakup 6 tema; satu tema (ditawarkan di S7) disisakan untuk slot
    pilihan sisa tema S7 → total 147 SKS (INK18)."""
    kur = ctx_dasar.kurikulum
    plan = susun_rencana(profil(kur, "B", 1, 2.7), "arm", ctx_dasar)
    s6 = plan.semester[5]
    assert s6.jenis == "konversi" and [i.kode for i in s6.items] == [KODE_KONVERSI] and s6.total_sks == 20
    konv = s6.items[0]
    assert len(konv.tema_konversi) == 6 and any("AD-14" in x for x in konv.perlu_verifikasi)
    sisa = [kur.mk[i.kode].tema for i in plan.semester[6].items if kur.mk[i.kode].sifat == "pilihan"]
    assert len(sisa) == 1 and sisa[0] not in konv.tema_konversi
    assert sum(s.total_sks for s in plan.semester) == 147
    assert plan.validasi.lolos, plan.validasi.pelanggaran
    assert any(h.kode == "AD-14" and h.status == "perlu_verifikasi_manual" for h in plan.validasi.rincian)


def test_jalur_b_tema_konversi_input(ctx_dasar):
    kur = ctx_dasar.kurikulum
    plan = susun_rencana(profil(kur, "B", 6, 3.2, tema_konversi={"D", "E"}), "arm", ctx_dasar)
    assert plan.semester[0].items[0].tema_konversi == ["D", "E"]
    lulus = plan.profil.mk_lulus | set(kode_rencana(plan))
    assert tema_terpenuhi(lulus, {"D", "E"}, kur) == set(kur.tema)


def test_jalur_b_tanpa_syarat_at12(ctx_dasar):
    """CSD60708 (wajib S5) belum lulus → konversi S6 tidak boleh (AT12); S6 direncanakan reguler."""
    plan = susun_rencana(profil(ctx_dasar.kurikulum, "B", 6, 3.2, gagal={"CSD60708"}), "arm", ctx_dasar)
    assert plan.semester[0].jenis == "reguler"
    assert KODE_KONVERSI not in {i.kode for s in plan.semester for i in s.items}
    assert any("AT12" in p for p in plan.peringatan)
    for kode in ATURAN_KERAS:
        assert plan.validasi.per_aturan[kode], plan.validasi.pelanggaran


def test_mahasiswa_semester_lanjut(ctx_dasar):
    """Semester 9 (masa studi s.d. AT04 = 14): rencana dimulai di S9."""
    kur = ctx_dasar.kurikulum
    plan = susun_rencana(profil(kur, "A", 9, 2.8, gagal={"CIE61208", "UBU60001"}), "arm", ctx_dasar)
    assert plan.semester[0].semester == 9
    assert {semester_dari(plan, "CIE61208"), semester_dari(plan, "UBU60001")} == {9}
    assert plan.validasi.lolos, plan.validasi.pelanggaran


def test_kode_tidak_dikenal_diabaikan(ctx_dasar):
    prof = profil(ctx_dasar.kurikulum, "A", 3, 3.2)
    prof = prof.model_copy(update={"mk_lulus": prof.mk_lulus | {"CIE61010"}})
    plan = susun_rencana(prof, "arm", ctx_dasar)
    assert any("CIE61010" in p and "penyetaraan" in p for p in plan.peringatan)


# ---- ARM & beban -------------------------------------------------------------------------


def test_arm_menentukan_pilihan_dalam_tema(ctx_dasar):
    """Support alumni menentukan MK pilihan dalam satu tema (tanpa ARM: urutan kode)."""
    kur = ctx_dasar.kurikulum
    arm = ModelARM(support={"CIE61403@S3": 0.8, "CIE61402@S3": 0.1})
    plan = susun_rencana(profil(kur, "A", 3, 3.2), "arm", replace(ctx_dasar, arm=arm))
    item = next(i for i in plan.semester[0].items if kur.mk[i.kode].tema == "A")
    assert item.kode == "CIE61403" and item.support == 0.8
    assert any(a.startswith("Support CIE61403@S3") for a in item.alasan)
    tanpa = susun_rencana(profil(kur, "A", 3, 3.2), "arm", ctx_dasar)
    assert any(i.kode == "CIE61402" for i in tanpa.semester[0].items)


def test_arm_tidak_menggeser_prioritas_wajib(ctx_dasar):
    """Skor ARM hanya berlaku di dalam tingkat prioritas: MK wajib paket tetap didahulukan."""
    kur = ctx_dasar.kurikulum
    arm = ModelARM(support={"CIE61402@S3": 1.0, "CIE61403@S3": 1.0})
    plan = susun_rencana(profil(kur, "A", 3, 2.7), "arm", replace(ctx_dasar, arm=arm))   # batas 21 = paket
    assert {i.kode for i in plan.semester[0].items} == {sl.kode_mk for sl in kur.slot("A", 3)}


def test_integrasi_arm_fixture_ke_planner(cfg_mini):
    """Preprocessing + ARM pada fixture mini → planner memakai support & menandai sumber data per semester."""
    from src.arm import jalankan_arm
    from src.preprocessing import jalankan_preprocessing

    jalankan_preprocessing(cfg_mini)
    jalankan_arm(cfg_mini)
    c = KonteksPlanner.muat(cfg_mini)
    assert not c.arm.kosong and c.status_krs == "fixture"
    plan = susun_rencana(profil(c.kurikulum, "A", 3, 3.2), "arm", c)
    assert plan.sumber_data == "fixture"
    # asal data tetap tersimpan di rencana, tetapi tidak muncul di asumsi/alasan yang ditampilkan
    assert "Skor ARM dari data KRS data latih." in plan.asumsi
    s3 = plan.semester[0]
    assert any(i.support > 0 for i in s3.items)
    teks = " ".join(plan.asumsi + [a for s in plan.semester for i in s.items for a in i.alasan]).lower()
    assert "fixture" not in teks and "sintetis" not in teks
    assert plan.validasi.lolos, plan.validasi.pelanggaran


def test_beban_kegiatan_transisi_tidak_dihitung(ctx):
    plan = susun_rencana(profil(ctx.kurikulum, "A", 7, 3.2), "arm", ctx)
    s7 = plan.semester[0]
    assert {i.kode for i in s7.items} == {"CIE61111", "CIE61112", "UBU60002", "COM60061", "UBU60001", "UBU60005"}
    assert s7.total_beban == pytest.approx(ctx.cl["UBU60001"] * 6)   # PLP/PKL/Capstone/PKM dikecualikan
    plp = next(i for i in s7.items if i.kode == "CIE61111")
    assert plp.jadwal and any("transisi semester" in j for j in plp.jadwal)


def test_beban_semester():
    assert hitung_beban([(5.0, 3), (2.0, 2)]) == 19.0
    assert hitung_beban([(5.0, 3), (2.0, 2)], "rata_tertimbang") == pytest.approx(3.8)
    assert hitung_beban([]) == 0.0


def test_theta_tidak_rusak_oleh_transaksi_basi(cfg_kosong):
    """Regresi: transaksi latih lama (kode di luar Kurikulum 2024 / data berubah) tidak boleh membuat θ hilang."""
    from src.data_loader import sidik_raw, simpan_transaksi
    from src.db import simpan_ringkasan
    from src.schemas import Transaksi

    lama = [Transaksi(id_anon="a", angkatan=2018, semester_ke=3, ip_semester=3.0, items=["PTI301@S3"])]
    simpan_transaksi(cfg_kosong, "latih", lama)
    simpan_ringkasan(cfg_kosong, "preprocessing", {"sidik_data": "sidik-lama"})
    basi = KonteksPlanner.muat(cfg_kosong)
    assert basi.theta is not None and "struktur" in basi.info_theta["sumber"] and "basi" in basi.info_theta["catatan"]
    simpan_ringkasan(cfg_kosong, "preprocessing", {"sidik_data": sidik_raw(cfg_kosong)})
    tanpa_mk = KonteksPlanner.muat(cfg_kosong)
    assert tanpa_mk.theta is not None and "tidak memuat MK" in tanpa_mk.info_theta["catatan"]


def test_theta_struktur_hitungan_manual(ctx_dasar):
    """CL = 1 untuk semua MK → B_s = SKS yang dihitung. Komposisi semester unik (PLP/PKL/PKM/Capstone
    dan konversi dikecualikan): A S1 20, S2 20, S3–S6 21 ×4, S8 6; B–F S7 sisa tema 3.
    Terurut [3, 6, 20, 20, 21, 21, 21, 21] → P75 (posisi 5,25) = 21; median (posisi 3,5) = 20,5."""
    c = replace(ctx_dasar, cl={k: 1.0 for k in ctx_dasar.kurikulum.kode}, cl_default=1.0)
    info = theta_struktur(c)
    assert info["theta"] == pytest.approx(21.0) and info["n_semester"] == 8
    assert info["distribusi"] == {"min": 3.0, "p25": 16.5, "median": 20.5, "p75": 21.0, "max": 21.0}
    assert info["dikalibrasi_data"] is False


# ---- penyimpanan (UC-03..05) ---------------------------------------------------------------


def test_simpan_daftar_dan_catatan_dpa(ctx, tmp_path):
    cfg = buat_cfg(tmp_path)
    plan = susun_rencana(profil(ctx.kurikulum, "A", 3, 3.2), "hybrid", ctx, label="uji")
    simpan_rencana(plan, cfg)
    assert plan.id in [p.id for p in daftar_rencana(cfg)]
    perbarui_catatan_dpa(cfg, plan.id, "Kurangi beban semester 5.")
    p2 = muat_rencana(cfg, plan.id)
    assert p2.catatan_dpa == "Kurangi beban semester 5." and p2.catatan_dpa_diperbarui is not None
    assert p2.semester == plan.semester and p2.validasi == plan.validasi


# ---- K5: lulus tepat waktu / keterlambatan seminimal mungkin -------------------------------------


def test_perkiraan_lulus_hitungan_manual(ctx_dasar):
    """Riwayat ideal sampai S2 = 40 SKS (paket). Sisa = maks(78 SKS wajib + 27 pilihan, 145 − 40) = 105.
    IP 2,2 (18 SKS): 3 + ⌈105/18⌉ − 1 = 3 + 6 − 1 = 8. IP 1,8 (15 SKS): 3 + 7 − 1 = 9.
    IP 3,5 (24 SKS): 3 + 5 − 1 = 7 → minimal horizon 8."""
    kur = ctx_dasar.kurikulum
    kd = Keadaan.dari_profil(profil(kur, "A", 3, 2.2), kur, ctx_dasar.cfg)
    assert kd.sks_lulus(kur) == 40
    assert perkiraan_lulus(3, kd, set(), 18, ctx_dasar) == 8
    assert perkiraan_lulus(3, kd, set(), 15, ctx_dasar) == 9
    assert perkiraan_lulus(3, kd, set(), 24, ctx_dasar) == 8


def test_kesempatan_terakhir_tema_ganjil(ctx_dasar):
    """Tema B hanya ditawarkan di semester ganjil ≥ 5 (CIE61404, CSD60701): di S7 dengan target lulus S8,
    S7 adalah kesempatan terakhir. Tema C masih ditawarkan di S8 (CIE62305/CIE62306, genap ≥ 6)."""
    kur = ctx_dasar.kurikulum
    kd = Keadaan.dari_profil(profil(kur, "A", 7, 2.2), kur, ctx_dasar.cfg)
    assert _kesempatan_terakhir("CIE61404", 7, kd, 8, ctx_dasar)
    assert not _kesempatan_terakhir("CIE61113", 7, kd, 8, ctx_dasar)
    assert not _kesempatan_terakhir("CIE61404", 7, kd, 9, ctx_dasar)      # S9 masih menawarkan Tema B
    assert not _kesempatan_terakhir("CIE61404", 8, kd, 8, ctx_dasar)      # di/sesudah target: tidak berlaku


@pytest.mark.parametrize("mode", ["arm", "hybrid"])
def test_regresi_tema_ganjil_tidak_membuat_molor(ctx, mode):
    """Regresi: IP 2,2 dari S1 (pita 16–18) — dulu Tema B (hanya ganjil) tergeser ke S9 karena slot S7
    diisi Tema C yang masih bisa diambil di S8. Kini rencana selesai paling lambat S8."""
    plan = susun_rencana(profil(ctx.kurikulum, "A", 1, 2.2), mode, ctx)
    akhir = max(s.semester for s in plan.semester if s.items)
    assert akhir <= 8 and plan.validasi.lolos, (akhir, plan.validasi.pelanggaran)
    assert any("Kesempatan terakhir" in a for s in plan.semester for i in s.items for a in i.alasan)


def test_ip_tinggi_lulus_tujuh_semester(ctx):
    """IP ≥ 3,00 (pita 22–24): Skripsi & PKM dapat ditambahkan di S7 (09) → lulus 7 semester bila
    syarat terpenuhi — tidak kurang dari masa studi minimal (AD-09)."""
    plan = susun_rencana(profil(ctx.kurikulum, "A", 1, 3.5), "arm", ctx)
    akhir = max(s.semester for s in plan.semester if s.items)
    assert akhir == 7 and plan.validasi.lolos, plan.validasi.pelanggaran
