"""Tes aturan domain kurikulum (AD-01..AD-16) —
fungsi aturan murni + validate_plan (UC-13). Nilai yang diharapkan dihitung manual di docstring."""
from __future__ import annotations

from datetime import datetime

import pytest

from src.config import load_config
from src.data_loader import load_kurikulum
from src.rules import (DEFINISI_ATURAN, Keadaan, aturan_agama_akhir, aturan_beban_sks, aturan_kelengkapan,
                       aturan_konversi, aturan_korequisit, aturan_masa_studi, aturan_nilai, aturan_paket,
                       aturan_total_sks, batas_sks, boleh_diambil, cek_mk, hitung_ipk, mk_paket_semester, mk_tertunda,
                       semester_paket, syarat_konversi, tema_terpenuhi, validate_plan)
from src.schemas import ItemRencana, Plan, Profile, SemesterRencana

CFG = load_config(use_override_file=False, use_env=False)
KUR = load_kurikulum(CFG)
KEPENDIDIKAN = {"CIE61101", "CIE61102", "CIE61103", "CIE62104", "CIE62105", "CIE62106", "CIE62107",
                "CIE61108", "CIE61109", "CIE61110"}
PEDOMAN = {"CIE61111", "CIE61112", "UBU60002", "COM60061", "UBU60001", "UBU60005"}


def cfg_dengan(**aturan):
    return load_config({"aturan": aturan}, use_override_file=False, use_env=False)


def kd(lulus=(), gagal=(), jalur="A", agama="Islam", ipk=None):
    lulus = frozenset(lulus)
    return Keadaan(jalur, KUR.mk_agama(agama), lulus, lulus | frozenset(gagal), ipk)


def status(hasil, kode):
    return {h.status for h in hasil if h.kode == kode}


# ---- aturan per MK -----------------------------------------------------------------------


def test_prasyarat_harus_lulus_dan_tafsiran_literal():
    """AD-03: CIE62209 ← CIE62204 harus LULUS. Tafsiran literal "≥ E" (pernah menempuh,
    flag PRASYARAT_E_BERARTI_PERNAH_MENEMPUH) tetap tersedia di config."""
    gagal_saja = kd(gagal={"CIE62204"})
    assert status(cek_mk("CIE62209", 4, gagal_saja, KUR, CFG), "AD-03") == {"gagal"}
    assert status(cek_mk("CIE62209", 4, kd(lulus={"CIE62204"}), KUR, CFG), "AD-03") == {"lolos"}
    literal = cfg_dengan(PRASYARAT_E_BERARTI_PERNAH_MENEMPUH=True)
    assert status(cek_mk("CIE62209", 4, gagal_saja, KUR, literal), "AD-03") == {"lolos"}
    assert status(cek_mk("CIE62209", 4, kd(), KUR, CFG), "AD-03") == {"gagal"}
    h = next(h for h in cek_mk("CIE62209", 4, kd(), KUR, CFG) if h.kode == "AD-03")
    assert "05_prasyarat.csv#PR001" in h.sumber


@pytest.mark.parametrize("kode,s,harap", [
    ("CIE61208", 3, "lolos"), ("CIE61208", 4, "gagal"), ("CIE61208", 5, "lolos"),   # wajib: paritas Jalur A
    ("UBU60003", 1, "lolos"), ("UBU60003", 2, "gagal"),                              # INK06: sem. 1 (ganjil)
    ("UBU60001", 7, "lolos"), ("UBU60001", 5, "gagal"), ("UBU60001", 8, "lolos"),    # 09 A/7: tambah UBU60001
    ("COM60061", 6, "lolos"), ("COM60061", 4, "gagal"),                              # 09 A/6: tambah COM60061
    ("CIE62118", 4, "lolos"), ("CIE62118", 6, "lolos"), ("CIE62118", 5, "gagal"),    # Tema F: Tabel 11 = 4 (INK08)
    ("CIE62118", 2, "gagal"),
    ("CIE61113", 5, "lolos"), ("CIE61113", 7, "lolos"), ("CIE61113", 6, "gagal"),    # Tema C ganjil (INK07)
    ("CSD60702", 6, "lolos"), ("CSD60702", 5, "gagal"),                              # INK05: bukan paritas Deskripsi
    ("UBU60029", 5, "gagal"),                                                        # penawaran tidak tercantum
])
def test_penawaran_mode_paritas(kode, s, harap):
    assert status(cek_mk(kode, s, kd(), KUR, CFG), "AD-04") == {harap}


def test_penawaran_tabel11_ketat_dan_tanpa_saran_09():
    ketat = cfg_dengan(MODE_PENAWARAN_PILIHAN="tabel11_ketat")
    assert status(cek_mk("CIE62118", 6, kd(), KUR, ketat), "AD-04") == {"gagal"}
    assert status(cek_mk("CIE62118", 4, kd(), KUR, ketat), "AD-04") == {"lolos"}
    assert status(cek_mk("CIE61113", 7, kd(), KUR, ketat), "AD-04") == {"gagal"}
    tanpa09 = cfg_dengan(PENAWARAN_DARI_REKOMENDASI_09=False)
    assert status(cek_mk("UBU60001", 7, kd(), KUR, tanpa09), "AD-04") == {"gagal"}
    h = next(h for h in cek_mk("UBU60001", 7, kd(), KUR, CFG) if h.kode == "AD-04")
    assert "09_rekomendasi_beban_sks.csv" in h.sumber


def test_ubu60030_tidak_pernah_boleh():
    for s in range(1, 15):
        hasil = cek_mk("UBU60030", s, kd(), KUR, CFG)
        assert status(hasil, "AD-15") == {"gagal"} and not boleh_diambil(hasil)


def test_sudah_lulus_tidak_direkomendasikan_gagal_boleh_diulang():
    assert status(cek_mk("CIE61208", 5, kd(lulus={"CIE61208"}), KUR, CFG), "AD-07") == {"gagal"}
    assert boleh_diambil(cek_mk("CIE61208", 5, kd(gagal={"CIE61208"}), KUR, CFG))


def test_plp_butuh_lulus_seluruh_mk_kependidikan():
    """§4.2: 10 MK Kependidikan harus LULUS (pernah menempuh saja tidak cukup)."""
    kurang_satu = kd(lulus=KEPENDIDIKAN - {"CIE61110"}, gagal={"CIE61110"})
    assert status(cek_mk("CIE61111", 7, kurang_satu, KUR, CFG), "AD-11") == {"gagal"}
    lengkap = kd(lulus=KEPENDIDIKAN)
    hasil = cek_mk("CIE61111", 7, lengkap, KUR, CFG)
    assert status(hasil, "AD-11") == {"lolos"}
    # pedoman eksternal: perlu verifikasi manual, bukan lolos dan bukan pelanggaran
    assert status(hasil, "AD-16") == {"perlu_verifikasi_manual"} and boleh_diambil(hasil)


def test_pedoman_eksternal_tidak_dianggap_terpenuhi():
    for k in PEDOMAN:
        h = [x for x in cek_mk(k, KUR.mk[k].semester_jalur_a, kd(lulus=KEPENDIDIKAN), KUR, CFG) if x.kode == "AD-16"]
        assert len(h) == 1 and h[0].status == "perlu_verifikasi_manual" and not h[0].lolos
        assert "05_prasyarat.csv#PR0" in h[0].sumber


def test_korequisit_plp():
    assert status(aturan_korequisit(7, ["CIE61111"], kd(), KUR, CFG), "AD-11") == {"gagal"}
    assert aturan_korequisit(7, ["CIE61111", "CIE61112"], kd(), KUR, CFG) == []
    assert aturan_korequisit(7, ["CIE61111"], kd(lulus={"CIE61112"}), KUR, CFG) == []


def test_agama_sesuai_input_dan_cukup_satu():
    assert status(cek_mk("MPK60001", 2, kd(agama=None), KUR, CFG), "AD-12") == {"gagal"}
    assert status(cek_mk("MPK60001", 2, kd(agama="Islam"), KUR, CFG), "AD-12") == {"lolos"}
    assert status(cek_mk("MPK60002", 2, kd(agama="Islam"), KUR, CFG), "AD-12") == {"gagal"}
    assert status(cek_mk("MPK60001", 4, kd(lulus={"MPK60003"}, agama="Islam"), KUR, CFG), "AD-12") == {"gagal"}


def _lulus_sks(target: int) -> set[str]:
    """Himpunan MK (tanpa Skripsi/agama/UBU60030) dengan total SKS tepat ``target`` (subset-sum, deterministik)."""
    kode = sorted(set(KUR.kode) - KUR.grup("AGAMA") - {"UBU60001", "UBU60030"})
    dapat: dict[int, tuple[str, ...]] = {0: ()}
    for k in kode:
        for total, isi in sorted(dapat.items(), reverse=True):
            baru = total + KUR.sks(k)
            if baru <= target and baru not in dapat:
                dapat[baru] = isi + (k,)
    assert target in dapat, f"tidak dapat menyusun {target} SKS"
    return set(dapat[target])


def test_skripsi_ad10_sks_dan_ipk():
    """AD-10 (Pedoman UB 8.5): Skripsi (UBU60001, 6 SKS) bila lulus ≥ 120 SKS dan IPK ≥ 2,00.
    119 SKS → gagal; 120 SKS + IPK 2,00 → lolos; IPK 1,99 → gagal; IPK tidak diketahui → perlu verifikasi.
    SKS konversi merdeka belajar ikut dihitung (119 + 1 = 120)."""
    def hasil(lulus, ipk, konv=0):
        k = Keadaan("A", KUR.mk_agama("Islam"), frozenset(lulus), frozenset(lulus), ipk, konv)
        return status(cek_mk("UBU60001", 8, k, KUR, CFG), "AD-10")
    assert hasil(_lulus_sks(119), 3.5) == {"gagal"}
    assert hasil(_lulus_sks(120), 2.0) == {"lolos"}
    assert hasil(_lulus_sks(120), 1.99) == {"gagal"}
    assert hasil(_lulus_sks(120), None) == {"perlu_verifikasi_manual"}
    assert hasil(_lulus_sks(119), 3.0, konv=1) == {"lolos"}
    assert status(cek_mk("COM60051", 5, kd(), KUR, CFG), "AD-10") == set()   # hanya berlaku untuk Skripsi
    ringan = cfg_dengan(SKRIPSI={"kode": "UBU60001", "min_sks_lulus": 100, "min_ipk": 2.0})
    k = Keadaan("A", None, frozenset(_lulus_sks(110)), frozenset(_lulus_sks(110)), 2.5)
    assert status(cek_mk("UBU60001", 8, k, KUR, ringan), "AD-10") == {"lolos"}


def test_hitung_ipk_dari_riwayat_nilai():
    """IPK = Σ(bobot × SKS)/Σ SKS: COM60014 A (5 SKS → 20), COM60015 C (2 → 4), CIE61301 B+ (3 → 10,5),
    CIE61208 E (3 → 0) → 34,5 / 13 = 2,6538 → 2,65."""
    ipk, ket = hitung_ipk({"COM60014": "A", "COM60015": "c", "CIE61301": "B+", "CIE61208": "E"}, KUR, CFG)
    assert ipk == 2.65 and "13 SKS" in ket
    assert hitung_ipk({}, KUR, CFG)[0] is None
    assert hitung_ipk({"COM60014": "T"}, KUR, CFG)[0] is None
    assert hitung_ipk({"PTI101": "A"}, KUR, CFG)[0] is None
    assert hitung_ipk({"COM60014": "A"}, KUR, None)[0] is None
    profil = Profile(semester_saat_ini=3, ip_terakhir=3.0, riwayat_nilai={"COM60014": "B", "CIE61301": "A"},
                     mk_lulus={"COM60014", "CIE61301"})
    assert Keadaan.dari_profil(profil, KUR, CFG).ipk == round((15 + 12) / 8, 2)       # 3,375 → 3,38
    assert Keadaan.dari_profil(profil, KUR).ipk is None                                # tanpa cfg tidak dihitung


# ---- aturan per semester -----------------------------------------------------------------


def test_batas_sks_paket_pita_dan_fallback():
    assert semester_paket(KUR) == {1: 20, 2: 20}
    assert batas_sks(1, "A", 0.0, KUR)[:2] == (0, 20)
    assert batas_sks(3, "A", 3.0, KUR)[:2] == (22, 24)
    assert batas_sks(6, "A", 2.2, KUR)[1] == 18
    lo, hi, sumber = batas_sks(9, "A", 3.0, KUR)
    assert (lo, hi) == (22, 24) and "AT05" in sumber
    assert batas_sks(6, "B", 2.2, KUR)[1] == 18  # semester konversi tanpa konversi → pita seragam


def test_beban_sks_dan_konversi():
    assert status([aturan_beban_sks(3, ["CIE61101"] * 1, 0, kd(), 3.0, KUR, CFG)], "AD-02") == {"lolos"}
    paket3 = [s.kode_mk for s in KUR.slot("A", 3)]  # 21 SKS
    assert aturan_beban_sks(3, paket3, 0, kd(), 2.2, KUR, CFG).status == "gagal"      # batas 18
    assert aturan_beban_sks(3, paket3, 0, kd(), 2.5, KUR, CFG).status == "lolos"      # batas 21
    b = kd(jalur="B")
    assert aturan_beban_sks(6, [], 20, b, 2.2, KUR, CFG).status == "lolos"            # AT13, bukan pita IP
    assert aturan_beban_sks(6, [], 21, b, 3.5, KUR, CFG).status == "gagal"


def test_mk_paket_dan_aturan_paket():
    assert "MPK60001" in mk_paket_semester(2, kd(agama="Islam"), KUR)
    assert not set(mk_paket_semester(2, kd(agama=None), KUR)) & KUR.grup("AGAMA")
    assert not set(mk_paket_semester(2, kd(lulus={"MPK60004"}, agama="Islam"), KUR)) & KUR.grup("AGAMA")
    paket1 = [s.kode_mk for s in KUR.slot("A", 1)]
    assert status(aturan_paket(1, paket1, kd(), KUR, CFG), "AD-08") == {"lolos"}
    assert status(aturan_paket(1, paket1[1:], kd(), KUR, CFG), "AD-08") == {"gagal"}
    assert status(aturan_paket(1, paket1 + ["CIE61208"], kd(), KUR, CFG), "AD-08") == {"gagal"}
    assert aturan_paket(3, [], kd(), KUR, CFG) == []


def test_syarat_dan_aturan_konversi():
    kecuali = {"CIE61111", "CIE61112", "UBU60002", "UBU60005", "COM60061", "UBU60001"}
    cukup = kd(lulus=(KUR.wajib - KUR.grup("AGAMA") - kecuali) | {"MPK60001"}, jalur="B")
    assert syarat_konversi(cukup, KUR) == (True, [])
    kurang = kd(lulus=(KUR.wajib - KUR.grup("AGAMA") - kecuali - {"CSD60708"}), jalur="B")
    assert syarat_konversi(kurang, KUR) == (False, ["CSD60708", "MK agama"])
    assert status(aturan_konversi(6, 20, cukup, KUR, CFG), "AD-14") == {"perlu_verifikasi_manual"}
    assert status(aturan_konversi(6, 20, kurang, KUR, CFG), "AD-14") == {"gagal"}
    assert status(aturan_konversi(6, 20, kd(lulus=cukup.lulus, jalur="A"), KUR, CFG), "AD-14") == {"gagal"}
    assert aturan_konversi(6, 0, cukup, KUR, CFG) == []


# ---- aturan akhir rencana ----------------------------------------------------------------


def test_tema_dan_kelengkapan():
    assert tema_terpenuhi({"CIE61402"}, set(), KUR) == {"A"}
    assert tema_terpenuhi({"CIE61402"}, {"C", "D"}, KUR) == {"A", "C", "D"}
    """AD-05: wajib + pilihan ≥ 27 SKS (9 × 3); AD-13: sembilan tema. Tanpa Tema A → 24 SKS pilihan:
    keduanya gagal. Konversi 3 SKS menutup kekurangan SKS (AD-05) dan tema (AD-13) bila temanya diakui."""
    wajib = KUR.wajib - KUR.grup("AGAMA")
    satu_per_tema = {sorted(KUR.mk_tema(t))[0] for t in KUR.tema}
    hasil = aturan_kelengkapan(wajib | satu_per_tema, set(), 0, KUR)
    assert status(hasil, "AD-05") == {"lolos"} and status(hasil, "AD-13") == {"lolos"}
    tanpa_a = wajib | satu_per_tema - {"CIE61402", "CIE61403"}
    hasil = aturan_kelengkapan(tanpa_a, set(), 0, KUR)
    assert status(hasil, "AD-05") == {"gagal"} and status(hasil, "AD-13") == {"gagal"}
    assert any("24 < 27" in h.alasan for h in hasil)
    hasil = aturan_kelengkapan(tanpa_a, {"A"}, 3, KUR)
    assert status(hasil, "AD-05") == {"lolos"} and status(hasil, "AD-13") == {"lolos"}
    hasil = aturan_kelengkapan(wajib - {"CIE61208"} | satu_per_tema, set(), 0, KUR)
    assert status(hasil, "AD-05") == {"gagal"} and status(hasil, "AD-13") == {"lolos"}


def test_total_sks_agama_masa_studi():
    wajib = KUR.wajib - KUR.grup("AGAMA") | {"MPK60001"}
    satu_per_tema = {sorted(KUR.mk_tema(t))[0] for t in KUR.tema}
    assert KUR.total_sks(wajib | satu_per_tema) == 145
    assert status(aturan_total_sks(wajib | satu_per_tema, 0, KUR, 8), "AD-01") == {"lolos"}
    assert status(aturan_total_sks(wajib, 0, KUR, 8), "AD-01") == {"gagal"}
    assert status(aturan_total_sks(wajib | satu_per_tema, 15, KUR, 8), "AD-06") == {"lolos"}   # 160 ≤ 160
    lebih = aturan_total_sks(wajib | satu_per_tema, 16, KUR, 8)                                # 161 > 160
    assert status(lebih, "AD-06") == {"gagal"} and status(lebih, "AD-01") == {"lolos"}
    assert aturan_agama_akhir({"MPK60001", "MPK60002"}, kd(), KUR).status == "gagal"
    assert aturan_agama_akhir(set(), kd(agama=None), KUR).status == "perlu_verifikasi_manual"
    assert aturan_agama_akhir({"MPK60001"}, kd(), KUR).status == "lolos"
    assert status(aturan_masa_studi([13, 14], 13, KUR), "AD-09") == {"lolos"}
    assert status(aturan_masa_studi([14, 15], 14, KUR), "AD-09") == {"gagal"}
    assert status(aturan_masa_studi([5, 6], 5, KUR), "AD-09") == {"gagal"}     # selesai < 7 semester (AT04)
    assert status(aturan_masa_studi([5, 6, 7], 5, KUR), "AD-09") == {"lolos"}


def test_nilai_d_maksimal_10_persen_dan_e_pilihan():
    """AD-07 (Pedoman UB 8.4), total beban kredit 145 SKS (batas 10% = 14,5 SKS):
    - D: COM60014 (5), CIE61101 (3); D+: CIE61102 (4) → 12 SKS = 8,3% → tidak ada catatan;
    - + CIE61103 D (3) → 15 SKS = 10,3% > 10% → perlu verifikasi; saran ulang dimulai dari nilai terendah
      dan SKS terbesar: COM60014 (D, 5 SKS) → sisa 10 SKS = 6,9% ≤ 10% → cukup satu MK.
    - MK pilihan bernilai E yang tidak direncanakan ulang → perlu verifikasi (status E di transkrip)."""
    wajib = KUR.wajib - KUR.grup("AGAMA") | {"MPK60001"}
    akhir = wajib | {sorted(KUR.mk_tema(t))[0] for t in KUR.tema}
    assert KUR.total_sks(akhir) == 145

    def profil(riwayat):
        return Profile(semester_saat_ini=5, ip_terakhir=3.0, riwayat_nilai=riwayat)
    aman = {"COM60014": "D", "CIE61101": "D", "CIE61102": "D+"}
    assert aturan_nilai(profil(aman), akhir, 0, KUR, CFG) == []
    lewat = aturan_nilai(profil(aman | {"CIE61103": "D"}), akhir, 0, KUR, CFG)
    assert len(lewat) == 1 and lewat[0].kode == "AD-07" and lewat[0].status == "perlu_verifikasi_manual"
    assert "15 dari 145" in lewat[0].alasan and "Pemrograman Dasar" in lewat[0].alasan
    assert "Perkembangan Peserta Didik" not in lewat[0].alasan
    e_pilihan = aturan_nilai(profil({"CIE61403": "E"}), akhir, 0, KUR, CFG)
    assert [h.status for h in e_pilihan] == ["perlu_verifikasi_manual"] and "CIE61403" in e_pilihan[0].alasan
    assert aturan_nilai(profil({"CIE61402": "E"}), akhir, 0, KUR, CFG) == []   # diulang/direncanakan → tidak dicatat


# ---- validator end-to-end: struktur resmi harus lolos seluruh aturan ------------------------

PILIHAN_A = {5: {"A": "CIE61402", "B": "CIE61404"},
             6: {"C": "CIE62305", "D": "CSD60702", "E": "CIE62116", "F": "CIE62118", "G": "CSD60716",
                 "H": "CSD60711", "I": "CIE61119"}}


def _item(kode):
    m = KUR.mk[kode]
    return ItemRencana(kode=kode, nama=m.nama, sks=m.sks, sifat=m.sifat)


def _konversi(sks, tema):
    return ItemRencana(kode="KONVERSI-MB", nama="Konversi merdeka belajar", sks=sks, sifat="pilihan",
                       jenis="konversi", tema_konversi=sorted(tema))


def rencana_struktur(jalur: str, agama: str = "Islam", pilihan=None, konversi_tema=None, sisa=None):
    """Susun rencana langsung dari 08 (slot tema diisi ``pilihan``) — tanpa planner."""
    pilihan = pilihan or PILIHAN_A
    sems = []
    for s in range(1, 9):
        items = []
        for slot in KUR.slot(jalur, s):
            if slot.jenis_slot == "MK_WAJIB":
                items.append(_item(slot.kode_mk))
            elif slot.jenis_slot == "MK_WAJIB_PILIH_SATU":
                items.append(_item(KUR.mk_agama(agama)))
            elif slot.jenis_slot == "PILIHAN_TEMA":
                items.append(_item(pilihan[s][slot.tema]))
            elif slot.jenis_slot == "PILIHAN_KONVERSI":
                items.append(_konversi(slot.sks, konversi_tema))
            elif slot.jenis_slot == "PILIHAN_SISA_TEMA":
                items.append(_item(sisa))
        sems.append(SemesterRencana(semester=s, items=items, total_sks=sum(i.sks for i in items)))
    return sems


def _plan(sems, profil, mode="arm", theta=None):
    return Plan(id="uji", mode=mode, dibuat=datetime.now(), profil=profil, semester=sems, theta=theta)


def test_struktur_jalur_a_lolos_seluruh_aturan():
    profil = Profile(semester_saat_ini=1, ip_terakhir=3.2, agama="Islam")
    v = validate_plan(_plan(rencana_struktur("A"), profil), profil, KUR, CFG)
    assert v.lolos, v.pelanggaran
    assert set(v.per_aturan) == set(DEFINISI_ATURAN) and all(v.per_aturan.values())
    # enam MK berpedoman eksternal selalu ditandai perlu verifikasi
    assert {h.kode_mk for h in v.rincian if h.kode == "AD-16"} == PEDOMAN
    # tanpa riwayat nilai, IPK tidak dapat dihitung → syarat IPK Skripsi perlu verifikasi (SKS lulus 135 ≥ 120)
    assert [h.status for h in v.rincian if h.kode == "AD-10"] == ["perlu_verifikasi_manual"]


def test_struktur_jalur_a_mode_tabel11_ketat_melanggar_tema_f():
    """INK08: Tabel 11 menawarkan Tema F hanya di semester 4, padahal slotnya di semester 6."""
    profil = Profile(semester_saat_ini=1, ip_terakhir=3.2, agama="Islam")
    ketat = cfg_dengan(MODE_PENAWARAN_PILIHAN="tabel11_ketat")
    v = validate_plan(_plan(rencana_struktur("A"), profil), profil, KUR, ketat)
    assert not v.per_aturan["AD-04"]
    assert {h.kode_mk for h in v.rincian if h.kode == "AD-04"} == {"CIE62118"}


def test_struktur_jalur_b_dengan_konversi_lolos():
    """Jalur B: konversi 20 SKS (6 tema) + sisa tema H di semester 7 → total 147 SKS."""
    profil = Profile(semester_saat_ini=1, ip_terakhir=3.2, agama="Islam", jalur="B")
    sems = rencana_struktur("B", konversi_tema={"C", "D", "E", "F", "G", "I"}, sisa="CIE61215")
    assert sum(s.total_sks for s in sems) == 147
    v = validate_plan(_plan(sems, profil), profil, KUR, CFG)
    assert v.lolos, v.pelanggaran
    assert any(h.kode == "AD-14" and h.status == "perlu_verifikasi_manual" for h in v.rincian)


def test_validator_mendeteksi_pelanggaran():
    profil = Profile(semester_saat_ini=1, ip_terakhir=3.2, agama="Islam")
    sems = rencana_struktur("A")
    s5 = sems[4]
    s5.items.append(_item("UBU60030"))                          # AD-15 (+ AD-04: penawaran tidak tercantum)
    s5.items.append(_item("UBU60001"))                          # skripsi di S5 → AD-04 + AD-10 (SKS lulus < 120)
    sems[6].items = [i for i in sems[6].items if i.kode != "CIE61112"]   # PLP 1 tanpa PLP 2 → AD-11
    sems[7].items.append(_item("CIE61208"))                     # duplikat (sudah di S3) → AD-07
    sems[5].items = [i for i in sems[5].items if i.kode != "CIE61119"]   # tema I hilang → AD-13
    v = validate_plan(_plan(sems, profil), profil, KUR, CFG)
    gagal = {k for k, ok in v.per_aturan.items() if not ok}
    assert {"AD-15", "AD-04", "AD-10", "AD-11", "AD-07", "AD-13", "AD-02"} <= gagal, v.pelanggaran
    assert not v.lolos
    assert all("[" in p and "]" in p for p in v.pelanggaran)  # setiap pelanggaran menyertakan sumber


def test_validator_prioritas_wajib_tertunda():
    """AT09: CIE61208 gagal di S3 → wajib diambil di S5 bila kapasitas cukup."""
    s12 = {s.kode_mk for s in KUR.slot("A", 1) + KUR.slot("A", 2) if s.jenis_slot == "MK_WAJIB"} | {"MPK60001"}
    s3 = {s.kode_mk for s in KUR.slot("A", 3)} - {"CIE61208"}
    s4 = {s.kode_mk for s in KUR.slot("A", 4)}
    profil = Profile(semester_saat_ini=5, ip_terakhir=3.2, agama="Islam", mk_lulus=s12 | s3 | s4,
                     mk_pernah_gagal={"CIE61208"})
    assert mk_tertunda(profil, Keadaan.dari_profil(profil, KUR, CFG), KUR) == {"CIE61208"}
    semua = rencana_struktur("A")[4:]
    tanpa = validate_plan(_plan(semua, profil), profil, KUR, CFG)
    assert not tanpa.per_aturan["AD-07"]
    semua[0].items.append(_item("CIE61208"))                   # 21 + 3 = 24 SKS ≤ 24
    dengan = validate_plan(_plan(semua, profil), profil, KUR, CFG)
    assert dengan.per_aturan["AD-07"] and dengan.lolos, dengan.pelanggaran


def test_validator_theta_hybrid_membenarkan_penundaan():
    """Penundaan wajib tertunda tidak dianggap pelanggaran bila θ rencana hybrid tidak memberi ruang."""
    s12 = {s.kode_mk for s in KUR.slot("A", 1) + KUR.slot("A", 2) if s.jenis_slot == "MK_WAJIB"} | {"MPK60001"}
    lulus = s12 | ({s.kode_mk for s in KUR.slot("A", 3)} - {"CIE61208"}) | {s.kode_mk for s in KUR.slot("A", 4)}
    profil = Profile(semester_saat_ini=5, ip_terakhir=3.2, agama="Islam", mk_lulus=lulus,
                     mk_pernah_gagal={"CIE61208"})
    semua = rencana_struktur("A")[4:]

    def beban(kode):                            # CL 5 untuk semua MK → B_s = 5 × SKS
        return 5.0 * KUR.total_sks(kode)

    # S5 Jalur A = 21 SKS (105) + CIE61208 3 SKS (15) = 120
    sempit = validate_plan(_plan(semua, profil, "hybrid", theta=110.0), profil, KUR, CFG, beban)
    assert sempit.per_aturan["AD-07"]          # 120 > 110 → penundaan terpaksa
    lapang = validate_plan(_plan(semua, profil, "hybrid", theta=200.0), profil, KUR, CFG, beban)
    assert not lapang.per_aturan["AD-07"]
    arm = validate_plan(_plan(semua, profil, "arm", theta=110.0), profil, KUR, CFG, beban)
    assert not arm.per_aturan["AD-07"]         # θ hanya berlaku bagi rencana hybrid
