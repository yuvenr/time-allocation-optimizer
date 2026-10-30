"""Tes lapisan LLM (UC-09; CLAUDE.md §5, §7): parsing/validasi/retry JSON, prompt Bloom (satu MK,
tanpa info terlarang, few-shot fiktif teracak + chain-of-thought), mock = heuristik, resume, konsensus,
CL per MK, impor pakar, klien Anthropic (thinking adaptif) & OpenAI (structured output, reasoning effort), parameter
yang ditolak model, dan eksekusi paralel."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from src import db
from src.config import load_config
from src.data_loader import load_kurikulum
from src.llm import beban
from src.llm.beban import (CONTOH_FEWSHOT, PetaBatch, bangun_prompt_batch, baris_rating, cl_mk, impor_pakar,
                           jalankan_penilaian_bloom, justifikasi_cpmk, konsensus_bloom, konsistensi_pengulangan,
                           muat_beban_mk, muat_rating, nilai_batch, peta_heuristik, pisah_per_mk, rng_batch,
                           skema_penilaian)
from src.llm.client import (AnthropicClient, LLMClient, LLMError, LLMKonfigurasiError, LLMResponse, MockClient, buat_client,
                            parse_json_keluaran)
from src.llm.konteks import SumberKonteks
from src.llm.prompts import muat_prompt
from src.schemas import PenilaianBloom
from tests.conftest import buat_cfg

CFG = load_config(use_override_file=False, use_env=False)
KUR = load_kurikulum(CFG)
SK = SumberKonteks.muat(CFG, KUR)
TEMPLAT = muat_prompt("bloom", CFG.llm.versi_prompt_bloom)


def cfg_tmp(tmp_path, **extra):
    return buat_cfg(tmp_path, **extra)


class KlienUrutan(LLMClient):
    """Klien palsu: mengembalikan teks sesuai urutan, mencatat setiap pesan."""

    provider = "palsu"

    def __init__(self, jawaban: list[str]):
        super().__init__("palsu-1", CFG)
        self.jawaban, self.pesan = list(jawaban), []

    def complete(self, system, user, skema=None):
        self.pesan.append(user)
        return LLMResponse(self.jawaban.pop(0) if len(self.jawaban) > 1 else self.jawaban[0], self.model)


# ---- parsing & retry generik ---------------------------------------------------------------

def test_parse_blok_kode_setelah_penalaran():
    assert parse_json_keluaran('Penalaran: menganalisis.\n```json\n{"a": 1}\n```')["a"] == 1


def test_parse_json_polos_ambil_yang_terakhir():
    obj = parse_json_keluaran('Contoh {"a": 1}. Jawaban: {"a": 2, "b": "x {y}"}')
    assert obj == {"a": 2, "b": "x {y}"}


@pytest.mark.parametrize("teks", ["", "tidak ada json", "```json\n{rusak\n```"])
def test_parse_gagal(teks):
    with pytest.raises(ValueError):
        parse_json_keluaran(teks)


def test_mock_tanpa_penjawab_galat():
    with pytest.raises(LLMError):
        MockClient("mock-bloom-v1", CFG).complete("s", "u")


# ---- kontrak JSON penilaian Bloom & pemisahan batch per MK --------------------------------

def _pb(i, bloom="C3", ranah="kognitif", **kw):
    return {"id_cpmk": i, "bloom": bloom, "ranah": ranah, "alasan": "a", **kw}


def test_pisah_per_mk_hanya_mk_lengkap_diterima():
    peta = PetaBatch(cpmk={"MK01-CPMK01": ("A1", "A1-C1"), "MK01-CPMK02": ("A1", "A1-C2"),
                           "MK02-CPMK01": ("B1", "B1-C1"), "MK03-CPMK01": ("C1", "C1-C1")},
                     mk={"MK01": "A1", "MK02": "B1", "MK03": "C1"})
    keluaran = {"penilaian": [_pb("MK01-CPMK01", "c4"), _pb("MK01-CPMK02", "A", "afektif", keyakinan=0.7),
                              _pb("MK02-CPMK01"), _pb("MK02-CPMK01", "C2"),          # ganda → MK02 ditolak
                              _pb("MK03-CPMK01", "C9"),                              # tidak sah → MK03 ditolak
                              _pb("MK99-CPMK01")]}                                   # asing → diabaikan
    hasil, catatan = pisah_per_mk(keluaran, peta)
    assert list(hasil) == ["A1"] and [p.bloom for p in hasil["A1"]] == ["C4", "A"]
    assert any("asing" in c for c in catatan) and any("MK03-CPMK01" in c for c in catatan)
    with pytest.raises(ValueError):
        pisah_per_mk({"penilaian": "C3"}, peta)


@pytest.mark.parametrize("obj", [_pb("x", "A", "kognitif"), _pb("x", "C3", "afektif"), _pb("x", "C7"),
                                 _pb("x", keyakinan=1.5), {"id_cpmk": "x", "bloom": "C2", "ranah": "kognitif",
                                                           "alasan": ""}])
def test_penilaian_bloom_menolak_nilai_tidak_sah(obj):
    with pytest.raises(ValidationError):
        PenilaianBloom.model_validate(obj)


# ---- prompt batch -------------------------------------------------------------------------

def test_prompt_batch_semua_mk_tanpa_info_terlarang():
    mks = [SK.mk(k) for k in SK.kode_dengan_cpmk()]
    system, user, peta = bangun_prompt_batch(mks, rng_batch(42, 1), TEMPLAT)
    teks = system + user
    assert sorted(peta.mk.values()) == sorted(m.kode for m in mks) and len(peta.cpmk) == sum(len(m.cpmk) for m in mks)
    assert user.count("### MATA KULIAH YANG DINILAI") == 1 and len(re.findall(r"^### MK\d{2}$", user, re.M)) == len(mks)
    assert len(re.findall(r"^### CONTOH\d$", user, re.M)) == len(CONTOH_FEWSHOT)
    assert f"({len(mks)} mata kuliah, {len(peta.cpmk)} CPMK)" in user
    assert not any(c in teks for c in KUR.kode), "kode MK dapat membocorkan semester"
    for lb, (kode, id_cpmk) in peta.cpmk.items():
        assert lb.startswith(next(m for m, k in peta.mk.items() if k == kode))
        assert f"[{lb}] {dict(SK.mk(kode).cpmk)[id_cpmk]}" in user
    statis = (system + user.split("### MATA KULIAH YANG DINILAI")[0]).lower()
    for kata in ("sks", "semester", "pakar", "heuristik"):
        assert kata not in statis, kata
    assert "beberapa mata kuliah" in system.lower() and TEMPLAT.id == "bloom-v3"


def test_fewshot_fiktif_dan_urutan_teracak_per_pengulangan():
    """Tiga MK fiktif rendah/sedang/tinggi dengan penalaran; urutan contoh DAN urutan MK diacak
    deterministik per pengulangan (seed) — bias posisi tidak sama di setiap batch."""
    semua_cpmk = set(SK.cpmk["teks"])
    nama_mk = {m.nama for m in KUR.mk.values()}
    assert [c["tingkat"] for c in CONTOH_FEWSHOT] == ["rendah", "sedang", "tinggi"]
    level = {}
    for c in CONTOH_FEWSHOT:
        assert c["nama"] not in nama_mk
        for teks, jawaban in c["cpmk"]:
            assert teks not in semua_cpmk
            p = PenilaianBloom.model_validate({"id_cpmk": "x", **jawaban})
            assert p.penalaran and len(p.penalaran) > 20
            level.setdefault(c["tingkat"], set()).add(p.bloom)
    assert level == {"rendah": {"C1", "C2"}, "sedang": {"C3", "C4", "A"}, "tinggi": {"C5", "C6"}}
    mks = [SK.mk(k) for k in SK.kode_dengan_cpmk()[:8]]

    def urutan(run):
        _, user, peta = bangun_prompt_batch(mks, rng_batch(42, run), TEMPLAT)
        assert "rendah" not in user.split("### MATA KULIAH")[0].lower()
        contoh = tuple(sorted(range(3), key=lambda i: user.index(CONTOH_FEWSHOT[i]["nama"])))
        return contoh, tuple(peta.mk.values())

    assert urutan(1) == urutan(1)
    assert len({urutan(r)[0] for r in range(1, 12)}) > 1 and len({urutan(r)[1] for r in range(1, 12)}) > 1


# ---- eksekusi batch -----------------------------------------------------------------------

class _KlienPrompt(LLMClient):
    """Klien uji aman-paralel (provider 'openai' → sumber 'llm'): menjawab heuristik KKO untuk setiap label
    ``[MKxx-CPMKyy] teks`` di prompt. ``buang`` = kode MK yang dihilangkan pada permintaan pertama;
    ``potong_bila_lebih`` = keluaran terpotong (kosong) bila batch memuat lebih dari n MK."""

    provider = "openai"

    def __init__(self, model, cfg, buang=(), potong_bila_lebih=None):
        super().__init__(model, cfg)
        self.buang, self.potong, self.n_panggil, self.ukuran = set(buang), potong_bila_lebih, 0, []

    def complete(self, system, user, skema=None):
        self.n_panggil += 1
        baris = re.findall(r"^\[(MK\d+-CPMK\d+)\] (.*)$", user, re.M)
        n_mk = len({lb.split("-")[0] for lb, _ in baris})
        self.ukuran.append(n_mk)
        if self.potong and n_mk > self.potong:
            return LLMResponse('{"penilaian": [', self.model, terpotong=True, token_masuk=10, token_keluar=5)
        teks_ke_id = dict(zip(SK.cpmk["teks"], SK.cpmk["id_cpmk"]))
        kode_dari_id = dict(zip(SK.cpmk["id_cpmk"], SK.cpmk["kode_mk"]))
        heur = peta_heuristik(SK)
        penilaian = []
        for lb, teks in baris:
            idc = teks_ke_id[teks]
            if self.n_panggil == 1 and kode_dari_id[idc] in self.buang:
                continue
            bloom, ranah, _ = heur[idc]
            penilaian.append({"id_cpmk": lb, "penalaran": "-", "bloom": bloom, "ranah": ranah, "alasan": "-",
                              "keyakinan": None})
        return LLMResponse(json.dumps({"penilaian": penilaian}), self.model, token_masuk=10, token_keluar=5)


def test_mock_batch_sama_dengan_heuristik_dan_deterministik():
    heur = peta_heuristik(SK)
    klien = buat_client(CFG, "mock")
    mks = [SK.mk(k) for k in ("COM60014", "MPK60003", "UBU60001")]
    hb = nilai_batch(klien, mks, 1, CFG, TEMPLAT, heur, "r1-b1")
    assert not hb.gagal and len(hb.permintaan) == 1 and hb.permintaan[0]["n_mk"] == 3
    rows = baris_rating(klien, hb, mks, 1, TEMPLAT, "r1-b1")
    assert [r["id_cpmk"] for r in rows] == [i for m in mks for i in m.id_cpmk]
    assert [r["bloom"] for r in rows] == [heur[r["id_cpmk"]][0] for r in rows]
    assert all(r["status"] == "ok" and r["versi_prompt"] == "bloom-v3" and r["penalaran"] for r in rows)
    ulang = baris_rating(klien, nilai_batch(klien, mks, 1, CFG, TEMPLAT, heur, "r1-b1"), mks, 1, TEMPLAT, "r1-b1")
    assert [(r["id_cpmk"], r["bloom"]) for r in ulang] == [(r["id_cpmk"], r["bloom"]) for r in rows]


def test_mk_tidak_lengkap_dikirim_ulang_sendiri():
    mks = [SK.mk(k) for k in ("COM60014", "CIE61206", "COM60016")]
    klien = _KlienPrompt("uji", CFG, buang={"CIE61206"})
    hb = nilai_batch(klien, mks, 1, CFG, TEMPLAT, peta_heuristik(SK), "r1-b1")
    assert not hb.gagal and set(hb.penilaian) == {m.kode for m in mks}
    assert klien.ukuran == [3, 1]                                       # percobaan ke-2 hanya MK yang kurang
    assert [e["status"] for e in hb.permintaan] == ["sebagian", "ok"]
    assert [p.id_cpmk for p in hb.penilaian["CIE61206"]] == SK.mk("CIE61206").id_cpmk   # label lokal dipulihkan


def test_mk_tetap_gagal_ditandai_tinjau_manual():
    class Rusak(_KlienPrompt):
        def complete(self, system, user, skema=None):
            self.n_panggil += 1
            return LLMResponse("bukan json", self.model)

    mks = [SK.mk("COM60014")]
    klien = Rusak("uji", CFG)
    hb = nilai_batch(klien, mks, 1, CFG, TEMPLAT, {}, "r1-b1")
    assert hb.gagal == ["COM60014"] and klien.n_panggil == CFG.llm.max_retry + 1
    rows = baris_rating(klien, hb, mks, 1, TEMPLAT, "r1-b1")
    assert {r["status"] for r in rows} == {"perlu_tinjau_manual"} and {r["bloom"] for r in rows} == {""}


def test_keluaran_terpotong_batch_dibelah_dua():
    mks = [SK.mk(k) for k in SK.kode_dengan_cpmk()[:5]]
    klien = _KlienPrompt("uji", CFG, potong_bila_lebih=2)
    hb = nilai_batch(klien, mks, 1, CFG, TEMPLAT, peta_heuristik(SK), "r1-b1")
    assert not hb.gagal and len(hb.penilaian) == 5
    assert klien.ukuran == [5, 2, 3, 1, 2]                              # 5 → 2 + 3; 3 masih terpotong → 1 + 2
    assert {e["id_batch"] for e in hb.permintaan} >= {"r1-b1", "r1-b1.1", "r1-b1.2"}


def test_galat_konfigurasi_tidak_diretry():
    class Rusak(LLMClient):
        provider = "rusak"

        def complete(self, system, user, skema=None):
            raise LLMKonfigurasiError("kunci API tidak ada")

    with pytest.raises(LLMKonfigurasiError):
        nilai_batch(Rusak("x", CFG), [SK.mk("COM60014")], 1, CFG, TEMPLAT, {}, "r1-b1")


def test_penilaian_batch_tiga_permintaan_dan_resume(tmp_path, monkeypatch):
    """70 MK × 3 pengulangan = 3 permintaan (bukan 210); resume melewati (MK, pengulangan) yang sudah ada."""
    monkeypatch.setattr(beban, "buat_client", lambda cfg, *a, **k: _KlienPrompt("uji-batch", cfg))
    cfg = cfg_tmp(tmp_path, llm={"konkurensi": 1})
    r = jalankan_penilaian_bloom(cfg)
    assert (r["n_batch"], r["n_permintaan_api"], r["n_mk_dinilai"]) == (3, 3, 70 * 3)
    assert (r["token_masuk"], r["token_keluar"]) == (30, 15)
    batch = db.baca(cfg, "SELECT * FROM batch_llm")
    assert len(batch) == 3 and set(batch["n_mk"]) == {70} and set(batch["status"]) == {"ok"}
    df = beban.muat_rating(cfg, "llm")
    assert len(df) == len(SK.cpmk) * 3 and set(df["penilai"]) == {"1", "2", "3"}
    assert df["catatan"].str.startswith("batch r").all()
    lagi = jalankan_penilaian_bloom(cfg)
    assert lagi["n_permintaan_api"] == 0 and lagi["n_dilewati_resume"] == 70 * 3
    with db.koneksi(cfg) as c:                                            # hapus satu (MK, pengulangan)
        c.execute("DELETE FROM rating_bloom WHERE kode_mk = 'COM60014' AND penilai = '2'")
    r = jalankan_penilaian_bloom(cfg)
    assert (r["n_permintaan_api"], r["n_mk_dinilai"]) == (1, 1)
    ulang = jalankan_penilaian_bloom(cfg, kode_list=["COM60014"], lanjutkan=False)
    assert ulang["n_permintaan_api"] == 3 and db.jumlah(cfg, "rating_bloom") == len(df)


def test_penilaian_batch_meninggalkan_jejak(tmp_path, monkeypatch):
    """Tracing: setiap permintaan tercatat real-time di app.log dan disimpan utuh (prompt, keluaran mentah,
    peta label lokal → id_cpmk) di logs/llm/; batch_llm merujuk berkasnya beserta durasi."""
    monkeypatch.setattr(beban, "buat_client", lambda cfg, *a, **k: _KlienPrompt("uji-jejak", cfg))
    cfg = cfg_tmp(tmp_path, llm={"konkurensi": 1})
    r = jalankan_penilaian_bloom(cfg, kode_list=SK.kode_dengan_cpmk()[:3])
    batch = db.baca(cfg, "SELECT * FROM batch_llm")
    assert len(batch) == 3 and batch["durasi_detik"].notna().all()
    for berkas in batch["berkas_jejak"]:
        isi = json.loads(Path(berkas).read_text(encoding="utf-8"))
        assert Path(berkas).parent == Path(r["folder_jejak"])
        assert isi["keluaran_mentah"] and isi["prompt"]["user"] and isi["galat"] is None
        assert {v["kode_mk"] for v in isi["peta_label"].values()} == set(SK.kode_dengan_cpmk()[:3])
    teks = (cfg.logs_dir / "app.log").read_text(encoding="utf-8")
    assert "kirim r1-b1 percobaan 1" in teks and "selesai r3-b1 percobaan 1: ok" in teks


def test_jejak_mencatat_galat(tmp_path):
    """Keluaran yang tidak bisa diparse: keluaran mentah & traceback tersimpan di berkas jejak."""
    class Rusak(LLMClient):
        provider = "uji"

        def complete(self, system, user, skema=None):
            return LLMResponse(teks="bukan json", model="m", id_respons="resp_123")

    mks = [SK.mk("COM60014")]
    hb = nilai_batch(Rusak("x", CFG), mks, 1, CFG, TEMPLAT, {}, "r1-b1", tmp_path)
    assert hb.gagal == ["COM60014"] and len(hb.permintaan) == CFG.llm.max_retry + 1
    isi = json.loads(Path(hb.permintaan[0]["berkas_jejak"]).read_text(encoding="utf-8"))
    assert isi["keluaran_mentah"] == "bukan json" and "ValueError" in isi["galat"]
    assert isi["entri"]["id_respons"] == "resp_123"


def test_mk_per_batch_memotong_permintaan(tmp_path, monkeypatch):
    monkeypatch.setattr(beban, "buat_client", lambda cfg, *a, **k: _KlienPrompt("uji-batch", cfg))
    cfg = cfg_tmp(tmp_path, llm={"konkurensi": 1, "mk_per_batch": 30, "n_pengulangan": 1})
    r = jalankan_penilaian_bloom(cfg)
    assert r["n_batch"] == 3 and sorted(db.baca(cfg, "SELECT n_mk FROM batch_llm")["n_mk"]) == [10, 30, 30]


def test_konsistensi_pengulangan_hitungan_manual():
    r = pd.DataFrame([{"kode_mk": k, "id_cpmk": i, "penilai": p, "bloom": b, "status": "ok"}
                      for k, i, p, b in [("X", "X-1", "1", "C3"), ("X", "X-1", "2", "C3"), ("X", "X-1", "3", "C3"),
                                         ("X", "X-2", "1", "C2"), ("X", "X-2", "2", "C4"), ("X", "X-2", "3", "C3"),
                                         ("Y", "Y-1", "1", "A"), ("Y", "Y-1", "2", "C3"), ("Y", "Y-1", "3", "A")]])
    df, rk = konsistensi_pengulangan(r)
    d = df.set_index("id_cpmk")
    assert list(d["sepakat_penuh"]) == [True, False, False] and list(d["rentang_level"]) == [0, 2, 0]
    assert list(d["konsensus"]) == ["C3", "C3", "A"] and d.loc["Y-1", "campur_afektif"]
    assert rk == {"n_cpmk": 3, "proporsi_sepakat_penuh": round(1 / 3, 4), "proporsi_rentang_le1": round(2 / 3, 4),
                  "n_campur_afektif": 1, "n_mk_semua_sepakat": 0}


# ---- konsensus & CL per MK ---------------------------------------------------------------

def _rating(rows):
    return pd.DataFrame([{"id_cpmk": i, "kode_mk": k, "bloom": b, "status": s} for i, k, b, s in rows])


def test_konsensus_bloom_hitungan_manual():
    r = _rating([("X-1", "X", "C3", "ok"), ("X-1", "X", "C4", "ok"), ("X-1", "X", "C5", "ok"),      # median → C4
                 ("X-2", "X", "C3", "ok"), ("X-2", "X", "C5", "ok"),                                # median bawah → C3
                 ("X-3", "X", "A", "ok"), ("X-3", "X", "A", "ok"), ("X-3", "X", "C2", "ok"),        # mayoritas A
                 ("X-4", "X", "A", "ok"), ("X-4", "X", "C2", "ok"), ("X-4", "X", "C4", "ok"),       # A minoritas → C2
                 ("X-5", "X", "C6", "perlu_tinjau_manual"), ("X-5", "X", "C1", "ok")])               # gagal diabaikan
    k = konsensus_bloom(r).set_index("id_cpmk")
    assert k["bloom"].to_dict() == {"X-1": "C4", "X-2": "C3", "X-3": "A", "X-4": "C2", "X-5": "C1"}
    assert k.loc["X-5", "n_penilaian"] == 1 and bool(k.loc["X-5", "sepakat"])


def test_cl_mk_agregasi_dan_afektif():
    """COM60014: C2, C6, C3 → median 3, maks 6, rata 3,6667; MPK60003: A, A → tanpa CL; UBU60029: tanpa CPMK."""
    kons = pd.DataFrame({"id_cpmk": ["a", "b", "c", "d", "e"],
                         "kode_mk": ["COM60014"] * 3 + ["MPK60003"] * 2, "bloom": ["C2", "C6", "C3", "A", "A"]})
    for metode, harap in (("median", 3.0), ("maks", 6.0), ("rata", 3.6667)):
        t = cl_mk(kons, KUR, metode).set_index("kode_mk")
        assert t.loc["COM60014", "cl"] == pytest.approx(harap)
    t = cl_mk(kons, KUR, "median").set_index("kode_mk")
    assert pd.isna(t.loc["MPK60003", "cl"]) and "afektif" in t.loc["MPK60003", "keterangan"]
    assert pd.isna(t.loc["UBU60029", "cl"]) and "§7a" in t.loc["UBU60029", "keterangan"]
    assert t.loc["COM60014", "n_cpmk_kognitif"] == 3


def test_cl_mk_seluruhnya_afektif_memakai_penilaian_individual_lalu_cl_afektif():
    """MPK60003 berkonsensus A semua: (1) ada pengulangan yang memberi C3 → CL 3 dari penilaian individual;
    (2) tanpa penilaian kognitif sama sekali → beban.cl_afektif. MK tanpa CPMK tetap tanpa CL."""
    kons = pd.DataFrame({"id_cpmk": ["d", "e"], "kode_mk": ["MPK60003"] * 2, "bloom": ["A", "A"]})
    rating = pd.DataFrame({"kode_mk": ["MPK60003"] * 4, "id_cpmk": ["d", "d", "d", "e"],
                           "bloom": ["A", "C3", "A", "A"], "status": "ok"})
    t = cl_mk(kons, KUR, "median", rating, cl_afektif=1.0).set_index("kode_mk")
    assert t.loc["MPK60003", "cl"] == 3.0 and "penilaian kognitif individual" in t.loc["MPK60003", "keterangan"]
    hanya_a = rating.assign(bloom="A")
    t = cl_mk(kons, KUR, "median", hanya_a, cl_afektif=1.0).set_index("kode_mk")
    assert t.loc["MPK60003", "cl"] == 1.0 and pd.isna(t.loc["UBU60029", "cl"])


def test_justifikasi_cpmk_dari_penilai_yang_sepakat():
    """KF-17: konsensus X1 = C3 (median bawah [C3, C3, C5]); alasan diambil dari penilai pertama berlabel C3.
    X2 = A (mayoritas afektif); alasan dari penilai berlabel A."""
    rating = pd.DataFrame([
        {"id_cpmk": "X1", "kode_mk": "K", "penilai": "1", "bloom": "C5", "alasan": "evaluasi", "status": "ok"},
        {"id_cpmk": "X1", "kode_mk": "K", "penilai": "2", "bloom": "C3", "alasan": "menerapkan", "status": "ok"},
        {"id_cpmk": "X1", "kode_mk": "K", "penilai": "3", "bloom": "C3", "alasan": "lain", "status": "ok"},
        {"id_cpmk": "X2", "kode_mk": "K", "penilai": "1", "bloom": "A", "alasan": "sikap", "status": "ok"},
        {"id_cpmk": "X2", "kode_mk": "K", "penilai": "2", "bloom": "A", "alasan": "", "status": "ok"},
        {"id_cpmk": "X2", "kode_mk": "K", "penilai": "3", "bloom": "C2", "alasan": "x", "status": "ok"},
    ])
    assert justifikasi_cpmk(rating, konsensus_bloom(rating)) == {"K": ["X1: C3, menerapkan", "X2: A, sikap"]}


def test_cl_heuristik_seluruh_kurikulum():
    """Tanpa rating LLM (basis data tes kosong) → CL kembali ke heuristik KKO untuk seluruh MK ber-CPMK."""
    b = muat_beban_mk(CFG, KUR)
    assert b.info["catatan"] == "rating 'llm' belum tersedia → memakai heuristik KKO"
    cl, info = b.cl, b.info
    assert len(cl) == 70 and set(KUR.kode) - set(cl) == {"UBU60029", "UBU60030"}
    assert all(1 <= v <= 6 for v in cl.values())
    assert info["sumber_rating"] == "heuristik" and info["metode"] == "median"


def test_cl_sumber_belum_ada_kembali_ke_heuristik(tmp_path):
    cfg = cfg_tmp(tmp_path, beban={"sumber_cl": "llm"})
    info = muat_beban_mk(cfg, KUR).info
    assert info["sumber_rating"] == "heuristik" and "belum tersedia" in info["catatan"]


def test_rating_pakar_template_dan_validasi(tmp_path):
    from src.export import template_pakar

    cfg = cfg_tmp(tmp_path, beban={"sumber_cl": "pakar"})
    path = template_pakar(cfg)
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(t) == 348 and list(t.columns) == ["id_cpmk", "kode_mk", "nama_mk", "teks", "kode_pakar", "bloom",
                                                 "catatan"]
    isi = t.copy()
    isi["kode_pakar"], isi["bloom"] = "P1", "C4"
    isi.loc[isi["kode_mk"] == "MPK60003", "bloom"] = "a"
    r = impor_pakar(cfg, isi)
    assert r == {"n_pakar": 1, "pakar": ["P1"], "n_baris": 348, "n_dinilai": 348, "n_cpmk_kurikulum": 348}
    b = muat_beban_mk(cfg, KUR)
    cl, info = b.cl, b.info
    assert info["sumber_rating"] == "pakar" and cl["COM60014"] == 4.0
    assert cl["MPK60003"] == CFG.beban.cl_afektif and "beban.cl_afektif" in b.keterangan["MPK60003"]   # semua A
    for ubah, pesan in [({"bloom": "C9"}, "label bloom"), ({"kode_pakar": ""}, "tanpa kode_pakar"),
                        ({"id_cpmk": "XXX-CPMK01"}, "id_cpmk tidak dikenal"), ({"kode_mk": "COM60015"}, "tidak cocok")]:
        salah = isi.copy()
        for k, v in ubah.items():
            salah.loc[0, k] = v
        with pytest.raises(ValueError, match=pesan):
            impor_pakar(cfg, salah)
    with pytest.raises(ValueError, match="ganda"):
        impor_pakar(cfg, pd.concat([isi, isi.head(1)]))
    assert db.jumlah(cfg, "rating_bloom", "sumber = 'pakar'") == 348          # impor gagal tidak mengubah data
    isi2 = isi.copy()
    isi2["kode_pakar"] = "P2"
    isi2["bloom"] = ""
    r2 = impor_pakar(cfg, isi2)
    assert r2["n_dinilai"] == 0 and db.jumlah(cfg, "rating_bloom", "sumber = 'pakar'") == 696
    pakar = muat_rating(cfg, "pakar")
    assert set(pakar.loc[pakar["penilai"] == "P2", "status"]) == {"belum_dinilai"}


def test_templat_prompt_wajib_bertanda(tmp_path, monkeypatch):
    import src.llm.prompts as pr

    (tmp_path / "x_v1.md").write_text("tanpa penanda", encoding="utf-8")
    monkeypatch.setattr(pr, "PROMPT_DIR", tmp_path)
    with pytest.raises(ValueError, match="SYSTEM"):
        pr.muat_prompt("x", "v1")
    with pytest.raises(FileNotFoundError):
        pr.muat_prompt("y", "v9")


# ---- Anthropic: parameter yang ditolak model ------------------------------

def _klien_anthropic(cfg, tolak: dict[str, str], konten=None, model="claude-opus-5-5"):
    """AnthropicClient dengan SDK palsu: ``tolak`` = {parameter: pesan galat 400}."""
    class BadReq(Exception):
        pass

    class Lain(Exception):
        pass

    sdk = SimpleNamespace(BadRequestError=BadReq, AuthenticationError=Lain, NotFoundError=Lain, APIError=Lain)
    c = object.__new__(AnthropicClient)
    LLMClient.__init__(c, model, cfg)
    c._sdk, c.ditolak = sdk, set()
    panggilan = []

    def create(**kw):
        panggilan.append(kw)
        for param, pesan in tolak.items():
            if param in kw:
                raise BadReq(pesan)
        return SimpleNamespace(stop_reason="end_turn", model="claude-uji", content=konten or [
            SimpleNamespace(type="thinking", thinking="Kata kerja 'merancang' → C6."),
            SimpleNamespace(type="text", text='{"penilaian": []}')])

    c._create = create
    return c, panggilan


def test_anthropic_thinking_adaptif_dan_effort():
    """Default config: adaptive thinking (ringkasan disimpan) + effort; temperature tidak dikirim bersama
    thinking (model Claude terbaru menolak parameter sampling); fallbacks server-side."""
    c, panggilan = _klien_anthropic(CFG, {})
    resp = c.complete("sys", "user")
    kw = panggilan[0]
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kw["output_config"] == {"effort": "high"} and "temperature" not in kw
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert resp.ringkasan_thinking == "Kata kerja 'merancang' → C6."
    assert parse_json_keluaran(resp.teks) == {"penilaian": []}
    assert c.thinking_dipakai and c.temperature_dipakai is None


def test_anthropic_melepas_parameter_yang_ditolak():
    """Model lama: thinking adaptif & effort ditolak → dilepas berurutan, lalu temperature dikirim; diingat."""
    tanpa_thinking = load_config({"llm": {"thinking": False}}, use_override_file=False, use_env=False)
    c, panggilan = _klien_anthropic(tanpa_thinking, {"temperature": "temperature: not supported for this model"})
    c.complete("sys", "user")
    assert panggilan[0]["temperature"] == 0 and "temperature" not in panggilan[1]
    assert c.temperature_dipakai is None and not c.thinking_dipakai
    c.complete("sys", "user")
    assert "temperature" not in panggilan[2]                     # diingat, tidak dicoba ulang

    c, panggilan = _klien_anthropic(CFG, {"thinking": "thinking.type: adaptive is not supported",
                                          "output_config": "effort is not supported on this model"})
    resp = c.complete("sys", "user")
    terakhir = panggilan[-1]
    assert "thinking" not in terakhir and "output_config" not in terakhir and terakhir["temperature"] == 0
    assert c.temperature_dipakai == 0 and not c.thinking_dipakai
    assert any("thinking ditolak" in x for x in resp.catatan)


def test_anthropic_structured_output_json_schema():
    """Skema diteruskan sebagai output_config.format (setara text.format OpenAI); bila model menolak skema,
    permintaan diulang tanpa skema (effort tetap dikirim) dan penolakan diingat."""
    skema = {"type": "object", "properties": {"penilaian": {"type": "array"}}, "required": ["penilaian"],
             "additionalProperties": False}
    c, panggilan = _klien_anthropic(CFG, {})
    c.complete("sys", "user", skema=skema)
    assert panggilan[0]["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": skema}}

    c, panggilan = _klien_anthropic(CFG, {})
    asli = c._create

    def tolak_skema(**kw):
        if "format" in kw.get("output_config", {}):
            panggilan.append(kw)
            raise c._sdk.BadRequestError("output_config.format: schema type not supported")
        return asli(**kw)
    c._create = tolak_skema
    resp = c.complete("sys", "user", skema=skema)
    assert panggilan[-1]["output_config"] == {"effort": "high"} and "format" in c.ditolak
    assert any("format ditolak" in x for x in resp.catatan)
    c.complete("sys", "user", skema=skema)
    assert "format" not in panggilan[-1]["output_config"]       # diingat


def test_anthropic_haiku_tanpa_fallback_dan_temperature_lewat_extra_body():
    """Haiku 5.5 tidak punya server-side fallback → ``fallbacks`` tidak dikirim. SDK anthropic 1.x tidak punya
    argumen ``temperature`` → ``_create`` memindahkannya ke ``extra_body`` dan selalu streaming."""
    c, panggilan = _klien_anthropic(CFG, {}, model="claude-haiku-5-5")
    c.complete("sys", "user")
    assert "fallbacks" not in panggilan[0] and "betas" not in panggilan[0]

    from src.llm.client import AnthropicClient
    terkirim = {}

    class Aliran:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return "pesan"

    def stream(**kw):
        terkirim.update(kw)
        return Aliran()

    k = object.__new__(AnthropicClient)
    k._client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    assert k._create(model="m", temperature=0) == "pesan"
    assert terkirim == {"model": "m", "extra_body": {"temperature": 0}}


def test_anthropic_refusal_dan_400_lain():
    c, _ = _klien_anthropic(CFG, {"messages": "messages: invalid"})
    with pytest.raises(LLMError, match="ditolak API"):
        c.complete("sys", "user")
    c, _ = _klien_anthropic(CFG, {})
    c._create = lambda **kw: SimpleNamespace(stop_reason="refusal", model="x", content=[])
    with pytest.raises(LLMError, match="refusal"):
        c.complete("sys", "user")


def test_buat_client_default_model():
    k = buat_client(CFG)
    assert k.provider == "mock" and k.model == "mock-bloom-v1"
    with pytest.raises(LLMKonfigurasiError):
        buat_client(CFG, provider="tidak-ada")
    assert buat_client(CFG, "mock").temperature_dipakai is None    # mock deterministik
    from src.llm.client import MODEL_DEFAULT
    assert MODEL_DEFAULT["anthropic"] == "claude-opus-5-5" and MODEL_DEFAULT["gemini"] == "gemini-3.8-flash"


# ---- Gemini (google-genai): structured output, thinking level, parameter yang ditolak -----------------

def _klien_gemini(cfg, tolak: dict | None = None, alasan="STOP"):
    from src.llm.client import GeminiClient

    class APIError(Exception):
        def __init__(self, code, pesan):
            super().__init__(pesan)
            self.code = code

    tipe = SimpleNamespace(ThinkingConfig=lambda **kw: kw, GenerateContentConfig=lambda **kw: kw)
    c = object.__new__(GeminiClient)
    LLMClient.__init__(c, "gemini-3.8-flash", cfg)
    c._types, c._errors = tipe, SimpleNamespace(APIError=APIError)
    panggilan = []

    def create(**kw):
        panggilan.append(kw)
        for param, (kode, pesan) in (tolak or {}).items():
            if param in kw:
                raise APIError(kode, pesan)
        bagian = [SimpleNamespace(text="Kata kerja merancang.", thought=True),
                  SimpleNamespace(text='{"penilaian": []}', thought=False)]
        return SimpleNamespace(
            model_version="gemini-3.8-flash-001",
            candidates=[SimpleNamespace(finish_reason=alasan, content=SimpleNamespace(parts=bagian))],
            usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=30))

    c._create = create
    return c, panggilan


def test_gemini_structured_output_thinking_level_dan_token():
    c, panggilan = _klien_gemini(CFG)
    skema = skema_penilaian(["CPMK01"])
    resp = c.complete("sys", "user", skema)
    kw = panggilan[0]
    assert kw["system_instruction"] == "sys" and kw["contents"] == "user" and "temperature" not in kw
    assert kw["thinking_config"] == {"thinking_level": "HIGH", "include_thoughts": True}
    assert kw["response_mime_type"] == "application/json" and kw["response_json_schema"] == skema
    assert parse_json_keluaran(resp.teks) == {"penilaian": []}          # bagian 'thought' tidak ikut teks
    assert resp.ringkasan_thinking == "Kata kerja merancang." and resp.model == "gemini-3.8-flash-001"
    assert (resp.token_masuk, resp.token_keluar, resp.thinking) == (100, 50, True)


def test_gemini_melepas_parameter_ditolak_dan_galat():
    c, panggilan = _klien_gemini(CFG, {"thinking_config": (400, "thinking_level is not supported for this model")})
    resp = c.complete("sys", "user")
    assert "thinking_config" not in panggilan[-1] and panggilan[-1]["temperature"] == 0
    assert "thinking_config" in c.ditolak and any("thinking_config ditolak" in x for x in resp.catatan)
    c, _ = _klien_gemini(CFG, {"contents": (403, "API key not valid")})
    with pytest.raises(LLMKonfigurasiError):
        c.complete("sys", "user")
    c, _ = _klien_gemini(CFG, {"contents": (429, "RESOURCE_EXHAUSTED. You exceeded your current quota")})
    with pytest.raises(LLMKonfigurasiError, match="Kuota"):            # kuota habis: tidak diulang
        c.complete("sys", "user")
    c, _ = _klien_gemini(CFG, {"contents": (500, "internal")})
    with pytest.raises(LLMError):
        c.complete("sys", "user")
    c, _ = _klien_gemini(CFG, alasan="SAFETY")
    with pytest.raises(LLMError, match="menolak"):
        c.complete("sys", "user")
    c, _ = _klien_gemini(CFG, alasan="MAX_TOKENS")
    assert c.complete("sys", "user").terpotong


# ---- OpenAI (Responses API): structured output, effort, parameter yang ditolak -------------------

def _klien_openai(cfg, tolak: dict[str, str] | None = None, respons=None):
    """OpenAIClient dengan SDK palsu: ``tolak`` = {param: pesan} → BadRequestError ber-``param`` (seperti API)."""
    from src.llm.client import OpenAIClient

    class BadReq(Exception):
        def __init__(self, pesan, param):
            super().__init__(pesan)
            self.param = param

    class Lain(Exception):
        pass

    sdk = SimpleNamespace(BadRequestError=BadReq, AuthenticationError=Lain, PermissionDeniedError=Lain,
                          NotFoundError=Lain, APIError=Lain)
    c = object.__new__(OpenAIClient)
    LLMClient.__init__(c, "gpt-6-luna", cfg)
    c._sdk = sdk
    panggilan = []

    def create(**kw):
        panggilan.append(kw)
        for param, pesan in (tolak or {}).items():
            kunci = param.split(".")[0]
            if kunci in kw and (param != "reasoning.effort" or kw["reasoning"]["effort"] in ("xhigh", "max")):
                raise BadReq(pesan, param)
        return respons or SimpleNamespace(
            status="completed", model="gpt-6-luna", output_text='{"penilaian": []}',
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text")])],
            usage=SimpleNamespace(input_tokens=120, output_tokens=45))

    c._create = create
    return c, panggilan


def test_openai_structured_output_reasoning_tanpa_temperature():
    """Model penalaran: reasoning.effort + JSON schema ketat; temperature tidak dikirim; store=False; token dicatat."""
    c, panggilan = _klien_openai(CFG)
    skema = skema_penilaian(["CPMK01"])
    resp = c.complete("sys", "user", skema)
    kw = panggilan[0]
    assert kw["instructions"] == "sys" and kw["input"] == "user" and kw["store"] is False
    assert kw["reasoning"] == {"effort": "high"} and "temperature" not in kw
    assert kw["text"]["format"] == {"type": "json_schema", "name": "keluaran", "schema": skema, "strict": True}
    assert (resp.token_masuk, resp.token_keluar, resp.thinking, resp.temperature) == (120, 45, True, None)
    assert "structured output (JSON schema ketat)" in resp.catatan
    c.complete("sys", "user")
    assert "text" not in panggilan[-1]                                  # tanpa skema → tanpa text.format


def test_openai_melepas_parameter_yang_ditolak_dan_menurunkan_effort():
    xhigh = load_config({"llm": {"effort": "xhigh"}}, use_override_file=False, use_env=False)
    c, panggilan = _klien_openai(xhigh, {"reasoning.effort": "Unsupported value: 'xhigh'"})
    resp = c.complete("sys", "user")
    assert panggilan[0]["reasoning"] == {"effort": "xhigh"} and panggilan[1]["reasoning"] == {"effort": "high"}
    assert any("diturunkan ke high" in x for x in resp.catatan)
    c.complete("sys", "user")
    assert panggilan[-1]["reasoning"] == {"effort": "high"}              # diingat

    c, panggilan = _klien_openai(CFG, {"reasoning": "Unsupported parameter: 'reasoning'",
                                       "text": "Unsupported parameter: 'text.format'"})
    resp = c.complete("sys", "user", skema_penilaian(["CPMK01"]))
    terakhir = panggilan[-1]
    assert "reasoning" not in terakhir and "text" not in terakhir and terakhir["temperature"] == 0
    assert resp.temperature == 0 and not resp.thinking


def test_openai_galat_lain_penolakan_dan_tidak_lengkap():
    c, _ = _klien_openai(CFG, {"input": "Invalid 'input'"})
    with pytest.raises(LLMError, match="ditolak API"):
        c.complete("sys", "user")
    tolak = SimpleNamespace(status="completed", model="m", output_text="", usage=None, output=[
        SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal="tidak bisa")])])
    c, _ = _klien_openai(CFG, respons=tolak)
    with pytest.raises(LLMError, match="menolak"):
        c.complete("sys", "user")
    potong = SimpleNamespace(status="incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                             model="m", output_text="{}", output=[], usage=None)
    c, _ = _klien_openai(CFG, respons=potong)
    assert any("tidak lengkap (max_output_tokens)" in x for x in c.complete("sys", "user").catatan)


def test_openai_base_url_kosong_memakai_endpoint_resmi(monkeypatch):
    """OPENAI_BASE_URL berisi spasi (.env) tidak boleh menjadi URL kosong (SDK membaca variabel ini sendiri)."""
    pytest.importorskip("openai")
    from src.llm.client import URL_OPENAI, OpenAIClient

    monkeypatch.setenv("OPENAI_BASE_URL", " ")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-uji")
    c = OpenAIClient("gpt-6-luna", CFG)
    assert str(c._client.base_url).rstrip("/") == URL_OPENAI


def test_openai_create_streaming():
    """``_create`` selalu streaming dan mengembalikan respons final; response.failed / putus di tengah stream
    menjadi ``LLMError`` (ditangani retry per batch, bukan menghentikan seluruh run). Galat transport berasal
    dari pustaka HTTP yang dipakai SDK (``httpx2``), bukan ``httpx``."""
    openai = pytest.importorskip("openai")
    from src.llm.client import OpenAIClient, modul_http

    hx = modul_http(openai)

    terkirim = {}

    def klien(peristiwa):
        def create(**kw):
            terkirim.update(kw)
            return peristiwa()
        k = object.__new__(OpenAIClient)
        k._client = SimpleNamespace(responses=SimpleNamespace(create=create))
        k._sdk = openai
        return k

    ok = klien(lambda: iter([SimpleNamespace(type="response.output_text.delta"),
                             SimpleNamespace(type="response.completed", response="final")]))
    assert ok._create(model="m", store=False) == "final"
    assert terkirim == {"model": "m", "store": False, "stream": True}

    gagal = klien(lambda: iter([SimpleNamespace(type="response.failed",
                                                response=SimpleNamespace(error=SimpleNamespace(message="x")))]))
    with pytest.raises(LLMError, match="respons gagal"):
        gagal._create(model="m")

    def putus():
        yield SimpleNamespace(type="response.created")
        raise hx.ReadTimeout("habis")
    with pytest.raises(LLMError, match="stream terputus"):
        klien(putus)._create(model="m")
    with pytest.raises(LLMError, match="tanpa respons final"):
        klien(lambda: iter([]))._create(model="m")


@pytest.mark.parametrize("nama_sdk", ["openai", "anthropic"])
def test_transport_tercatat_merekam_retry_sdk(nama_sdk, caplog):
    """Tanpa jaringan: SDK sungguhan + transport tiruan 429 → 500 → koneksi putus → 200. Setiap percobaan
    (termasuk retry diam-diam SDK) tercatat di ``crs.llm.http`` beserta nomor retry & request-id."""
    sdk = pytest.importorskip(nama_sdk)
    from src.llm.client import http_client_tercatat, modul_http

    hx = modul_http(sdk)
    urutan = iter([429, 500, "putus", 200])

    def jawab(request):
        x = next(urutan)
        if x == "putus":
            raise hx.ConnectError("koneksi putus", request=request)
        tubuh = {"object": "list", "data": [], "has_more": False, "first_id": None, "last_id": None}
        return hx.Response(x, json=tubuh if x == 200 else {"error": {"message": "galat", "type": "x"}},
                           headers={"x-request-id": f"req_{x}", "request-id": f"req_{x}", "retry-after-ms": "1"})

    kelas = sdk.OpenAI if nama_sdk == "openai" else sdk.Anthropic
    klien = kelas(api_key="sk-uji", base_url="https://contoh.invalid/v1", max_retries=3,
                  http_client=http_client_tercatat(sdk, nama_sdk, hx.MockTransport(jawab)))
    with caplog.at_level("INFO", logger="crs.llm.http"):
        klien.models.list()
    log = [r.getMessage() for r in caplog.records if r.name == "crs.llm.http"]
    assert len(log) == 4
    assert "→ 429 (retry SDK ke-0" in log[0] and "req_429" in log[0]
    assert "→ 500 (retry SDK ke-1" in log[1]
    assert "GALAT ConnectError" in log[2] and "retry SDK ke-2" in log[2]
    assert "→ 200 (retry SDK ke-3" in log[3]
    assert [r.levelname for r in caplog.records if r.name == "crs.llm.http"] == ["WARNING"] * 3 + ["INFO"]


def test_klien_sungguhan_memakai_transport_tercatat(monkeypatch):
    pytest.importorskip("openai")
    from src.llm.client import OpenAIClient

    monkeypatch.setenv("OPENAI_API_KEY", "sk-uji")
    c = OpenAIClient("gpt-6-luna", CFG)
    assert type(c._client._client._transport).__name__ == "TransportTercatat"


def test_skema_penilaian_sesuai_kontrak():
    s = skema_penilaian(["CPMK01", "CPMK02"])
    butir = s["properties"]["penilaian"]["items"]
    assert butir["properties"]["id_cpmk"]["enum"] == ["CPMK01", "CPMK02"]
    assert set(butir["required"]) == set(butir["properties"]) and butir["additionalProperties"] is False
    assert butir["properties"]["bloom"]["enum"] == ["C1", "C2", "C3", "C4", "C5", "C6", "A"]


def test_penilaian_paralel_sama_dengan_berurutan(tmp_path, monkeypatch):
    """llm.konkurensi > 1 (pengulangan paralel) menghasilkan rating yang sama dengan eksekusi berurutan;
    penulisan basis data tetap di thread utama."""
    monkeypatch.setattr(beban, "buat_client", lambda cfg, *a, **k: _KlienPrompt("uji-paralel", cfg))
    kode = SK.kode_dengan_cpmk()[:5]
    hasil = {}
    for n in (1, 3):
        cfg = buat_cfg(tmp_path / f"k{n}", llm={"konkurensi": n})
        r = jalankan_penilaian_bloom(cfg, kode_list=kode)
        assert (r["n_permintaan_api"], r["konkurensi"], r["token_masuk"]) == (3, n, 30)
        df = beban.muat_rating(cfg, "llm")
        hasil[n] = sorted(zip(df["id_cpmk"], df["penilai"], df["bloom"]))
    assert hasil[1] == hasil[3] and len(hasil[1]) > 0
