"""Tes pemuatan konfigurasi (KNF-07)."""
import pytest
from pydantic import ValidationError

from src.config import deep_merge, load_config


def test_config_default_termuat():
    cfg = load_config(use_override_file=False, use_env=False)
    assert cfg.kurikulum.horizon_semester == 8 and cfg.kurikulum.perpanjang_melewati_horizon
    assert cfg.arm.min_support == 0.2
    assert cfg.llm.provider == "mock"                      # tes selalu luring (tests/conftest.py)
    import yaml
    from src.config import CONFIG_PATH
    llm = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["llm"]
    assert (llm["provider"], llm["model"]) == ("openai", "gpt-6-luna")   # selaras dengan .env produksi
    assert (cfg.kurikulum_dir / "00_kamus_data.csv").exists()
    a = cfg.aturan
    # Prasyarat harus lulus (AD-03); lulus = bukan E (AD-07); Skripsi ≥ 120 SKS & IPK ≥ 2,00 (AD-10)
    assert not a.PRASYARAT_E_BERARTI_PERNAH_MENEMPUH and a.MODE_PENAWARAN_PILIHAN == "paritas"
    assert a.NILAI_LULUS_MINIMUM == "D" and a.PENAWARAN_DARI_REKOMENDASI_09
    assert a.NILAI_DIBATASI == ["D+", "D"] and a.BATAS_PROPORSI_NILAI_DIBATASI == 0.10
    assert (a.SKRIPSI.kode, a.SKRIPSI.min_sks_lulus, a.SKRIPSI.min_ipk) == ("UBU60001", 120, 2.0)
    assert [a.bobot(n) for n in ["A", "b+", "C", "E", "T"]] == [4.0, 3.5, 2.0, 0.0, None]


def test_ambang_nilai_lulus():
    a = load_config(use_override_file=False, use_env=False).aturan
    assert [a.lulus(n) for n in ["A", "B+", "C", "D+", "D", "E"]] == [True, True, True, True, True, False]
    assert a.lulus(" b+ ") and not a.lulus("T") and not a.lulus(None)
    c = load_config({"aturan": {"NILAI_LULUS_MINIMUM": "C"}}, use_override_file=False, use_env=False).aturan
    assert c.lulus("C") and not c.lulus("D+")


def test_config_menolak_nilai_tidak_valid():
    with pytest.raises(ValidationError):
        load_config({"aturan": {"NILAI_LULUS_MINIMUM": "Z"}}, use_override_file=False, use_env=False)
    with pytest.raises(ValidationError, match="BOBOT_NILAI"):
        load_config({"aturan": {"BOBOT_NILAI": {"Z": 1.0}}}, use_override_file=False, use_env=False)
    with pytest.raises(ValidationError, match="NILAI_DIBATASI"):
        load_config({"aturan": {"NILAI_DIBATASI": ["X"]}}, use_override_file=False, use_env=False)
    with pytest.raises(ValidationError):
        load_config({"aturan": {"MODE_PENAWARAN_PILIHAN": "bebas"}}, use_override_file=False, use_env=False)
    with pytest.raises(ValidationError):
        load_config({"kurikulum": {"horizon_semester": 15}}, use_override_file=False, use_env=False)


def test_deep_merge_tidak_mengubah_asal():
    a = {"x": {"y": 1, "z": 2}}
    b = deep_merge(a, {"x": {"y": 5}})
    assert b == {"x": {"y": 5, "z": 2}}
    assert a == {"x": {"y": 1, "z": 2}}


def test_save_override_mengganti_kunci_utuh(tmp_path, monkeypatch):
    """BATAS_SKS_IP diganti utuh (bukan di-merge) agar baris yang dikosongkan di UI benar-benar terhapus."""
    import src.config as c

    monkeypatch.setattr(c, "OVERRIDE_PATH", tmp_path / "override.yaml")
    c.save_override({"aturan": {"BATAS_SKS_IP": {3.0: [20, 22], 2.5: [18, 20]}}}, ganti=("aturan.BATAS_SKS_IP",))
    c.save_override({"aturan": {"BATAS_SKS_IP": {3.0: [21, 23]}}}, ganti=("aturan.BATAS_SKS_IP",))
    assert c.read_override()["aturan"]["BATAS_SKS_IP"] == {3.0: [21, 23]}
    c.save_override({"aturan": {"BATAS_SKS_IP": None}}, ganti=("aturan.BATAS_SKS_IP",))
    assert load_config(use_env=False).aturan.BATAS_SKS_IP is None
