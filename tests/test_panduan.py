"""Panduan pengguna (docs/PANDUAN_PENGGUNA.md) mencakup setiap tombol aplikasi."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANDUAN = "".join((ROOT / "docs" / f).read_text(encoding="utf-8")
                  for f in ("PANDUAN_PENGGUNA.md", "PANDUAN_ADMINISTRATOR.md"))


def test_setiap_tombol_dijelaskan_di_panduan():
    kode = "".join((ROOT / "app" / f).read_text(encoding="utf-8") for f in ("streamlit_app.py", "tampilan.py"))
    label = set(re.findall(r'\.(?:button|form_submit_button|download_button)\(\s*"([^"{]+)"', kode))
    assert len(label) >= 25
    kurang = sorted(lb for lb in label if lb.split(" (")[0] not in PANDUAN)
    assert not kurang, f"tombol belum dijelaskan di PANDUAN_PENGGUNA.md / PANDUAN_ADMINISTRATOR.md: {kurang}"

