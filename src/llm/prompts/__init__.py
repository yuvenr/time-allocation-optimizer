"""Templat prompt berversi (CLAUDE.md §5: ``llm/prompts/`` v1, v2, ...).

Setiap berkas ``<tugas>_<versi>.md`` berisi dua bagian yang dipisah baris penanda
``=== SYSTEM ===`` dan ``=== USER ===``. Placeholder memakai ``string.Template``
(``$nama``) sehingga contoh JSON di dalam prompt tidak perlu di-escape. Versi
prompt (``<tugas>-<versi>``) disimpan pada setiap baris hasil LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template

PROMPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TemplatPrompt:
    tugas: str
    versi: str
    system: str
    user: Template

    @property
    def id(self) -> str:
        return f"{self.tugas}-{self.versi}"


def muat_prompt(tugas: str, versi: str) -> TemplatPrompt:
    path = PROMPT_DIR / f"{tugas}_{versi}.md"
    if not path.exists():
        raise FileNotFoundError(f"Templat prompt tidak ditemukan: {path}")
    teks = path.read_text(encoding="utf-8")
    if "=== SYSTEM ===" not in teks or "=== USER ===" not in teks:
        raise ValueError(f"{path.name}: harus memuat penanda '=== SYSTEM ===' dan '=== USER ==='")
    system, user = teks.split("=== SYSTEM ===", 1)[1].split("=== USER ===", 1)
    return TemplatPrompt(tugas, versi, system.strip(), Template(user.strip()))
