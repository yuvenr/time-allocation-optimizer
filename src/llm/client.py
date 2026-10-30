"""Abstraksi klien LLM + parsing JSON + validasi + retry (UC-09).

Provider:
- ``mock``      : offline & deterministik. Jawaban disuntikkan oleh tugas pemanggil lewat
                  ``MockClient.penjawab`` (penilaian Bloom = heuristik KKO dataset) sehingga
                  seluruh alur — prompt → klien → parsing → validasi → penyimpanan — sama
                  persis dengan provider sungguhan.
- ``anthropic`` : Claude via SDK resmi ``anthropic`` (``pip install anthropic``).
- ``openai``    : OpenAI Responses API (``pip install openai``; ``OPENAI_BASE_URL`` opsional), mis.
                  ``gpt-6-luna``: model penalaran (``reasoning.effort``) + *structured outputs* (JSON schema
                  ketat) sehingga keluaran selalu berformat benar; ``store=False`` (respons tidak disimpan
                  di server penyedia).
- ``gemini``    : Google Gemini via SDK ``google-genai`` (``GEMINI_API_KEY``), mis. ``gemini-3.8-flash``;
                  dipakai untuk perbandingan tiga model terhadap instrumen pakar.

Mengganti mock dengan LLM sungguhan = mengubah ``llm.provider``/``LLM_MODEL``; kontrak JSON
(``src.schemas.PenilaianBloom``) tidak berubah. Kunci API dibaca dari ``.env``
oleh SDK masing-masing (KNF-04). Yang dikirim hanya teks kurikulum — tidak ada data mahasiswa.
Parameter yang ditolak model (HTTP 400) dilepas otomatis lalu permintaan diulang; galat jaringan,
429, dan 5xx di-retry oleh SDK (``llm.max_retry_jaringan``, backoff eksponensial). Setiap percobaan HTTP —
termasuk retry diam-diam SDK — dicatat oleh ``transport_tercatat`` ke logger ``crs.llm.http`` (``logs/app.log``).
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable

from src.config import Config

MODEL_DEFAULT = {"mock": "mock-bloom-v1", "anthropic": "claude-opus-5-5", "openai": "gpt-6-luna",
                 "gemini": "gemini-3.8-flash"}
URL_OPENAI = "https://api.openai.com/v1"
LOG_HTTP = logging.getLogger("crs.llm.http")


def modul_http(sdk: ModuleType) -> ModuleType:
    """Pustaka HTTP yang benar-benar dipakai SDK (``httpx2`` pada openai 3.x/anthropic 1.x, ``httpx`` pada versi
    lama) — galat & transport harus dari modul yang sama, bukan diasumsikan ``httpx``."""
    return importlib.import_module(sdk.DefaultHttpxClient.__mro__[1].__module__.split(".")[0])


def galat_http(*sdk: ModuleType) -> tuple[type[BaseException], ...]:
    """Kelas galat transport (putus/timeout di tengah stream) dari pustaka HTTP SDK — tidak dibungkus SDK."""
    return tuple({modul_http(s).HTTPError for s in sdk})


def transport_tercatat(hx: ModuleType, provider: str, dasar=None):
    """Transport pembungkus yang mencatat SETIAP percobaan HTTP ke ``crs.llm.http``: status, durasi sampai header,
    nomor retry SDK (``x-stainless-retry-count``), request-id penyedia, dan ``retry-after``. 429/5xx dan galat
    koneksi yang di-retry diam-diam oleh SDK jadi terlihat tanpa bergantung pada logger internal SDK (yang
    berubah antarversi). ``dasar`` = transport sungguhan (bawaan ``HTTPTransport``; tes memakai ``MockTransport``)."""

    class TransportTercatat(hx.BaseTransport):
        def __init__(self):
            self.dasar = dasar if dasar is not None else hx.HTTPTransport()

        def handle_request(self, request):
            ulang = request.headers.get("x-stainless-retry-count", "0")
            t0 = time.perf_counter()
            try:
                r = self.dasar.handle_request(request)
            except Exception as e:
                LOG_HTTP.warning("%s %s %s → GALAT %s: %s (retry SDK ke-%s, %.1f dtk)", provider, request.method,
                                 request.url.path, type(e).__name__, str(e)[:200], ulang, time.perf_counter() - t0)
                raise
            LOG_HTTP.log(logging.WARNING if r.status_code >= 400 else logging.INFO,
                         "%s %s %s → %d (retry SDK ke-%s, %.1f dtk sampai header, request-id %s, retry-after %s)",
                         provider, request.method, request.url.path, r.status_code, ulang, time.perf_counter() - t0,
                         r.headers.get("x-request-id") or r.headers.get("request-id"), r.headers.get("retry-after"))
            return r

        def close(self):
            self.dasar.close()

    return TransportTercatat()


def http_client_tercatat(sdk: ModuleType, provider: str, dasar=None):
    """``DefaultHttpxClient`` SDK (default batas koneksi & redirect SDK) dengan ``transport_tercatat``."""
    return sdk.DefaultHttpxClient(transport=transport_tercatat(modul_http(sdk), provider, dasar))


class LLMError(Exception):
    """Kegagalan satu panggilan LLM (dihitung sebagai satu percobaan gagal)."""


class LLMKonfigurasiError(Exception):
    """Salah konfigurasi (paket/kunci/model) — dilempar segera, tidak di-retry."""


@dataclass
class LLMResponse:
    teks: str
    model: str
    catatan: list[str] = field(default_factory=list)
    ringkasan_thinking: str = ""      # ringkasan penalaran internal model (bila tersedia)
    temperature: float | None = None  # temperature yang benar-benar dikirim (None = tidak dikirim)
    thinking: bool = False            # penalaran internal model aktif pada permintaan ini
    token_masuk: int | None = None
    token_keluar: int | None = None
    terpotong: bool = False           # keluaran berhenti karena batas token
    id_respons: str | None = None     # ID respons penyedia (tracing)


class LLMClient(ABC):
    """``complete(system, user, skema)``; ``skema`` = JSON schema keluaran (dipakai provider yang mendukung
    *structured outputs*; provider lain cukup mengandalkan instruksi prompt + validasi + retry)."""

    provider: str = ""

    def __init__(self, model: str, cfg: Config):
        self.model = model
        self.cfg = cfg
        self.temperature_dipakai: float | None = cfg.llm.temperature   # None = tidak dikirim/tidak didukung
        self.thinking_dipakai: bool = False
        self.ditolak: set[str] = set()          # parameter opsional yang pernah ditolak model (diingat)

    @property
    def aman_paralel(self) -> bool:
        """Boleh dipanggil dari beberapa thread sekaligus (``llm.konkurensi``)."""
        return True

    @abstractmethod
    def complete(self, system: str, user: str, skema: dict | None = None) -> LLMResponse: ...


# ---------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------


class MockClient(LLMClient):
    """Klien offline deterministik; ``penjawab(system, user) -> teks`` diatur oleh tugas pemanggil."""

    provider = "mock"

    def __init__(self, model: str, cfg: Config):
        super().__init__(model, cfg)
        self.penjawab: Callable[[str, str], str] | None = None
        self.temperature_dipakai = None    # deterministik; parameter sampling tidak relevan

    @property
    def aman_paralel(self) -> bool:
        return False                       # ``penjawab`` diganti per permintaan

    def complete(self, system: str, user: str, skema: dict | None = None) -> LLMResponse:
        if self.penjawab is None:
            raise LLMError("MockClient belum diberi penjawab untuk tugas ini")
        return LLMResponse(teks=self.penjawab(system, user), model=self.model)


# ---------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------


class AnthropicClient(LLMClient):
    """Claude via SDK resmi (Messages API).

    - ``llm.thinking`` → ``thinking: {type: "adaptive", display: ...}`` (chain-of-thought internal);
      ringkasan penalaran disimpan bila ``display = summarized``. ``llm.effort`` → ``output_config.effort``.
    - ``llm.structured_output`` + skema → ``output_config.format`` (JSON schema; didukung Haiku 4.5/5.5, Sonnet,
      Opus ≥ 4.5) sehingga format keluaran setara dengan provider OpenAI. Bila model menolak, dilepas & dicatat.
    - ``temperature`` hanya dikirim bila thinking nonaktif (model Claude terbaru menolak parameter sampling
      selain bawaan, mis. Haiku 5.5: hanya 1). SDK ``anthropic`` 1.x tidak lagi punya argumen ``temperature``,
      sehingga dikirim lewat ``extra_body``.
    - ``fallbacks: "default"`` (server-side) bila ``llm.anthropic_fallback`` DAN modelnya mendukung (Haiku 5.5
      tidak punya *server-side fallback*).
    - Selalu *streaming* (``get_final_message``): ``max_tokens`` batch besar tidak terkena batas waktu HTTP.
    Parameter yang ditolak model (HTTP 400) dilepas, percobaan diulang, dan hal ini dicatat.
    """

    provider = "anthropic"
    MODEL_FALLBACK = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")

    def __init__(self, model: str, cfg: Config):
        super().__init__(model, cfg)
        try:
            import anthropic
        except ImportError as e:
            raise LLMKonfigurasiError("Paket 'anthropic' belum terpasang. Jalankan: pip install anthropic") from e
        self._sdk = anthropic
        try:
            self._client = anthropic.Anthropic(timeout=cfg.llm.timeout_detik, max_retries=cfg.llm.max_retry_jaringan,
                                               http_client=http_client_tercatat(anthropic, "anthropic"))
        except Exception as e:  # kredensial tidak ditemukan, dsb.
            raise LLMKonfigurasiError(f"Klien Anthropic gagal dibuat: {e}. Isi ANTHROPIC_API_KEY di .env") from e

    def _create(self, **kwargs):
        if "temperature" in kwargs:
            kwargs["extra_body"] = {"temperature": kwargs.pop("temperature")}
        with self._client.beta.messages.stream(**kwargs) as aliran:
            return aliran.get_final_message()

    # parameter opsional → (kunci kw, kata kunci pada pesan galat 400)
    _OPSIONAL = (("thinking", "thinking"), ("output_config", "effort"), ("temperature", "temperature"),
                 ("fallbacks", "fallback"))

    def complete(self, system: str, user: str, skema: dict | None = None) -> LLMResponse:
        llm = self.cfg.llm
        kw: dict = {"model": self.model, "max_tokens": llm.max_tokens, "system": system,
                    "messages": [{"role": "user", "content": user}]}
        if llm.thinking and "thinking" not in self.ditolak:
            kw["thinking"] = {"type": "adaptive", "display": llm.thinking_display}
        oc: dict = {}
        if llm.effort and "output_config" not in self.ditolak:
            oc["effort"] = llm.effort
        if skema is not None and llm.structured_output and "format" not in self.ditolak:
            oc["format"] = {"type": "json_schema", "schema": skema}
        if oc:
            kw["output_config"] = oc
        if "thinking" not in kw and "temperature" not in self.ditolak:
            kw["temperature"] = llm.temperature
        if llm.anthropic_fallback and "fallbacks" not in self.ditolak and self.model in self.MODEL_FALLBACK:
            kw["betas"] = ["server-side-fallback-2026-07-01"]
            kw["fallbacks"] = "default"
        catatan: list[str] = []
        for _ in range(len(self._OPSIONAL) + 2):
            try:
                resp = self._create(**kw)
                break
            except self._sdk.BadRequestError as e:
                pesan = str(e).lower()
                oc = kw.get("output_config") or {}
                if "format" in oc and ("format" in pesan or "schema" in pesan):
                    self.ditolak.add("format")                 # skema ditolak → JSON divalidasi Pydantic saja
                    oc.pop("format")
                    if not oc:
                        kw.pop("output_config")
                    catatan.append("output_config.format ditolak model; dikirim tanpa JSON schema")
                    continue
                lepas = next((k for k, kata in self._OPSIONAL if k in kw and kata in pesan), None)
                if lepas is None:
                    raise LLMError(f"permintaan ditolak API: {e}") from e
                self.ditolak.add(lepas)
                if lepas == "output_config" and "format" in oc:  # hanya effort yang ditolak
                    oc.pop("effort", None)
                else:
                    kw.pop(lepas)
                if lepas == "fallbacks":
                    kw.pop("betas", None)
                if lepas == "thinking" and "temperature" not in self.ditolak:
                    kw["temperature"] = llm.temperature      # tanpa thinking: kembali ke sampling deterministik
                catatan.append(f"{lepas} ditolak model; dikirim tanpa {lepas}")
            except self._sdk.AuthenticationError as e:
                raise LLMKonfigurasiError(f"Autentikasi Anthropic gagal: {e}") from e
            except self._sdk.NotFoundError as e:
                raise LLMKonfigurasiError(f"Model '{self.model}' tidak ditemukan: {e}") from e
            except self._sdk.APIError as e:
                raise LLMError(f"galat API: {e}") from e
            except galat_http(self._sdk) as e:      # putus/timeout di tengah stream: tidak dibungkus SDK
                LOG_HTTP.warning("anthropic stream terputus: %s: %s", type(e).__name__, e)
                raise LLMError(f"stream terputus: {type(e).__name__}: {e}") from e
        else:
            raise LLMError("permintaan gagal setelah melepas parameter yang ditolak")
        self.temperature_dipakai = kw.get("temperature")
        self.thinking_dipakai = "thinking" in kw
        if resp.stop_reason == "refusal":
            raise LLMError("model menolak menjawab (stop_reason=refusal)")
        if resp.stop_reason == "max_tokens":
            catatan.append("keluaran terpotong max_tokens")
        teks = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        pikir = "\n".join(t for b in resp.content if getattr(b, "type", "") == "thinking"
                          and (t := getattr(b, "thinking", "")))
        usage = getattr(resp, "usage", None)
        return LLMResponse(teks=teks, model=getattr(resp, "model", self.model), catatan=catatan,
                           ringkasan_thinking=pikir, temperature=kw.get("temperature"), thinking="thinking" in kw,
                           terpotong=resp.stop_reason == "max_tokens", id_respons=getattr(resp, "id", None),
                           token_masuk=getattr(usage, "input_tokens", None),
                           token_keluar=getattr(usage, "output_tokens", None))


# ---------------------------------------------------------------------
# OpenAI (Responses API)
# ---------------------------------------------------------------------


class OpenAIClient(LLMClient):
    """OpenAI Responses API.

    - ``llm.thinking`` + ``llm.effort`` → ``reasoning: {effort}`` (model penalaran, mis. ``gpt-6-luna``);
      ``xhigh``/``max`` diturunkan ke ``high`` bila ditolak.
    - ``skema`` + ``llm.structured_output`` → ``text.format`` JSON schema ketat (``strict``).
    - ``temperature`` hanya dikirim bila penalaran nonaktif (model penalaran menolak parameter sampling).
    - ``store=False``: respons tidak disimpan di server penyedia (KNF-04).
    Endpoint: ``OPENAI_BASE_URL`` (bila kosong/spasi → endpoint resmi; SDK membaca variabel ini sendiri,
    sehingga nilai kosong harus ditimpa eksplisit).
    """

    provider = "openai"
    _OPSIONAL = ("reasoning", "text", "temperature", "max_output_tokens")

    def __init__(self, model: str, cfg: Config):
        super().__init__(model, cfg)
        try:
            import openai
        except ImportError as e:
            raise LLMKonfigurasiError("Paket 'openai' belum terpasang. Jalankan: pip install openai") from e
        self._sdk = openai
        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or URL_OPENAI
        try:
            self._client = openai.OpenAI(base_url=base_url, timeout=cfg.llm.timeout_detik,
                                         max_retries=cfg.llm.max_retry_jaringan,
                                         http_client=http_client_tercatat(openai, "openai"))
        except Exception as e:
            raise LLMKonfigurasiError(f"Klien OpenAI gagal dibuat: {e}. Isi OPENAI_API_KEY di .env") from e

    def _create(self, **kwargs):
        """Selalu streaming: batch besar (puluhan ribu token keluaran + penalaran) melampaui batas waktu HTTP
        non-streaming, dan timeout memicu retry SDK yang ikut ditagih. ``timeout_detik`` berlaku per jeda
        antar-event, bukan total durasi."""
        akhir = None
        aliran = self._client.responses.create(stream=True, **kwargs)
        try:
            for ev in aliran:
                tipe = getattr(ev, "type", "")
                if tipe in ("response.completed", "response.incomplete"):
                    akhir = ev.response
                elif tipe == "response.failed":
                    galat = getattr(ev.response, "error", None)
                    raise LLMError(f"respons gagal: {getattr(galat, 'message', galat)}")
                elif tipe == "error":
                    raise LLMError(f"galat stream: {getattr(ev, 'message', ev)}")
        except (*galat_http(self._sdk), self._sdk.APIError) as e:   # putus/timeout/galat di tengah stream
            LOG_HTTP.warning("openai stream terputus: %s: %s", type(e).__name__, e)
            raise LLMError(f"stream terputus: {type(e).__name__}: {e}") from e
        if akhir is None:
            raise LLMError("stream berakhir tanpa respons final")
        return akhir

    @staticmethod
    def _param_ditolak(e: Exception, kw: dict) -> str | None:
        """Parameter opsional yang disebut galat 400 (``param`` mis. ``reasoning.effort``, ``text.format``)."""
        param = str(getattr(e, "param", None) or (getattr(e, "body", None) or {}).get("param") or "")
        pesan = f"{param} {e}".lower()
        return next((k for k in OpenAIClient._OPSIONAL if k in kw and (param.startswith(k) or f"'{k}" in pesan)),
                    None)

    def complete(self, system: str, user: str, skema: dict | None = None) -> LLMResponse:
        llm = self.cfg.llm
        kw: dict = {"model": self.model, "instructions": system, "input": user,
                    "max_output_tokens": llm.max_tokens, "store": False}
        if llm.thinking and llm.effort and "reasoning" not in self.ditolak:
            kw["reasoning"] = {"effort": "high" if "effort_turun" in self.ditolak else llm.effort}
        if skema is not None and llm.structured_output and "text" not in self.ditolak:
            kw["text"] = {"format": {"type": "json_schema", "name": "keluaran", "schema": skema, "strict": True}}
        if "reasoning" not in kw and "temperature" not in self.ditolak:
            kw["temperature"] = llm.temperature
        catatan: list[str] = []
        for _ in range(len(self._OPSIONAL) + 2):
            try:
                resp = self._create(**kw)
                break
            except self._sdk.BadRequestError as e:
                lepas = self._param_ditolak(e, kw)
                if lepas is None:
                    raise LLMError(f"permintaan ditolak API: {e}") from e
                if lepas == "reasoning" and kw["reasoning"]["effort"] in ("xhigh", "max"):
                    self.ditolak.add("effort_turun")         # coba effort tertinggi yang umum dulu
                    kw["reasoning"] = {"effort": "high"}
                    catatan.append(f"effort {llm.effort} ditolak model; diturunkan ke high")
                    continue
                self.ditolak.add(lepas)
                kw.pop(lepas)
                if lepas == "reasoning" and "temperature" not in self.ditolak:
                    kw["temperature"] = llm.temperature
                catatan.append(f"{lepas} ditolak model; dikirim tanpa {lepas}")
            except (self._sdk.AuthenticationError, self._sdk.PermissionDeniedError) as e:
                raise LLMKonfigurasiError(f"Autentikasi OpenAI gagal (periksa OPENAI_API_KEY): {e}") from e
            except self._sdk.NotFoundError as e:
                raise LLMKonfigurasiError(f"Model '{self.model}' tidak ditemukan: {e}") from e
            except self._sdk.APIError as e:          # jaringan/429/5xx setelah retry SDK habis
                raise LLMError(f"galat API: {e}") from e
        else:
            raise LLMError("permintaan gagal setelah melepas parameter yang ditolak")
        self.temperature_dipakai = kw.get("temperature")
        self.thinking_dipakai = "reasoning" in kw
        penolakan = [c.refusal for o in (resp.output or []) if getattr(o, "type", "") == "message"
                     for c in (getattr(o, "content", None) or []) if getattr(c, "type", "") == "refusal"]
        if penolakan:
            raise LLMError(f"model menolak menjawab: {penolakan[0][:200]}")
        terpotong = False
        if getattr(resp, "status", "completed") == "incomplete":
            alasan = getattr(getattr(resp, "incomplete_details", None), "reason", "") or "tidak diketahui"
            catatan.append(f"keluaran tidak lengkap ({alasan})")
            terpotong = "token" in alasan
        if "text" in kw:
            catatan.append("structured output (JSON schema ketat)")
        usage = getattr(resp, "usage", None)
        return LLMResponse(teks=resp.output_text or "", model=getattr(resp, "model", None) or self.model,
                           catatan=catatan, temperature=kw.get("temperature"), thinking="reasoning" in kw, terpotong=terpotong,
                           id_respons=getattr(resp, "id", None),
                           token_masuk=getattr(usage, "input_tokens", None),
                           token_keluar=getattr(usage, "output_tokens", None))


# ---------------------------------------------------------------------
# Google Gemini (google-genai)
# ---------------------------------------------------------------------


class GeminiClient(LLMClient):
    """Gemini via SDK resmi ``google-genai`` (``pip install google-genai``; kunci ``GEMINI_API_KEY``).

    - ``llm.thinking`` + ``llm.effort`` → ``thinking_config.thinking_level`` (low/medium/high; xhigh/max → high).
    - ``skema`` + ``llm.structured_output`` → ``response_mime_type=application/json`` + ``response_json_schema``.
    - ``temperature`` hanya dikirim bila penalaran nonaktif (setara provider lain).
    Parameter yang ditolak (HTTP 400) dilepas, permintaan diulang, dan dicatat (instruksi &
    skema identik dengan provider lain).
    """

    provider = "gemini"
    _OPSIONAL = ("thinking_config", "response_json_schema", "temperature")
    _LEVEL = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "HIGH", "max": "HIGH"}

    def __init__(self, model: str, cfg: Config):
        super().__init__(model, cfg)
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as e:
            raise LLMKonfigurasiError("Paket 'google-genai' belum terpasang. Jalankan: pip install google-genai") from e
        self._types, self._errors = types, errors
        try:
            self._client = genai.Client(http_options=types.HttpOptions(
                timeout=int(cfg.llm.timeout_detik * 1000),
                retry_options=types.HttpRetryOptions(attempts=cfg.llm.max_retry_jaringan + 1)))
        except Exception as e:
            raise LLMKonfigurasiError(f"Klien Gemini gagal dibuat: {e}. Isi GEMINI_API_KEY di .env") from e

    def _create(self, **kwargs):
        return self._client.models.generate_content(
            model=self.model, contents=kwargs.pop("contents"), config=self._types.GenerateContentConfig(**kwargs))

    def complete(self, system: str, user: str, skema: dict | None = None) -> LLMResponse:
        llm = self.cfg.llm
        kw: dict = {"contents": user, "system_instruction": system, "max_output_tokens": llm.max_tokens}
        if llm.thinking and "thinking_config" not in self.ditolak:
            kw["thinking_config"] = self._types.ThinkingConfig(
                thinking_level=self._LEVEL.get(llm.effort or "high", "HIGH"),
                include_thoughts=llm.thinking_display == "summarized")
        if skema is not None and llm.structured_output and "response_json_schema" not in self.ditolak:
            kw.update(response_mime_type="application/json", response_json_schema=skema)
        if "thinking_config" not in kw and "temperature" not in self.ditolak:
            kw["temperature"] = llm.temperature
        catatan: list[str] = []
        for _ in range(len(self._OPSIONAL) + 2):
            try:
                resp = self._create(**dict(kw))
                break
            except self._errors.APIError as e:
                kode = getattr(e, "code", None)
                if kode in (401, 403):
                    raise LLMKonfigurasiError(f"Autentikasi Gemini gagal (periksa GEMINI_API_KEY): {e}") from e
                if kode == 404:
                    raise LLMKonfigurasiError(f"Model '{self.model}' tidak ditemukan: {e}") from e
                pesan = str(e).lower()
                if kode == 429 and "quota" in pesan:     # kuota/billing habis: mengulang tidak akan berhasil
                    raise LLMKonfigurasiError(f"Kuota Gemini habis (periksa billing/kuota GEMINI_API_KEY): "
                                              f"{str(e)[:200]}") from e
                lepas = next((k for k in self._OPSIONAL if k in kw and (k in pesan or k.split("_")[0] in pesan)),
                             None) if kode == 400 else None
                if lepas is None:
                    raise LLMError(f"galat API: {e}") from e
                self.ditolak.add(lepas)
                kw.pop(lepas)
                if lepas == "response_json_schema":
                    kw.pop("response_mime_type", None)
                if lepas == "thinking_config" and "temperature" not in self.ditolak:
                    kw["temperature"] = llm.temperature
                catatan.append(f"{lepas} ditolak model; dikirim tanpa {lepas}")
        else:
            raise LLMError("permintaan gagal setelah melepas parameter yang ditolak")
        self.temperature_dipakai = kw.get("temperature")
        self.thinking_dipakai = "thinking_config" in kw
        kandidat = (getattr(resp, "candidates", None) or [None])[0]
        alasan = str(getattr(kandidat, "finish_reason", "") or "")
        if any(x in alasan for x in ("SAFETY", "PROHIBITED", "BLOCKLIST")):
            raise LLMError(f"model menolak menjawab (finish_reason={alasan})")
        bagian = getattr(getattr(kandidat, "content", None), "parts", None) or []
        teks = "".join(p.text for p in bagian if getattr(p, "text", None) and not getattr(p, "thought", False))
        pikir = "\n".join(p.text for p in bagian if getattr(p, "text", None) and getattr(p, "thought", False))
        terpotong = "MAX_TOKENS" in alasan
        if terpotong:
            catatan.append("keluaran terpotong max_output_tokens")
        if "response_json_schema" in kw:
            catatan.append("structured output (response_json_schema)")
        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(teks=teks, model=getattr(resp, "model_version", None) or self.model, catatan=catatan,
                           ringkasan_thinking=pikir, temperature=kw.get("temperature"), thinking=self.thinking_dipakai,
                           terpotong=terpotong, token_masuk=getattr(usage, "prompt_token_count", None),
                           id_respons=getattr(resp, "response_id", None),
                           token_keluar=(getattr(usage, "candidates_token_count", 0) or 0)
                           + (getattr(usage, "thoughts_token_count", 0) or 0) if usage else None)


def buat_client(cfg: Config, provider: str | None = None, model: str | None = None) -> LLMClient:
    provider = (provider or cfg.llm.provider).lower()
    model = model or cfg.llm.model
    if provider == "mock":
        model = model if model.startswith("mock") else MODEL_DEFAULT["mock"]
    elif not model or model.startswith("mock"):
        model = MODEL_DEFAULT.get(provider)
        if not model:
            raise LLMKonfigurasiError(f"Isi LLM_MODEL di .env untuk provider '{provider}'")
    kelas = {"mock": MockClient, "anthropic": AnthropicClient, "openai": OpenAIClient,
             "gemini": GeminiClient}.get(provider)
    if kelas is None:
        raise LLMKonfigurasiError(f"Provider tidak dikenal: {provider}")
    return kelas(model, cfg)


# ---------------------------------------------------------------------
# Parsing keluaran
# ---------------------------------------------------------------------


def parse_json_keluaran(teks: str) -> dict:
    """Ambil objek JSON terakhir dari keluaran (blok ```json``` atau {...} bebas)."""
    if not teks or not teks.strip():
        raise ValueError("keluaran kosong")
    for blok in reversed(re.findall(r"```(?:json)?\s*(.*?)```", teks, flags=re.S)):
        try:
            obj = json.loads(blok)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # tanpa blok kode: ambil objek JSON tingkat-atas terakhir di teks
    dekoder = json.JSONDecoder()
    terakhir, i = None, 0
    while (i := teks.find("{", i)) != -1:
        try:
            obj, akhir = dekoder.raw_decode(teks, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            terakhir = obj
        i = akhir
    if terakhir is None:
        raise ValueError("tidak ada objek JSON yang valid")
    return terakhir
