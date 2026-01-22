# compressors_v2.py
from __future__ import annotations

import gzip
import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from utils_bits_v2 import MagicInfo, detect_magic, safe_basename, write_bytes, zlib_wrap


# ----------------------------
# ZIP helpers (Mode1/Mode2)
# ----------------------------

def zip_single_file(input_path: str, level: int = 6) -> Tuple[bytes, Dict[str, Any]]:
    """
    ZIP a single file (DEFLATED). This is self-describing (filename + CRC inside ZIP).
    """
    base = safe_basename(os.path.basename(input_path))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
        zf.write(input_path, arcname=base)
    meta = {"kind": "zip_single_file", "compression": "deflated", "level": level, "filename": base}
    return buf.getvalue(), meta


def zip_store_single_file(input_path: str) -> Tuple[bytes, Dict[str, Any]]:
    """
    ZIP a single file with STORED (no compression). Still self-describing (keeps extension).
    """
    base = safe_basename(os.path.basename(input_path))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(input_path, arcname=base)
    meta = {"kind": "zip_store_single_file", "compression": "stored", "filename": base}
    return buf.getvalue(), meta


def unzip_single_file(zip_bytes: bytes, out_dir: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract the first non-directory member from ZIP bytes.
    If there are multiple members, extracts the first file member.
    """
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        if not members:
            raise ValueError("ZIP has no file members")
        name = members[0]
        safe = safe_basename(name, fallback="unzipped.bin")
        dst = os.path.join(out_dir, safe)
        with zf.open(name, "r") as src, open(dst, "wb") as f:
            shutil.copyfileobj(src, f)
        return dst, {"kind": "unzip_single_file", "member": name, "out": dst}


# ----------------------------
# Domain detection
# ----------------------------

def _looks_text(b: bytes) -> bool:
    if not b:
        return False
    if b.count(b"\x00") > 0:
        return False
    try:
        s = b[:4096].decode("utf-8")
        printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in s)
        return (printable / max(1, len(s))) > 0.95
    except Exception:
        return False


def detect_domain(input_path: str, raw_bytes: bytes) -> str:
    """
    Returns one of: image | audio | video | text | other
    """
    m = detect_magic(raw_bytes)
    if m:
        if m.kind in {"png","jpeg","webp","gif","bmp","tiff"}:
            return "image"
        if m.kind in {"wav","mp3","flac","opus_ogg","ogg"}:
            return "audio"
        if m.kind in {"mp4","avi","mkv_webm"}:
            return "video"
        if m.kind in {"pdf","docx","pptx","xlsx","epub","zip","gzip"}:
            return "other"
        if m.kind == "text":
            return "text"

    ext = os.path.splitext(input_path)[1].lower()
    if ext in {".png",".jpg",".jpeg",".webp",".gif",".bmp",".tif",".tiff"}:
        return "image"
    if ext in {".wav",".mp3",".flac",".ogg",".opus",".m4a",".aac"}:
        return "audio"
    if ext in {".mp4",".mov",".mkv",".avi",".webm"}:
        return "video"
    if ext in {".txt",".md",".json",".csv",".tsv",".log",".xml",".yaml",".yml"}:
        return "text"

    if _looks_text(raw_bytes):
        return "text"
    return "other"


# ----------------------------
# Representation encoding
# ----------------------------

@dataclass
class RepresentationResult:
    rep_bytes: bytes
    rep_meta: Dict[str, Any]


def _encode_gzip(raw: bytes) -> bytes:
    return gzip.compress(raw)


def _decode_gzip(gz: bytes) -> bytes:
    return gzip.decompress(gz)


def _run_ffmpeg(args: list[str]) -> None:
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + p.stderr.decode("utf-8", errors="ignore")[:2000])


def _encode_image_webp(raw: bytes, quality: int, lossless: bool, allow_external_ffmpeg: bool) -> bytes:
    # Try PIL first
    try:
        from PIL import Image  # type: ignore
        img = Image.open(io.BytesIO(raw))
        out = io.BytesIO()
        img.save(out, format="WEBP", quality=int(quality), lossless=bool(lossless))
        return out.getvalue()
    except Exception:
        if not allow_external_ffmpeg:
            raise
    # Fallback: ffmpeg -> webp
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in.bin")
        outp = os.path.join(td, "out.webp")
        with open(inp, "wb") as f:
            f.write(raw)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", inp]
        if lossless:
            cmd += ["-lossless", "1"]
        else:
            cmd += ["-q:v", str(max(0, min(100, int(quality))))]
        cmd += [outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def _encode_audio_opus(input_path: str, allow_external_ffmpeg: bool, bitrate_kbps: int = 64) -> bytes:
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode Opus.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.ogg")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:a", "libopus", "-b:a", f"{int(bitrate_kbps)}k", outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def _encode_audio_flac(input_path: str, allow_external_ffmpeg: bool) -> bytes:
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode FLAC.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.flac")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:a", "flac", outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def _encode_video_h264_mp4(input_path: str, allow_external_ffmpeg: bool, crf: int = 28, preset: str = "medium") -> bytes:
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode MP4.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.mp4")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
               "-pix_fmt", "yuv420p",
               "-movflags", "+faststart",
               "-c:a", "aac", "-b:a", "128k",
               outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()

def _encode_video_vp9_webm(input_path: str, allow_external_ffmpeg: bool, crf: int = 32, speed: int = 4) -> bytes:
    """VP9 in WebM container. Good compression; slower than H.264."""
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode WebM/VP9.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.webm")
        # libvpx-vp9 CRF uses 0-63 (lower is better quality). We keep audio as Opus.
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:v", "libvpx-vp9", "-crf", str(int(crf)), "-b:v", "0", "-row-mt", "1",
               "-cpu-used", str(int(speed)),
               "-pix_fmt", "yuv420p",
               "-c:a", "libopus", "-b:a", "96k",
               outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def _encode_video_av1_mkv(input_path: str, allow_external_ffmpeg: bool, crf: int = 35, preset: int = 8) -> bytes:
    """AV1 via libaom-av1 in Matroska container. Best compression, but slow."""
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode AV1.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.mkv")
        # libaom-av1 uses crf 0-63, preset 0(slowest/best)-13(fastest/worst).
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:v", "libaom-av1", "-crf", str(int(crf)), "-b:v", "0",
               "-cpu-used", str(int(preset)),
               "-pix_fmt", "yuv420p",
               "-c:a", "libopus", "-b:a", "96k",
               outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()



def domain_detect_and_encode_rep(
    input_path: str,
    raw_bytes: bytes,
    *,
    image_policy: str = "webp_lossy",   # webp_lossy | webp_lossless | png_lossless | keep
    webp_quality: int = 80,
    text_policy: str = "gzip",          # gzip | keep
    allow_external_ffmpeg: bool = False,
    zlib_policy: str = "auto",
    audio_policy: str = "opus_ogg",     # opus_ogg | flac_lossless | keep
    opus_bitrate_kbps: int = 64,
    video_policy: str = "mp4_h264",     # mp4_h264 | webm_vp9 | mkv_av1 | keep
    video_crf: int = 28,
) -> RepresentationResult:
    """
    Produce self-describing bytes with clear magic (no custom header required).
    For lossy: output format may change (e.g., WAV->OGG Opus, PNG->WebP).
    """
    dom = detect_domain(input_path, raw_bytes)

    # --- Image ---
    if dom == "image":
        if image_policy == "keep":
            return RepresentationResult(raw_bytes, {"domain": "image", "policy": "keep", "lossy": False})
        if image_policy == "png_lossless":
            # If already PNG, keep; else attempt to re-encode PNG (lossless but may be larger)
            try:
                from PIL import Image  # type: ignore
                img = Image.open(io.BytesIO(raw_bytes))
                out = io.BytesIO()
                img.save(out, format="PNG", optimize=True)
                return RepresentationResult(out.getvalue(), {"domain": "image", "policy": "png_lossless", "lossy": False})
            except Exception:
                return RepresentationResult(raw_bytes, {"domain": "image", "policy": "png_lossless_fallback_keep", "lossy": False})
        if image_policy == "webp_lossless":
            rep = _encode_image_webp(raw_bytes, quality=100, lossless=True, allow_external_ffmpeg=allow_external_ffmpeg)
            return RepresentationResult(rep, {"domain": "image", "policy": "webp_lossless", "lossy": False})
        # default lossy webp
        rep = _encode_image_webp(raw_bytes, quality=webp_quality, lossless=False, allow_external_ffmpeg=allow_external_ffmpeg)
        return RepresentationResult(rep, {"domain": "image", "policy": "webp_lossy", "lossy": True, "webp_quality": int(webp_quality)})

    # --- Text ---
    if dom == "text":
        if text_policy == "keep":
            return RepresentationResult(raw_bytes, {"domain": "text", "policy": "keep", "lossy": False})
        rep = _encode_gzip(raw_bytes)
        return RepresentationResult(rep, {"domain": "text", "policy": "gzip", "lossy": False})

    # --- Audio ---
    if dom == "audio":
        if audio_policy == "keep" or not allow_external_ffmpeg:
            return RepresentationResult(raw_bytes, {"domain": "audio", "policy": "keep", "lossy": False, "note": "ffmpeg_disabled_or_keep"})
        if audio_policy == "flac_lossless":
            rep = _encode_audio_flac(input_path, allow_external_ffmpeg=True)
            return RepresentationResult(rep, {"domain": "audio", "policy": "flac_lossless", "lossy": False})
        rep = _encode_audio_opus(input_path, allow_external_ffmpeg=True, bitrate_kbps=opus_bitrate_kbps)
        return RepresentationResult(rep, {"domain": "audio", "policy": "opus_ogg", "lossy": True, "opus_bitrate_kbps": int(opus_bitrate_kbps)})

    # --- Video ---
    if dom == "video":
        if video_policy == "keep" or not allow_external_ffmpeg:
            return RepresentationResult(raw_bytes, {"domain": "video", "policy": "keep", "lossy": False, "note": "ffmpeg_disabled_or_keep"})

        if video_policy == "mp4_h264":
            rep = _encode_video_h264_mp4(input_path, allow_external_ffmpeg=True, crf=video_crf)
            return RepresentationResult(rep, {"domain": "video", "policy": "mp4_h264", "lossy": True, "crf": int(video_crf)})

        if video_policy == "webm_vp9":
            rep = _encode_video_vp9_webm(input_path, allow_external_ffmpeg=True, crf=video_crf)
            return RepresentationResult(rep, {"domain": "video", "policy": "webm_vp9", "lossy": True, "crf": int(video_crf)})

        if video_policy == "mkv_av1":
            rep = _encode_video_av1_mkv(input_path, allow_external_ffmpeg=True, crf=video_crf)
            return RepresentationResult(rep, {"domain": "video", "policy": "mkv_av1", "lossy": True, "crf": int(video_crf)})

        raise ValueError(f"Unknown video_policy: {video_policy}")

    # --- Other ---
    return RepresentationResult(raw_bytes, {"domain": "other", "policy": "keep", "lossy": False})


# ----------------------------
# Restore representation (headerless)

# ----------------------------
# Benchmark candidate generation (Mode 3 Best)
# ----------------------------

def _encode_image_png(raw: bytes) -> bytes:
    """Lossless PNG re-encode (self-describing magic)."""
    from PIL import Image  # type: ignore
    img = Image.open(io.BytesIO(raw))
    out = io.BytesIO()
    # Convert to a standard mode to avoid palette edge cases
    if img.mode not in ("RGB","RGBA","L"):
        img = img.convert("RGBA")
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _encode_image_jpeg(raw: bytes, quality: int) -> bytes:
    """Lossy JPEG re-encode (self-describing magic)."""
    from PIL import Image  # type: ignore
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    out = io.BytesIO()
    # subsampling=0 improves quality but can increase size; let PIL decide default
    img.save(out, format="JPEG", quality=int(quality), optimize=True)
    return out.getvalue()


def _encode_audio_mp3(input_path: str, *, allow_external_ffmpeg: bool, bitrate_kbps: int) -> bytes:
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode MP3.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.mp3")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:a", "libmp3lame", "-b:a", f"{int(bitrate_kbps)}k", outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def _encode_audio_aac_m4a(input_path: str, *, allow_external_ffmpeg: bool, bitrate_kbps: int) -> bytes:
    """AAC in M4A container (MP4 magic 'ftyp')."""
    if not allow_external_ffmpeg:
        raise RuntimeError("ffmpeg not allowed; cannot encode AAC/M4A.")
    with tempfile.TemporaryDirectory() as td:
        outp = os.path.join(td, "out.m4a")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
               "-c:a", "aac", "-b:a", f"{int(bitrate_kbps)}k", outp]
        _run_ffmpeg(cmd)
        return open(outp, "rb").read()


def benchmark_domain_encode_rep(
    input_path: str,
    raw_bytes: bytes,
    *,
    quality_mode: str = "Lossy",  # "Lossless" | "Lossy"
    allow_external_ffmpeg: bool = False,
    zlib_policy: str = "auto",
    image_webp_qualities: Tuple[int, ...] = (60, 80, 90),
    image_jpeg_qualities: Tuple[int, ...] = (60, 80, 90),
    opus_bitrates_kbps: Tuple[int, ...] = (32, 64, 96),
    mp3_bitrates_kbps: Tuple[int, ...] = (96, 128, 192),
    aac_bitrates_kbps: Tuple[int, ...] = (96, 128),
    video_crfs: Tuple[int, ...] = (18, 23, 28),
    video_vp9_crfs: Tuple[int, ...] = (28, 32, 36),
    video_av1_crfs: Tuple[int, ...] = (30, 35, 40),
) -> Tuple[RepresentationResult, Dict[str, Any]]:
    """
    Generate multiple representation candidates for the detected domain and return the best one
    (smallest rep size). This does NOT apply zlib framing; the caller should frame via zlib_wrap.

    Returns:
      (best_rep, bench_meta) where bench_meta includes:
        - detected_domain
        - quality_mode
        - chosen_candidate
        - candidates: list of {name, rep_size_bytes, lossy, policy, ...}
    """
    dom = detect_domain(input_path, raw_bytes)
    qmode = str(quality_mode or "Lossy")
    qmode_norm = "Lossless" if qmode.lower().startswith("lossless") else "Lossy"

    candidates: list[Tuple[str, bytes, Dict[str, Any]]] = []

    def _try_add(name: str, fn, meta: Dict[str, Any]):
        try:
            b = fn()
            # Ensure we only keep self-describing outputs (magic detectable)
            m = detect_magic(b)
            if not m:
                return
            candidates.append((name, b, meta))
        except Exception:
            return

    # Helper: allow keep only if magic detectable
    m0 = detect_magic(raw_bytes)
    if m0:
        candidates.append(("keep", raw_bytes, {"domain": dom, "policy": "keep", "lossy": False, "magic": m0.kind}))

    # --- Text: prefer gzip (magic), sweep levels ---
    if dom == "text":
        for lvl in (1, 6, 9):
            _try_add(
                f"gzip_lvl{lvl}",
                lambda lvl=lvl: gzip.compress(raw_bytes, compresslevel=int(lvl)),
                {"domain": dom, "policy": "gzip", "lossy": False, "gzip_level": int(lvl)},
            )

    # --- Image ---
    elif dom == "image":
        # Lossless candidates
        _try_add("png_lossless", lambda: _encode_image_png(raw_bytes), {"domain": dom, "policy": "png_lossless", "lossy": False})
        _try_add("webp_lossless", lambda: _encode_image_webp(raw_bytes, quality=100, lossless=True, allow_external_ffmpeg=allow_external_ffmpeg),
                 {"domain": dom, "policy": "webp_lossless", "lossy": False})

        if qmode_norm == "Lossy":
            for q in image_webp_qualities:
                _try_add(f"webp_q{int(q)}", lambda q=q: _encode_image_webp(raw_bytes, quality=int(q), lossless=False, allow_external_ffmpeg=allow_external_ffmpeg),
                         {"domain": dom, "policy": "webp_lossy", "lossy": True, "webp_quality": int(q)})
            for q in image_jpeg_qualities:
                _try_add(f"jpeg_q{int(q)}", lambda q=q: _encode_image_jpeg(raw_bytes, quality=int(q)),
                         {"domain": dom, "policy": "jpeg_lossy", "lossy": True, "jpeg_quality": int(q)})

    # --- Audio ---
    elif dom == "audio":
        # Lossless: FLAC (signal-lossless)
        _try_add("flac_lossless", lambda: _encode_audio_flac(input_path, allow_external_ffmpeg=allow_external_ffmpeg),
                 {"domain": dom, "policy": "flac_lossless", "lossy": False})
        if qmode_norm == "Lossy":
            for br in opus_bitrates_kbps:
                _try_add(f"opus_{int(br)}k", lambda br=br: _encode_audio_opus(input_path, allow_external_ffmpeg=allow_external_ffmpeg, bitrate_kbps=int(br)),
                         {"domain": dom, "policy": "opus_ogg", "lossy": True, "opus_bitrate_kbps": int(br)})
            for br in mp3_bitrates_kbps:
                _try_add(f"mp3_{int(br)}k", lambda br=br: _encode_audio_mp3(input_path, allow_external_ffmpeg=allow_external_ffmpeg, bitrate_kbps=int(br)),
                         {"domain": dom, "policy": "mp3", "lossy": True, "mp3_bitrate_kbps": int(br)})
            for br in aac_bitrates_kbps:
                _try_add(f"aac_{int(br)}k", lambda br=br: _encode_audio_aac_m4a(input_path, allow_external_ffmpeg=allow_external_ffmpeg, bitrate_kbps=int(br)),
                         {"domain": dom, "policy": "aac_m4a", "lossy": True, "aac_bitrate_kbps": int(br)})

    # --- Video ---
    elif dom == "video":
        # Keep is already included if magic detectable
        if qmode_norm == "Lossy":
            for crf in video_crfs:
                _try_add(
                    f"h264_crf{int(crf)}",
                    lambda crf=crf: _encode_video_h264_mp4(input_path, allow_external_ffmpeg=allow_external_ffmpeg, crf=int(crf)),
                    {"domain": dom, "policy": "mp4_h264", "lossy": True, "crf": int(crf)}
                )

            for crf in video_vp9_crfs:
                _try_add(
                    f"vp9_crf{int(crf)}",
                    lambda crf=crf: _encode_video_vp9_webm(input_path, allow_external_ffmpeg=allow_external_ffmpeg, crf=int(crf)),
                    {"domain": dom, "policy": "webm_vp9", "lossy": True, "crf": int(crf)}
                )

            for crf in video_av1_crfs:
                _try_add(
                    f"av1_crf{int(crf)}",
                    lambda crf=crf: _encode_video_av1_mkv(input_path, allow_external_ffmpeg=allow_external_ffmpeg, crf=int(crf)),
                    {"domain": dom, "policy": "mkv_av1", "lossy": True, "crf": int(crf)}
                )

    # --- Other ---
    else:
        # Let caller fallback to Mode-2 ZIP STORE
        candidates = []

    if not candidates:
        raise RuntimeError(f"No valid candidates for domain={dom}. (ffmpeg disabled? unsupported input?)")

    # Pick smallest *zlib-framed* size (this is the true size right before DNA)
    scored: list[Tuple[int, int, str, bytes, Dict[str, Any]]] = []
    for (n, b, meta) in candidates:
        try:
            z = zlib_wrap(b, policy=str(zlib_policy or "auto"))
            scored.append((len(z), len(b), n, b, meta))
        except Exception:
            continue
    if not scored:
        raise RuntimeError(f"No candidates could be framed by zlib (policy={zlib_policy}).")
    scored.sort(key=lambda t: (t[0], t[1]))  # (zlib_size, rep_size)
    best_zlib_size, best_rep_size, best_name, best_bytes, best_meta = scored[0]

    # Sort candidates for reporting
    candidates_sorted = []
    for (zsz, rsz, n, b, meta) in scored:
        candidates_sorted.append((n, b, meta, zsz, rsz))
    candidates_sorted.sort(key=lambda t: (t[3], t[4]))
    bench_meta = {
        "detected_domain": dom,
        "quality_mode": qmode_norm,
        "chosen_candidate": best_name,
        "candidates": [
            {"name": n, "rep_size_bytes": int(rsz), "zlib_size_bytes": int(zsz), **(meta or {})}
            for (n, b, meta, zsz, rsz) in candidates_sorted
        ],
    }

    # Ensure required keys for UI
    best_meta = dict(best_meta or {})
    best_meta.setdefault("domain", dom)
    best_meta.setdefault("lossy", bool(best_meta.get("lossy", False)))
    best_meta["chosen_candidate"] = best_name
    best_meta["quality_mode"] = qmode_norm
    best_meta["candidates"] = bench_meta["candidates"]

    return RepresentationResult(best_bytes, best_meta), bench_meta

# ----------------------------

def _write_with_magic(data: bytes, out_dir: str, stem: str) -> Tuple[str, Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    m = detect_magic(data)
    ext = m.ext if m else ".bin"
    out_path = os.path.join(out_dir, safe_basename(stem + ext, fallback=stem + ".bin"))
    write_bytes(out_path, data)
    return out_path, {"detected_magic": (m.kind if m else None), "ext": ext, "restore_kind": "write_bytes"}


def restore_rep(inner_bytes: bytes, out_dir: str, preferred_stem: str = "restored") -> Tuple[str, Dict[str, Any]]:
    """
    Convert inner bytes -> a concrete file on disk, using only standard magic signatures.
    No custom header is required.
    """
    m = detect_magic(inner_bytes)

    # gzip => gunzip then write result (or route again)
    if m and m.kind == "gzip":
        payload = _decode_gzip(inner_bytes)
        outp, meta = _write_with_magic(payload, out_dir, preferred_stem)
        meta.update({"input_magic": "gzip", "restore_kind": "gunzip_then_write"})
        return outp, meta

    # OOXML containers: do NOT unzip, write as docx/pptx/xlsx
    if m and m.kind in {"docx","pptx","xlsx","epub"}:
        outp, meta = _write_with_magic(inner_bytes, out_dir, preferred_stem)
        meta.update({"restore_kind": "write_container"})
        return outp, meta

    # Generic ZIP: if it is a single-file archive, extract; else write as .zip
    if m and m.kind == "zip":
        try:
            with zipfile.ZipFile(io.BytesIO(inner_bytes), "r") as zf:
                members = [n for n in zf.namelist() if not n.endswith("/")]
                if len(members) == 1:
                    extracted, meta = unzip_single_file(inner_bytes, out_dir=out_dir)
                    meta.update({"detected_magic": "zip", "restore_kind": "unzip_single_file"})
                    return extracted, meta
        except Exception:
            pass
        outp, meta = _write_with_magic(inner_bytes, out_dir, preferred_stem)
        meta.update({"restore_kind": "write_zip_bytes"})
        return outp, meta

    # Everything else: just write based on magic
    return _write_with_magic(inner_bytes, out_dir, preferred_stem)
