#!/usr/bin/env python3
from __future__ import annotations

import bz2
import gzip
import io
import json
import lzma
import math
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dna_codec
from utils_bits_v2 import (
    MagicInfo,
    bitstring_to_bytes,
    bytes_to_bitstring,
    detect_magic,
    ensure_dir as _ensure_dir,
    read_bytes as _read_bytes,
    safe_basename,
    sha256_bytes as _sha256_bytes,
    write_bytes as _write_bytes,
    write_text as _write_text,
)

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import numpy as np
except Exception:
    np = None

ENGINE_VERSION = "hierarchical_headerless_self_describing_v1"

DNA_SCHEMES = ("RINF_B16", "R2_B15", "R1_B12", "R0_B9")
INIT_DIMERS = ("TA", "AT", "CG", "GC")
FORBIDDEN_MOTIFS = ("GAATTC", "GGATCC", "AAGCTT", "GCGGCCGC", "TTTTTT", "AAAAAA", "CCCCCC", "GGGGGG")


# ============================================================
# Public IO helpers expected by the UI
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_bytes(path: str | Path) -> bytes:
    return _read_bytes(str(path))


def write_bytes(path: str | Path, data: bytes) -> None:
    _write_bytes(str(path), data)


def write_text(path: str | Path, text: str) -> None:
    _write_text(str(path), text)


def sha256_bytes(data: bytes) -> str:
    return _sha256_bytes(data)


def now_utc_string() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def human_bytes(n: int) -> str:
    x = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0 or u == "TB":
            return f"{x:.2f} {u}" if u != "B" else f"{int(x)} B"
        x /= 1024.0
    return f"{x:.2f} TB"


def safe_stem(name: str, fallback: str = "restored") -> str:
    stem = Path(name).stem or fallback
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or fallback


def safe_suffix(name: str, fallback: str = ".bin") -> str:
    suf = Path(name).suffix.lower()
    if suf and len(suf) <= 16 and re.fullmatch(r"\.[A-Za-z0-9]+", suf):
        return suf
    return fallback


# ============================================================
# Input profiling and domain detection
# ============================================================

@dataclass
class InputProfile:
    domain: str
    subtype: str
    magic: Optional[MagicInfo]
    input_size_bytes: int
    entropy_estimate: float
    text_likeness_value: float
    already_compressed_likelihood_value: float
    profiler_confidence: float
    image_handling_mode: str = "content_preserving"


def estimate_entropy(data: bytes, sample_limit: int = 8192) -> float:
    sample = data[:sample_limit]
    if not sample:
        return 0.0
    counts = [0] * 256
    for b in sample:
        counts[b] += 1
    ent = 0.0
    total = len(sample)
    for c in counts:
        if c:
            p = c / total
            ent -= p * math.log2(p)
    return float(ent)


def text_likeness(data: bytes) -> float:
    if not data:
        return 0.0
    if b"\x00" in data[:4096]:
        return 0.0
    try:
        s = data[:4096].decode("utf-8")
        printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in s)
        return printable / max(1, len(s))
    except Exception:
        return 0.0


def already_compressed_likelihood(data: bytes) -> float:
    m = detect_magic(data)
    if m and m.kind in {
        "zip", "docx", "pptx", "xlsx", "epub", "png", "jpeg", "webp", "gzip",
        "flac", "opus_ogg", "ogg", "mp4", "pdf", "mp3", "avi", "mkv_webm", "gif", "bmp", "tiff", "xz", "bz2",
    }:
        return 0.95
    ent = estimate_entropy(data)
    return min(0.95, max(0.05, (ent - 5.5) / 2.5))


def _looks_text(data: bytes) -> bool:
    return text_likeness(data) > 0.95


def detect_domain(input_path: str | Path, raw_bytes: bytes) -> str:
    m = detect_magic(raw_bytes)
    if m:
        if m.kind in {"png", "jpeg", "webp", "gif", "bmp", "tiff"}:
            return "image"
        if m.kind in {"wav", "mp3", "flac", "opus_ogg", "ogg"}:
            return "audio"
        if m.kind in {"mp4", "avi", "mkv_webm"}:
            return "video"
        if m.kind in {"pdf", "docx", "pptx", "xlsx", "epub"}:
            return "document"
        if m.kind in {"zip", "gzip", "xz", "bz2"}:
            return "archive"
        if m.kind == "text":
            return "text"
    ext = Path(input_path).suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
        return "image"
    if ext in {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}:
        return "audio"
    if ext in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}:
        return "video"
    if ext in {".txt", ".md", ".json", ".csv", ".tsv", ".log", ".xml", ".yaml", ".yml", ".html", ".htm", ".py", ".js", ".ts"}:
        return "text"
    if ext in {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".rtf", ".epub"}:
        return "document"
    if ext in {".zip", ".gz", ".tgz", ".tar", ".xz", ".bz2", ".7z", ".rar"}:
        return "archive"
    if _looks_text(raw_bytes):
        return "text"
    return "binary"


def profile_input(input_path: str | Path, raw_bytes: bytes) -> InputProfile:
    magic = detect_magic(raw_bytes)
    domain = detect_domain(input_path, raw_bytes)
    subtype = magic.kind if magic else safe_suffix(str(input_path), ".bin")
    ent = estimate_entropy(raw_bytes)
    tl = text_likeness(raw_bytes)
    acl = already_compressed_likelihood(raw_bytes)
    conf = 0.95 if magic else (0.75 if domain not in {"binary", "other"} else 0.55)
    return InputProfile(domain, subtype, magic, len(raw_bytes), ent, tl, acl, conf)


# ============================================================
# Compression stage
# ============================================================

@dataclass
class CompressionCandidate:
    method_id: int
    method_name: str
    domain: str
    lossless: bool
    encoded_bytes: bytes
    restored_bytes: Optional[bytes]
    restore_ext: str
    transform_meta: Dict[str, Any]
    size_ratio: float
    runtime_sec: float
    exact_roundtrip: bool
    already_compressed_penalty: float
    quality_penalty: float
    score: float


def _gzip_encode(data: bytes, level: int = 9) -> bytes:
    return gzip.compress(data, compresslevel=int(level), mtime=0)


def _bz2_encode(data: bytes, level: int = 9) -> bytes:
    return bz2.compress(data, compresslevel=int(level))


def _xz_encode(data: bytes, preset: int = 6) -> bytes:
    return lzma.compress(data, format=lzma.FORMAT_XZ, preset=int(preset))


def _zip_single_file(input_path: str | Path, raw_bytes: bytes, compress: bool, level: int = 6) -> bytes:
    base = safe_basename(Path(input_path).name)
    buf = io.BytesIO()
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    kwargs: Dict[str, Any] = {"compression": compression}
    if compress:
        kwargs["compresslevel"] = int(level)
    with zipfile.ZipFile(buf, "w", **kwargs) as zf:
        zf.writestr(base, raw_bytes)
    return buf.getvalue()


def _unzip_first(zip_bytes: bytes) -> Tuple[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        if not members:
            raise ValueError("ZIP archive has no file members")
        name = members[0]
        return name, zf.read(name)


def _encode_png_lossless(raw: bytes) -> bytes:
    if Image is None:
        raise RuntimeError("Pillow is not available")
    img = Image.open(io.BytesIO(raw))
    if img.mode not in ("RGB", "RGBA", "L", "P"):
        img = img.convert("RGBA")
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _encode_webp(raw: bytes, quality: int = 80, lossless: bool = False) -> bytes:
    if Image is None:
        raise RuntimeError("Pillow is not available")
    img = Image.open(io.BytesIO(raw))
    out = io.BytesIO()
    img.save(out, format="WEBP", quality=int(quality), lossless=bool(lossless), method=6)
    return out.getvalue()


def _encode_jpeg(raw: bytes, quality: int = 80) -> bytes:
    if Image is None:
        raise RuntimeError("Pillow is not available")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=int(quality), optimize=True)
    return out.getvalue()


def image_quality_penalty(raw_bytes: bytes, transformed_bytes: bytes) -> Tuple[float, Dict[str, Any]]:
    if Image is None or np is None:
        return 0.0, {"psnr": None, "mse": None, "ssim": None}
    try:
        src = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        dst = Image.open(io.BytesIO(transformed_bytes)).convert("RGB")
        if dst.size != src.size:
            dst = dst.resize(src.size)
        a = np.asarray(src).astype(np.float32)
        b = np.asarray(dst).astype(np.float32)
        mse = float(np.mean((a - b) ** 2))
        psnr = 99.0 if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse))
        # Simple global SSIM approximation for a lightweight dependency-free metric.
        scores = []
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        for ch in range(3):
            x = a[:, :, ch].astype(np.float64)
            y = b[:, :, ch].astype(np.float64)
            mux, muy = float(np.mean(x)), float(np.mean(y))
            vx, vy = float(np.var(x)), float(np.var(y))
            cov = float(np.mean((x - mux) * (y - muy)))
            den = (mux * mux + muy * muy + c1) * (vx + vy + c2)
            scores.append(1.0 if abs(den) < 1e-12 else ((2 * mux * muy + c1) * (2 * cov + c2)) / den)
        ssim = max(-1.0, min(1.0, float(np.mean(scores))))
        psnr_penalty = max(0.0, min(1.0, (40.0 - psnr) / 40.0))
        ssim_penalty = max(0.0, min(1.0, 1.0 - max(0.0, min(1.0, ssim))))
        penalty = 0.55 * psnr_penalty + 0.45 * ssim_penalty
        return float(penalty), {"psnr": float(psnr), "mse": float(mse), "ssim": float(ssim), "psnr_penalty": psnr_penalty, "ssim_penalty": ssim_penalty}
    except Exception:
        return 1.0, {"psnr": None, "mse": None, "ssim": None}


def compression_objective(output_size_bytes: int, input_size_bytes: int, runtime_sec: float, already_comp_penalty: float, exact_ok: bool, operation_mode: str, quality_penalty: float = 0.0) -> float:
    """Original-style hierarchical compression objective, adapted to headerless payload size.

    Score lower = better.
    - exact_recovery rejects non-exact candidates.
    - content_preserving allows lossy candidates but penalizes lower PSNR/SSIM.
    """
    if operation_mode == "exact_recovery" and not exact_ok:
        return float("inf")
    size_ratio = output_size_bytes / max(1, input_size_bytes)
    runtime_norm = min(5.0, runtime_sec / 1.0)
    if operation_mode == "content_preserving":
        return float(0.60 * size_ratio + 0.10 * runtime_norm + 0.25 * quality_penalty + 0.05 * already_comp_penalty)
    return float(0.75 * size_ratio + 0.10 * runtime_norm + 0.15 * already_comp_penalty)


def _add_candidate(candidates: List[CompressionCandidate], *, method_id: int, method_name: str, domain: str, raw_bytes: bytes, encoded: bytes, restored: Optional[bytes], restore_ext: str, lossless: bool, input_size: int, elapsed: float, profile: InputProfile, operation_mode: str, quality_penalty: float = 0.0, meta: Optional[Dict[str, Any]] = None) -> None:
    if not encoded:
        return
    if detect_magic(encoded) is None:
        # Only self-describing output bytes can be sent to DNA in this headerless pipeline.
        return
    exact_ok = restored == raw_bytes if restored is not None else lossless
    already_penalty = profile.already_compressed_likelihood_value if profile.already_compressed_likelihood_value > 0.7 and method_name != "keep" else 0.0
    score = compression_objective(len(encoded), input_size, elapsed, already_penalty, bool(exact_ok), operation_mode, quality_penalty)
    if not math.isfinite(score):
        return
    candidates.append(CompressionCandidate(
        method_id=method_id,
        method_name=method_name,
        domain=domain,
        lossless=lossless,
        encoded_bytes=encoded,
        restored_bytes=restored,
        restore_ext=restore_ext,
        transform_meta=meta or {},
        size_ratio=len(encoded) / max(1, input_size),
        runtime_sec=float(elapsed),
        exact_roundtrip=bool(exact_ok),
        already_compressed_penalty=float(already_penalty),
        quality_penalty=float(quality_penalty),
        score=float(score),
    ))


def build_compression_candidates(input_path: str | Path, raw_bytes: bytes, profile: InputProfile, operation_mode: str = "content_preserving") -> Tuple[str, List[CompressionCandidate]]:
    domain = profile.domain
    input_size = max(1, len(raw_bytes))
    suffix = safe_suffix(str(input_path))
    candidates: List[CompressionCandidate] = []
    method_id = 0

    # Keep original if it is self-describing.
    if detect_magic(raw_bytes) is not None:
        _add_candidate(candidates, method_id=method_id, method_name="keep_original", domain=domain, raw_bytes=raw_bytes, encoded=raw_bytes, restored=raw_bytes, restore_ext=suffix, lossless=True, input_size=input_size, elapsed=0.0, profile=profile, operation_mode=operation_mode, meta={"policy": "no_compression"})
        method_id += 1

    def try_add(name: str, fn, dec_fn=None, ext: str = suffix, lossless: bool = True, is_image_quality: bool = False):
        nonlocal method_id
        t0 = time.time()
        try:
            encoded = fn()
            restored = dec_fn(encoded) if dec_fn else encoded
            elapsed = time.time() - t0
            qpen, qmeta = (image_quality_penalty(raw_bytes, encoded) if is_image_quality else (0.0, {}))
            _add_candidate(candidates, method_id=method_id, method_name=name, domain=domain, raw_bytes=raw_bytes, encoded=encoded, restored=restored, restore_ext=ext, lossless=lossless, input_size=input_size, elapsed=elapsed, profile=profile, operation_mode=operation_mode, quality_penalty=qpen, meta=qmeta)
        except Exception:
            pass
        method_id += 1

    if domain == "image":
        try_add("png_lossless", lambda: _encode_png_lossless(raw_bytes), ext=".png", lossless=True, is_image_quality=True)
        try_add("webp_lossless", lambda: _encode_webp(raw_bytes, quality=100, lossless=True), ext=".webp", lossless=True, is_image_quality=True)
        if operation_mode == "content_preserving":
            for q in (50, 60, 70, 80, 90):
                try_add(f"webp_q{q}", lambda q=q: _encode_webp(raw_bytes, quality=q, lossless=False), ext=".webp", lossless=False, is_image_quality=True)
                try_add(f"jpeg_q{q}", lambda q=q: _encode_jpeg(raw_bytes, quality=q), ext=".jpg", lossless=False, is_image_quality=True)
    # Generic self-describing lossless wrappers for all domains as fallback/comparison.
    for lvl in (1, 6, 9):
        try_add(f"gzip_lvl{lvl}", lambda lvl=lvl: _gzip_encode(raw_bytes, lvl), dec_fn=lambda b: gzip.decompress(b), ext=suffix, lossless=True)
    for lvl in (1, 6, 9):
        try_add(f"bz2_lvl{lvl}", lambda lvl=lvl: _bz2_encode(raw_bytes, lvl), dec_fn=lambda b: bz2.decompress(b), ext=suffix, lossless=True)
    for preset in (0, 6, 9):
        try_add(f"xz_p{preset}", lambda preset=preset: _xz_encode(raw_bytes, preset), dec_fn=lambda b: lzma.decompress(b, format=lzma.FORMAT_XZ), ext=suffix, lossless=True)
    try_add("zip_store", lambda: _zip_single_file(input_path, raw_bytes, compress=False), dec_fn=lambda b: _unzip_first(b)[1], ext=suffix, lossless=True)
    for lvl in (1, 6, 9):
        try_add(f"zip_deflate_lvl{lvl}", lambda lvl=lvl: _zip_single_file(input_path, raw_bytes, compress=True, level=lvl), dec_fn=lambda b: _unzip_first(b)[1], ext=suffix, lossless=True)

    if not candidates:
        raise RuntimeError("No valid self-describing compression candidates were produced.")
    candidates.sort(key=lambda c: (c.score, len(c.encoded_bytes), c.method_name))
    return operation_mode, candidates


# ============================================================
# DNA design stage: hierarchical, no whitening, Simple + rule-based × 4 dimers
# ============================================================

@dataclass
class DNACandidate:
    rule_id: int
    rule_name: str
    mode: str
    init_dimer: str
    whiten: bool
    prepend_one: bool
    remove_leading_one: bool
    dna: str
    bits_per_nt: float
    gc_fraction: float
    homopolymer: Dict[str, int]
    motif_count: int
    decode_complexity: float
    score: float


def _homopolymer_profile(dna: str) -> Dict[str, int]:
    hp = dna_codec.homopolymer_stats(dna)
    return {
        "max_run": int(hp.get("longest", 0)),
        "longest": int(hp.get("longest", 0)),
        "count_ge2": int(hp.get("count_ge2", hp.get("homo_count", 0))),
        "count_ge4": int(hp.get("count_ge4", 0)),
        "count_ge5": int(hp.get("exact_len_ge5", 0)),
        "total_runs": int(hp.get("total_runs", 0)),
    }


def _motif_count(dna: str) -> int:
    total = 0
    for motif in FORBIDDEN_MOTIFS:
        start = 0
        while True:
            idx = dna.find(motif, start)
            if idx < 0:
                break
            total += 1
            start = idx + 1
    return total


def dna_design_score(dna: str, payload_bits_len: int, mode: str) -> Tuple[float, Dict[str, Any]]:
    bits_per_nt = payload_bits_len / max(1, len(dna))
    nt_per_bit = len(dna) / max(1, payload_bits_len)
    gc = dna_codec.gc_content(dna)
    hp = _homopolymer_profile(dna)
    motifs = _motif_count(dna)
    hp_norm = ((max(0, hp["max_run"] - 3) / 6.0) + (hp["count_ge4"] / max(1, len(dna) / 100.0)) + (hp["count_ge5"] / max(1, len(dna) / 120.0))) / 3.0
    motif_norm = motifs / max(1.0, len(dna) / 100.0)
    decode_complexity = 0.0 if mode == "SIMPLE" else 0.5
    score = 0.45 * nt_per_bit + 0.20 * abs(gc - 0.50) + 0.20 * hp_norm + 0.10 * motif_norm + 0.05 * decode_complexity
    return float(score), {
        "bits_per_nt": float(bits_per_nt),
        "gc_fraction": float(gc),
        "homopolymer": hp,
        "motif_count": int(motifs),
        "decode_complexity": float(decode_complexity),
    }


def build_dna_candidates(payload_bytes: bytes, seed: str = "ddss-seed") -> List[DNACandidate]:
    bits = bytes_to_bitstring(payload_bytes)
    candidates: List[DNACandidate] = []

    # Simple Mapping baseline
    dna, _ = dna_codec.encode_bits_to_dna(bits, scheme_name="RINF_B16", mode="SIMPLE", seed=seed, prepend_one=False, whiten=False)
    score, meta = dna_design_score(dna, len(bits), "SIMPLE")
    candidates.append(DNACandidate(0, "Simple Mapping", "SIMPLE", "TA", False, False, False, dna, meta["bits_per_nt"], meta["gc_fraction"], meta["homopolymer"], meta["motif_count"], meta["decode_complexity"], score))

    rule_id = 1
    for rule_name in DNA_SCHEMES:
        for init_dimer in INIT_DIMERS:
            try:
                dna, _ = dna_codec.encode_bits_to_dna(
                    bits,
                    scheme_name=rule_name,
                    mode="TABLE",
                    seed=seed,
                    init_dimer=init_dimer,
                    prepend_one=True,
                    whiten=False,
                    target_gc=0.50,
                    w_gc=2.0,
                    w_motif=1.0,
                    ks=(4, 6),
                )
                score, meta = dna_design_score(dna, len(bits), "TABLE")
                candidates.append(DNACandidate(rule_id, rule_name, "TABLE", init_dimer, False, True, True, dna, meta["bits_per_nt"], meta["gc_fraction"], meta["homopolymer"], meta["motif_count"], meta["decode_complexity"], score))
                rule_id += 1
            except Exception:
                continue
    if not candidates:
        raise RuntimeError("No valid DNA candidates")
    candidates.sort(key=lambda c: (c.score, len(c.dna), c.rule_name, c.init_dimer))
    return candidates


# ============================================================
# Decode/restore
# ============================================================

def _clean_dna_text(text: str) -> str:
    return re.sub(r"[^ACGTacgt]", "", text or "").upper()


def restore_rep(inner_bytes: bytes, out_dir: str | Path, preferred_stem: str = "restored") -> Tuple[str, Dict[str, Any]]:
    out_dir = ensure_dir(out_dir)
    m = detect_magic(inner_bytes)
    ext = m.ext if m else ".bin"
    outp = Path(out_dir) / f"{safe_stem(preferred_stem)}{ext}"
    write_bytes(outp, inner_bytes)
    return str(outp), {"detected_magic": m.kind if m else None, "restore_kind": "write_self_describing_bytes", "ext": ext, "mime": m.mime if m else "application/octet-stream"}


def _blind_decode_candidates(dna_text: str, seed: str) -> List[Dict[str, Any]]:
    dna_text = _clean_dna_text(dna_text)
    out: List[Dict[str, Any]] = []
    rule_space: List[Tuple[str, str, str, bool, bool]] = [("Simple Mapping", "SIMPLE", "TA", False, False)]
    for rule_name in DNA_SCHEMES:
        for init_dimer in INIT_DIMERS:
            rule_space.append((rule_name, "TABLE", init_dimer, True, True))
    for rule_name, mode, init_dimer, prepend_one, remove_leading_one in rule_space:
        try:
            bits, _ = dna_codec.decode_dna_to_bits(
                dna_text,
                scheme_name="RINF_B16" if mode == "SIMPLE" else rule_name,
                mode=mode,
                seed=seed,
                init_dimer=init_dimer,
                remove_leading_one=remove_leading_one,
                whiten=False,
            )
            payload_bytes, _ = bitstring_to_bytes(bits, pad_to_byte=True)
            m = detect_magic(payload_bytes)
            if not m:
                continue
            out.append({
                "rule_name": rule_name,
                "mode": mode,
                "init_dimer": init_dimer,
                "whiten": False,
                "payload_bytes": payload_bytes,
                "magic": m.__dict__,
                "confidence": float(m.confidence),
                "payload_size_bytes": len(payload_bytes),
                "payload_sha256": sha256_bytes(payload_bytes),
            })
        except Exception:
            continue
    out.sort(key=lambda r: (-float(r.get("confidence", 0.0)), int(r.get("payload_size_bytes", 0)), str(r.get("rule_name", ""))))
    return out


def decode_dna_sequence(dna_text: str, output_dir: str | Path, preferred_stem: str = "restored", seed: str = "ddss-seed", enable_blind_fallback: bool = True, verify_only: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    output_dir = ensure_dir(output_dir)
    dna_text = _clean_dna_text(dna_text)
    candidates = _blind_decode_candidates(dna_text, seed=seed)
    if not candidates:
        raise RuntimeError("Decoding failed: no valid Simple/rule-based candidate produced self-describing bytes.")
    chosen = candidates[0]
    payload_bytes = chosen["payload_bytes"]
    restored_path, restore_meta = restore_rep(payload_bytes, output_dir, preferred_stem=preferred_stem)
    restored_bytes = read_bytes(restored_path)
    restored_magic = detect_magic(restored_bytes)
    report = {
        "created_utc": now_utc_string(),
        "engine": ENGINE_VERSION,
        "decode_rule_used": {k: v for k, v in chosen.items() if k != "payload_bytes"},
        "blind_candidates_count": len(candidates),
        "payload": {
            "dna_len_nt": len(dna_text),
            "payload_size_bytes": len(payload_bytes),
            "payload_sha256": sha256_bytes(payload_bytes),
        },
        "restored": {
            "path": str(restored_path),
            "size_bytes": len(restored_bytes),
            "size_human": human_bytes(len(restored_bytes)),
            "sha256": sha256_bytes(restored_bytes),
            "magic": restored_magic.__dict__ if restored_magic else None,
            "restore_meta": restore_meta,
        },
        "verification": {"self_describing_payload_detected": restored_magic is not None, "blind_decode_ok": True},
        "runtime_sec": round(time.time() - t0, 6),
    }
    write_text(Path(output_dir) / "decode_report.json", json.dumps(report, indent=2, ensure_ascii=False))
    if verify_only:
        return {"verification": report["verification"], "report": report}
    return {"output_path": str(restored_path), "report": report, "verification": report["verification"]}


# ============================================================
# Reporting flatteners and encode entry point
# ============================================================

def flatten_candidate(c: CompressionCandidate) -> Dict[str, Any]:
    return {
        "method_id": c.method_id,
        "method_name": c.method_name,
        "domain": c.domain,
        "lossless": c.lossless,
        "size_bytes": len(c.encoded_bytes),
        "restore_ext": c.restore_ext,
        "score": round(float(c.score), 8),
        "size_ratio": float(c.size_ratio),
        "runtime_sec": round(float(c.runtime_sec), 6),
        "exact_roundtrip": bool(c.exact_roundtrip),
        "already_compressed_penalty": float(c.already_compressed_penalty),
        "quality_penalty": float(c.quality_penalty),
        "transform_meta": c.transform_meta,
    }


def flatten_dna_candidate(c: DNACandidate) -> Dict[str, Any]:
    return {
        "rule_id": c.rule_id,
        "rule_name": c.rule_name,
        "mode": c.mode,
        "init_dimer": c.init_dimer,
        "whiten": False,
        "prepend_one": c.prepend_one,
        "remove_leading_one": c.remove_leading_one,
        "dna_len_nt": len(c.dna),
        "bits_per_nt": c.bits_per_nt,
        "gc_fraction": c.gc_fraction,
        "homopolymer": c.homopolymer,
        "motif_count": c.motif_count,
        "decode_complexity": c.decode_complexity,
        "score": round(float(c.score), 8),
    }


def encode_file(input_path: str | Path, output_dir: str | Path, operation_mode: str = "content_preserving", strategy: str = "hierarchical_full_automatic", seed: str = "ddss-seed") -> Dict[str, Any]:
    """Hierarchical automatic pipeline.

    Stage 1: choose one compressor using the original compression objective.
    Stage 2: run DNA design on the selected self-describing output bytes.
             DNA rules = Simple Mapping + 4 rule-based schemes × 4 initial dimers.
             Whitening is disabled.
    """
    t0 = time.time()
    input_path = Path(input_path)
    output_dir = ensure_dir(output_dir)
    artifacts_dir = ensure_dir(Path(output_dir) / "artifacts")
    raw_bytes = input_path.read_bytes()
    profile = profile_input(input_path, raw_bytes)

    effective_operation_mode, compression_candidates = build_compression_candidates(input_path, raw_bytes, profile, operation_mode=operation_mode)
    chosen_comp = compression_candidates[0]
    payload_bytes = chosen_comp.encoded_bytes

    dna_candidates = build_dna_candidates(payload_bytes, seed=seed)
    chosen_dna = dna_candidates[0]
    final_dna = chosen_dna.dna

    # Artifacts
    write_text(Path(output_dir) / "dna_sequence.txt", final_dna)
    write_bytes(Path(output_dir) / "representation.bin", payload_bytes)
    write_bytes(Path(output_dir) / "artifacts" / input_path.name, raw_bytes)
    rep_magic = detect_magic(payload_bytes)
    if rep_magic:
        write_bytes(Path(output_dir) / f"selected_{chosen_comp.method_name}{rep_magic.ext}", payload_bytes)

    # Verify decode from DNA back to selected payload bytes.
    decode_check = decode_dna_sequence(final_dna, artifacts_dir / "verification_decode", preferred_stem=safe_stem(input_path.name), seed=seed, enable_blind_fallback=True, verify_only=True)

    report = {
        "created_utc": now_utc_string(),
        "engine": ENGINE_VERSION,
        "operation_mode": operation_mode,
        "effective_operation_mode": effective_operation_mode,
        "strategy": "hierarchical_compressor_then_dna_rule_no_whitening",
        "input_profile": {
            "domain": profile.domain,
            "subtype": profile.subtype,
            "input_size_bytes": profile.input_size_bytes,
            "entropy_estimate": profile.entropy_estimate,
            "text_likeness": profile.text_likeness_value,
            "already_compressed_likelihood": profile.already_compressed_likelihood_value,
            "profiler_confidence": profile.profiler_confidence,
            "magic": profile.magic.__dict__ if profile.magic else None,
        },
        "input": {
            "path": str(input_path),
            "filename": input_path.name,
            "size_bytes": len(raw_bytes),
            "size_human": human_bytes(len(raw_bytes)),
            "sha256": sha256_bytes(raw_bytes),
        },
        "compression": {
            "selection_mode": "hierarchical_stage_1",
            "formula": {
                "exact_recovery": "0.75*size_ratio + 0.10*runtime_norm + 0.15*already_compressed_penalty; non-exact candidates are rejected",
                "content_preserving": "0.60*size_ratio + 0.10*runtime_norm + 0.25*quality_penalty + 0.05*already_compressed_penalty",
                "size_ratio": "self_describing_output_bytes / input_bytes",
            },
            "selected": flatten_candidate(chosen_comp),
            "candidates": [flatten_candidate(c) for c in compression_candidates],
        },
        "dna_payload": {
            "description": "selected self-describing compressor output bytes used directly as DNA payload",
            "size_bytes": len(payload_bytes),
            "size_human": human_bytes(len(payload_bytes)),
            "sha256": sha256_bytes(payload_bytes),
            "magic": rep_magic.__dict__ if rep_magic else None,
        },
        "dna_design": {
            "selection_mode": "hierarchical_stage_2",
            "rules": "Simple Mapping + RINF_B16/R2_B15/R1_B12/R0_B9 with initial dimers TA/AT/CG/GC; whitening disabled",
            "formula": "0.45*nt_per_bit + 0.20*abs(GC-0.50) + 0.20*homopolymer_penalty + 0.10*motif_penalty + 0.05*decode_complexity",
            "selected": flatten_dna_candidate(chosen_dna),
            "candidates": [flatten_dna_candidate(c) for c in dna_candidates],
            "final_dna_len_nt": len(final_dna),
        },
        "verification": decode_check["verification"],
        "runtime_sec": round(time.time() - t0, 6),
        "artifacts": {
            "dna_sequence_txt": str(Path(output_dir) / "dna_sequence.txt"),
            "representation_bin": str(Path(output_dir) / "representation.bin"),
        },
    }
    write_text(Path(output_dir) / "encode_report.json", json.dumps(report, indent=2, ensure_ascii=False))
    return {"dna_sequence": final_dna, "report": report, "output_dir": str(output_dir)}
