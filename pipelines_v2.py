# pipelines_v2.py
# Step-1 only: bytes -> (representation) -> zlib framing -> DNA -> zlib inflate -> rep bytes -> restore file
# - No custom header in the stored DNA.
# - Integrity + end-of-stream come from zlib stream framing (header+Adler32).
# - File-type routing comes from magic bytes of the representation (ZIP/PNG/WEBP/Ogg/MP4/PDF/DOCX/...).
#
# IMPORTANT: dna_codec.py is reused as-is. Do not modify dna_codec.py.

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import dna_codec  # must remain unchanged

from compressors_v2 import (
    domain_detect_and_encode_rep,
    benchmark_domain_encode_rep,
    restore_rep,
    zip_single_file,
    zip_store_single_file,
)
from utils_bits_v2 import (
    MagicInfo,
    bitstring_to_bytes,
    bytes_to_bitstring,
    detect_magic,
    ensure_dir,
    read_bytes,
    sha256_bytes,
    sha256_file,
    write_bytes,
    write_text,
    zlib_inflate_until_eof,
    zlib_wrap,
)


# ----------------------------
# Job + artifacts helpers
# ----------------------------

def _new_job_dir(root: str = "jobs") -> str:
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(root, job_id)
    ensure_dir(job_dir)
    return job_dir


def _paths(job_dir: str) -> Dict[str, str]:
    art = os.path.join(job_dir, "artifacts")
    out = os.path.join(job_dir, "output")
    ensure_dir(art)
    ensure_dir(out)
    return {
        "job_dir": job_dir,
        "artifacts": art,
        "output": out,
        "report_json": os.path.join(job_dir, "report.json"),
    }


def _write_common_artifacts(job_dir: str, input_path: str, raw: bytes) -> Dict[str, str]:
    P = _paths(job_dir)
    input_copy = os.path.join(P["artifacts"], os.path.basename(input_path))
    write_bytes(input_copy, raw)
    return {"input_original": input_copy}


# ----------------------------
# Core encode/decode helpers
# ----------------------------

def _encode_to_dna(
    zlib_stream: bytes,
    *,
    scheme_name: str,
    mode: str,
    seed: str,
    init_dimer: str,
    prepend_one: bool,
    whiten: bool,
    target_gc: float,
    w_gc: float,
    w_motif: float,
    ks: Tuple[int, ...],
) -> Tuple[str, Dict[str, Any]]:
    bits = bytes_to_bitstring(zlib_stream)
    if bits == "":
        bits = "0"
    t0 = time.time()
    dna, digits = dna_codec.encode_bits_to_dna(
        bits,
        scheme_name=scheme_name,
        mode=mode,
        seed=seed,
        init_dimer=init_dimer,
        prepend_one=prepend_one,
        whiten=whiten,
        target_gc=target_gc,
        w_gc=w_gc,
        w_motif=w_motif,
        ks=ks,
    )
    t1 = time.time()
    meta = {
        "encode_time_sec": t1 - t0,
        "bits_len_payload": len(bits),
        "digits_len": (len(digits) if isinstance(digits, list) else None),
    }
    return dna, meta


def _decode_from_dna(
    dna: str,
    *,
    scheme_name: str,
    mode: str,
    seed: str,
    init_dimer: str,
    remove_leading_one: bool,
    whiten: bool,
    target_gc: float,
    w_gc: float,
    w_motif: float,
    ks: Tuple[int, ...],
) -> Tuple[bytes, Dict[str, Any]]:
    t0 = time.time()
    bits, digits = dna_codec.decode_dna_to_bits(
        dna,
        scheme_name=scheme_name,
        mode=mode,
        seed=seed,
        init_dimer=init_dimer,
        remove_leading_one=remove_leading_one,
        whiten=whiten,
        target_gc=target_gc,
        w_gc=w_gc,
        w_motif=w_motif,
        ks=ks,
    )
    t1 = time.time()
    buf, pad_bits_to_byte = bitstring_to_bytes(bits, pad_to_byte=True)
    meta = {
        "decode_time_sec": t1 - t0,
        "bits_len": len(bits),
        "bytes_len": len(buf),
        "pad_bits_to_byte": pad_bits_to_byte,
        "digits_len": (len(digits) if isinstance(digits, list) else None),
    }
    return buf, meta


def _dna_stats(dna: str) -> Dict[str, Any]:
    gc = dna_codec.gc_content(dna) if hasattr(dna_codec, "gc_content") else None
    hp = dna_codec.homopolymer_stats(dna) if hasattr(dna_codec, "homopolymer_stats") else None
    return {"dna_len_nt": len(dna), "gc_fraction": gc, "homopolymer": hp}


# ----------------------------
# Mode runners (Step-1)
# ----------------------------

def _run_common(
    *,
    mode_name: str,
    input_path: str,
    make_rep_fn,
    scheme_name: str = "RINF_B16",
    mode_codec: str = "TABLE",
    seed: str = "rn",
    init_dimer: str = "TA",
    prepend_one: bool = True,
    remove_leading_one: bool = True,
    whiten: bool = True,
    target_gc: float = 0.50,
    w_gc: float = 2.0,
    w_motif: float = 1.0,
    ks: Tuple[int, ...] = (4, 6),
    zlib_policy: str = "auto",  # auto|stored|compress
) -> Tuple[str, Dict[str, Any]]:
    job_dir = _new_job_dir("jobs")
    P = _paths(job_dir)

    raw = read_bytes(input_path)
    sha_in = sha256_bytes(raw)
    artifacts = _write_common_artifacts(job_dir, input_path, raw)

    # 1) Build representation bytes (inner_bytes)
    inner_bytes, rep_meta = make_rep_fn(input_path, raw)
    inner_sha = sha256_bytes(inner_bytes)

    inner_path = os.path.join(P["artifacts"], "inner.bin")
    write_bytes(inner_path, inner_bytes)

    # 2) ZLIB framing to get self-terminating + self-checking stream (handles padding tail)
    zlib_stream, zmeta = zlib_wrap(inner_bytes, policy=zlib_policy)
    zlib_path = os.path.join(P["artifacts"], "zlib_stream.bin")
    write_bytes(zlib_path, zlib_stream)

    # 3) Encode to DNA
    dna, enc_meta = _encode_to_dna(
        zlib_stream,
        scheme_name=scheme_name,
        mode=mode_codec,
        seed=seed,
        init_dimer=init_dimer,
        prepend_one=prepend_one,
        whiten=whiten,
        target_gc=target_gc,
        w_gc=w_gc,
        w_motif=w_motif,
        ks=ks,
    )

    dna_full_path = os.path.join(P["artifacts"], "dna.txt")
    dna_fasta_path = os.path.join(P["artifacts"], "dna.fasta")
    write_text(dna_full_path, dna + "\n")
    # simple FASTA wrap
    wrap = 100
    fasta_lines = [">dna"] + [dna[i:i+wrap] for i in range(0, len(dna), wrap)]
    write_text(dna_fasta_path, "\n".join(fasta_lines) + "\n")

    # 4) Decode back (DNA -> bytes buffer)
    decoded_buf, dec_meta = _decode_from_dna(
        dna,
        scheme_name=scheme_name,
        mode=mode_codec,
        seed=seed,
        init_dimer=init_dimer,
        remove_leading_one=remove_leading_one,
        whiten=whiten,
        target_gc=target_gc,
        w_gc=w_gc,
        w_motif=w_motif,
        ks=ks,
    )

    # 5) Inflate zlib stream (first stream only); ignore unused tail safely
    inflated, infl_meta = zlib_inflate_until_eof(decoded_buf)
    zlib_ok = bool(infl_meta.get("eof", False)) and (infl_meta.get("error") is None)
    infl_tail = int(infl_meta.get("unused_tail_len", 0) or 0)

    recovered_inner_path = os.path.join(P["artifacts"], "recovered_inner.bin")
    write_bytes(recovered_inner_path, inflated)

    # 6) Restore to output file via magic-based routing
    restored_file, restore_meta = restore_rep(inflated, out_dir=P["output"], preferred_stem="restored")

    # 7) Verifications
    roundtrip_inner_ok = (sha256_bytes(inflated) == inner_sha)
    original_roundtrip_ok = False
    if mode_name == "mode0_raw":
        # For raw pipeline, restored file should match original bytes (if routing wrote exact bytes)
        try:
            original_roundtrip_ok = (sha256_file(restored_file) == sha_in)
        except Exception:
            original_roundtrip_ok = False
    elif mode_name in {"mode1_zip","mode2_zip_store"}:
        # For ZIP pipelines, restored_file should be extracted original
        try:
            original_roundtrip_ok = (sha256_file(restored_file) == sha_in)
        except Exception:
            original_roundtrip_ok = False
    else:
        # For mode3, original may NOT match if lossy; report accordingly
        original_roundtrip_ok = (sha256_file(restored_file) == sha_in)

    # 8) Report
    dna_meta = _dna_stats(dna)
    bits_per_nt = (enc_meta.get("bits_len_payload", 0) / max(1, dna_meta["dna_len_nt"])) if dna_meta["dna_len_nt"] else None

    report: Dict[str, Any] = {
        "job_uuid": os.path.basename(job_dir),
        "mode": mode_name,
        "status": "ok" if zlib_ok else "error",
        "error": (infl_meta.get("error") if not zlib_ok else None),

        "input": {"path": input_path, "size_bytes": len(raw), "sha256": sha_in},
        "rep": {"size_bytes": len(inner_bytes), "sha256": inner_sha, "meta": rep_meta},
        "zlib_stream": {"size_bytes": len(zlib_stream), "meta": zmeta},
        "dna": {
            "dna_len_nt": dna_meta["dna_len_nt"],
            "bits_len": enc_meta.get("bits_len_payload"),
            "bits_per_nt_est": bits_per_nt,
            "encode_time_sec": enc_meta.get("encode_time_sec"),
            "decode_time_sec": dec_meta.get("decode_time_sec"),
            "gc_fraction": dna_meta.get("gc_fraction"),
            "homopolymer": dna_meta.get("homopolymer"),
        },
        "decoded": {
            "bits_len": dec_meta.get("bits_len"),
            "bytes_len": dec_meta.get("bytes_len"),
            "pad_bits_to_byte": dec_meta.get("pad_bits_to_byte"),
        },
        "inflate": {
            "eof": infl_meta.get("eof"),
            "unused_tail_len_bytes": infl_tail,
            "zlib_integrity_ok": zlib_ok,
        },
        "flags": {
            "dna_roundtrip_ok": roundtrip_inner_ok,
            "zlib_integrity_ok": zlib_ok,
            "magic_detect_ok": (detect_magic(inflated) is not None),
            "roundtrip_ok": original_roundtrip_ok if mode_name != "mode3_domain" else None,
        },
        "output": {
            "restored_file": restored_file,
            "restore_meta": restore_meta,
        },
        "artifacts": {
            **artifacts,
            "inner_bin": inner_path,
            "zlib_stream_bin": zlib_path,
            "dna_txt": dna_full_path,
            "dna_fasta": dna_fasta_path,
            "recovered_inner_bin": recovered_inner_path,
            "report_json": P["report_json"],
        },
        "metrics": {
            "compression_ratio_rep_over_input": (len(inner_bytes) / max(1, len(raw))),
            "compression_ratio_zlib_over_rep": (len(zlib_stream) / max(1, len(inner_bytes))),
        }
    }

    write_text(P["report_json"], json.dumps(report, indent=2, ensure_ascii=False))
    return job_dir, report


def run_mode0_raw(
    input_path: str,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    def make_rep(_path: str, raw: bytes) -> Tuple[bytes, Dict[str, Any]]:
        m = detect_magic(raw)
        return raw, {"kind": "raw_bytes", "magic": (m.kind if m else None)}
    return _run_common(mode_name="mode0_raw", input_path=input_path, make_rep_fn=make_rep, **kwargs)


def run_mode1_zip(
    input_path: str,
    zip_level: int = 6,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    def make_rep(path: str, _raw: bytes) -> Tuple[bytes, Dict[str, Any]]:
        rep, meta = zip_single_file(path, level=zip_level)
        return rep, meta
    return _run_common(mode_name="mode1_zip", input_path=input_path, make_rep_fn=make_rep, **kwargs)


def run_mode2_zip_store(
    input_path: str,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    def make_rep(path: str, _raw: bytes) -> Tuple[bytes, Dict[str, Any]]:
        rep, meta = zip_store_single_file(path)
        return rep, meta
    return _run_common(mode_name="mode2_zip_store", input_path=input_path, make_rep_fn=make_rep, **kwargs)


def run_mode3_domain(
    input_path: str,
    *,
    image_policy: str = "webp_lossy",
    webp_quality: int = 80,
    text_policy: str = "gzip",
    allow_external_ffmpeg: bool = False,
    audio_policy: str = "opus_ogg",
    opus_bitrate_kbps: int = 64,
    video_policy: str = "mp4_h264",
    video_crf: int = 28,
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    def make_rep(path: str, raw: bytes) -> Tuple[bytes, Dict[str, Any]]:
        rr = domain_detect_and_encode_rep(
            path,
            raw,
            image_policy=image_policy,
            webp_quality=webp_quality,
            text_policy=text_policy,
            allow_external_ffmpeg=allow_external_ffmpeg,
            audio_policy=audio_policy,
            opus_bitrate_kbps=opus_bitrate_kbps,
            video_policy=video_policy,
            video_crf=video_crf,
        )
        return rr.rep_bytes, rr.rep_meta
    return _run_common(mode_name="mode3_domain", input_path=input_path, make_rep_fn=make_rep, **kwargs)


def run_mode3_best(
    input_path: str,
    *,
    # Policy-style knobs (kept for UI compatibility; used to bias the benchmark search space)
    image_policy: str = "webp_lossy",
    webp_quality: int = 80,
    text_policy: str = "gzip",
    allow_external_ffmpeg: bool = False,
    audio_policy: str = "opus_ogg",
    opus_bitrate_kbps: int = 64,
    video_policy: str = "mp4_h264",
    video_crf: int = 28,
    # Benchmark controls
    quality_mode: str = "Lossy",  # "Lossless" | "Lossy"
    **kwargs,
) -> Tuple[str, Dict[str, Any]]:
    """
    Mode 3 (Best-of) — benchmark multiple representation codecs for the detected domain
    and pick the candidate that yields the smallest *zlib-framed* size (binary right before DNA).

    Notes:
    - This function intentionally accepts the same policy parameters as run_mode3_domain so the UI
      can call either without changing argument lists.
    - For non-supported domains, callers should fall back to Mode 2 ZIP STORE.
    """
    # Use the same zlib framing policy for candidate scoring as the one used for the final pipeline.
    zlib_policy = kwargs.get("zlib_policy", "auto")

    # Build candidate grids biased around the UI values (so user sliders still matter).
    def _clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(v)))

    # Images
    if str(quality_mode).lower().startswith("lossless"):
        image_webp_qualities = (100,)  # webp-lossless path inside compressor
        image_jpeg_qualities = ()      # jpeg is inherently lossy
    else:
        q = _clamp(webp_quality, 10, 95)
        image_webp_qualities = (max(10, q - 20), q, min(95, q + 10))
        image_jpeg_qualities = (60, 80, 90)

    # Audio
    if str(quality_mode).lower().startswith("lossless"):
        opus_bitrates_kbps = ()  # opus is lossy
        mp3_bitrates_kbps = ()   # mp3 is lossy
        aac_bitrates_kbps = ()   # aac is lossy
    else:
        br = _clamp(opus_bitrate_kbps, 16, 192)
        opus_bitrates_kbps = (max(16, br - 32), br, min(192, br + 32))
        mp3_bitrates_kbps = (96, 128, 192)
        aac_bitrates_kbps = (96, 128)

    # Video (H.264 CRF lower=better quality, bigger size)
    if str(quality_mode).lower().startswith("lossless"):
        video_crfs = ()  # we don't do lossless video in v8 (future work)
        video_vp9_crfs = ()
        video_av1_crfs = ()
    else:
        crf = _clamp(video_crf, 0, 51)
        video_crfs = (max(0, crf - 5), crf, min(51, crf + 5))
        # VP9/AV1 grids (same CRF basis)
        video_vp9_crfs = (max(0, crf + 0), min(63, crf + 4), min(63, crf + 8))
        video_av1_crfs = (max(0, crf + 2), min(63, crf + 7), min(63, crf + 12))

    def make_rep(path: str, raw: bytes) -> Tuple[bytes, Dict[str, Any]]:
        rr, _bench = benchmark_domain_encode_rep(
            path,
            raw,
            quality_mode=quality_mode,
            allow_external_ffmpeg=allow_external_ffmpeg,
            zlib_policy=zlib_policy,
            image_webp_qualities=image_webp_qualities,
            image_jpeg_qualities=image_jpeg_qualities,
            opus_bitrates_kbps=opus_bitrates_kbps,
            mp3_bitrates_kbps=mp3_bitrates_kbps,
            aac_bitrates_kbps=aac_bitrates_kbps,
            video_crfs=video_crfs,
            video_vp9_crfs=video_vp9_crfs,
            video_av1_crfs=video_av1_crfs,
        )
        return rr.rep_bytes, rr.rep_meta

    return _run_common(mode_name="mode3_best", input_path=input_path, make_rep_fn=make_rep, **kwargs)
