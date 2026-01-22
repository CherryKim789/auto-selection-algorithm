# tab_designing.py (Universal Step-1 tab)
# Streamlit UI port of "Universal DNA Compressor + Sequencing-Aware DNA Storage System"
# Focus: Step-1 (file <-> single DNA string, headerless at medium level).
#
# Backend:
#   - dna_codec.py (UNCHANGED)
#   - pipelines_v2.py, compressors_v2.py, utils_bits_v2.py (Step-1 v2 implementation)
#
# Run:
#   pip install -r requirements.txt
#   streamlit run tab_designing_universal_streamlit.py
#
# Notes (Windows):
#   - For Mode 3 audio/video conversions, ffmpeg must be installed and available on PATH
#     OR disable "allow_ffmpeg" to force keep-bytes.
#
import os
import re
import io
import json
import uuid
import tempfile
from typing import Any, Dict, Optional, Tuple

import streamlit as st

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import math
from PIL import Image

import dna_codec
from compressors_v2 import restore_rep
from pipelines_v2 import run_mode0_raw, run_mode1_zip, run_mode2_zip_store, run_mode3_domain, run_mode3_best
from utils_bits_v2 import (
    detect_magic,
    ensure_dir,
    safe_basename,
    zlib_inflate_until_eof,
    bitstring_to_bytes,
)

# ----------------------------
# Global dirs
# ----------------------------
ensure_dir("jobs")
ensure_dir("recovery_out")
ensure_dir(".streamlit_tmp_uploads")

# CSS chuẩn: Tab mẹ 24px, Tab con 18px (giữ style của platform)
st.markdown("""
<style>
.stTabs [data-baseweb="tab"] p { font-size: 24px !important; font-weight: bold !important; }
[data-testid="stHorizontalBlock"] .stTabs [data-baseweb="tab"] p { font-size: 18px !important; font-weight: normal !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Helpers
# ----------------------------

def _compute_psnr_ssim(img_a_path: str, img_b_path: str):
    """Compute PSNR and (optionally) SSIM for two images. Returns (psnr, ssim)."""
    try:
        a = Image.open(img_a_path).convert("RGB")
        b = Image.open(img_b_path).convert("RGB")
        if a.size != b.size:
            b = b.resize(a.size)

        a_np = np.asarray(a).astype(np.float32)
        b_np = np.asarray(b).astype(np.float32)

        mse = float(np.mean((a_np - b_np) ** 2))
        if mse <= 1e-12:
            psnr = 99.0
        else:
            psnr = 20.0 * math.log10(255.0 / math.sqrt(mse))

        ssim_val = None
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_c = []
            for ch in range(3):
                ssim_c.append(float(ssim(a_np[:, :, ch], b_np[:, :, ch], data_range=255.0)))
            ssim_val = float(np.mean(ssim_c))
        except Exception:
            ssim_val = None

        return float(psnr), (float(ssim_val) if ssim_val is not None else None)
    except Exception:
        return None, None

def _plot_bits_grouped(job: str, input_bits: int, compressed_bits: int):
    # Two thin bars: Input vs Compressed
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Input", "Compressed"],
        y=[input_bits, compressed_bits],
        width=[0.35, 0.35],
    ))
    fig.update_layout(
        title="Bit-size (before DNA) — Input vs Compressed",
        barmode="group",
        bargap=0.55,
        height=320,
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=False,
    )
    fig.update_yaxes(title="bits")
    return fig


def _plot_compression_dial(job: str, input_bits: int, compressed_bits: int):
    # Dial shows compression factor = input/compressed (how many times smaller)
    if compressed_bits <= 0 or input_bits <= 0:
        factor = 0.0
    else:
        factor = float(input_bits) / float(compressed_bits)

    max_v = max(2.0, factor * 1.2, 10.0)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=factor,
        number={"suffix": "×"},
        title={"text": "Compression factor (Input / Compressed)"},
        gauge={
            "axis": {"range": [0, max_v]},
            "bar": {"thickness": 0.25},
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def _plot_psnr_ssim_combo(job: str, psnr: float, ssim_val: float):
    # Combo chart with secondary axis: PSNR on left, SSIM on right
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=[job], y=[psnr], name="PSNR (dB)", width=[0.35]), secondary_y=False)
    fig.add_trace(go.Scatter(x=[job], y=[ssim_val], name="SSIM", mode="lines+markers"), secondary_y=True)

    fig.update_layout(
        title="Image quality — PSNR (primary) + SSIM (secondary)",
        barmode="group",
        height=340,
        margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="PSNR (dB)", secondary_y=False)
    fig.update_yaxes(title_text="SSIM", secondary_y=True, range=[0, 1])
    return fig



def _word_edit_distance(a_words, b_words) -> int:
    """Levenshtein distance at word level."""
    n, m = len(a_words), len(b_words)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a_words[i - 1] == b_words[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def _is_text_like(path: str) -> bool:
    ext = (os.path.splitext(path)[1] or "").lower()
    if ext in {".txt",".md",".csv",".tsv",".json",".xml",".html",".htm",".py",".js",".css",".yaml",".yml",".log"}:
        return True
    try:
        data = open(path, "rb").read(4096)
        if b"\x00" in data:
            return False
        s = data.decode("utf-8", errors="replace")
        repl = s.count("\ufffd")
        return repl / max(1, len(s)) < 0.02
    except Exception:
        return False


def _read_text_words(path: str):
    try:
        s = open(path, "rb").read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    return [w for w in re.split(r"\s+", s.strip()) if w]

def _build_analysis_df(reports: list) -> "pd.DataFrame":
    rows = []
    for r in reports:
        inp = (r.get("input", {}) or {})
        rep = (r.get("rep", {}) or {})
        zlb = (r.get("zlib_stream", {}) or {})
        dna = (r.get("dna", {}) or {})
        out = (r.get("output", {}) or {})
        art = (r.get("artifacts", {}) or {})
        flags = (r.get("flags", {}) or {})
        rep_meta = ((rep.get("meta", {}) or {}) if isinstance(rep, dict) else {})

        input_bytes = int(inp.get("size_bytes") or 0)
        rep_bytes   = int(rep.get("size_bytes") or 0) if isinstance(rep, dict) else 0
        zlib_bytes  = int(zlb.get("size_bytes") or 0) if isinstance(zlb, dict) else 0

        input_bits = input_bytes * 8
        rep_bits   = rep_bytes * 8
        zlib_bits  = zlib_bytes * 8

        dna_nt = dna.get("dna_len_nt") if isinstance(dna, dict) else None
        bits_per_nt = dna.get("bits_per_nt_est") if isinstance(dna, dict) else None
        dna_bits_est = None
        try:
            if dna_nt is not None and bits_per_nt is not None:
                dna_bits_est = float(dna_nt) * float(bits_per_nt)
        except Exception:
            dna_bits_est = None

        rows.append({
            "job_uuid": r.get("job_uuid"),
            "mode": r.get("mode"),
            "status": r.get("status"),
            "domain": rep_meta.get("domain") or rep_meta.get("detected_domain"),
            "lossy": rep_meta.get("lossy"),
            "input_bits": input_bits,
            "rep_bits": rep_bits,
            "zlib_bits": zlib_bits,
            "dna_nt": dna_nt,
            "bits_per_nt": bits_per_nt,
            "dna_bits_est": dna_bits_est,
            "ratio_rep_over_input": (rep_bytes / input_bytes) if input_bytes else None,
            "ratio_zlib_over_input": (zlib_bytes / input_bytes) if input_bytes else None,
            "zlib_ok": (flags.get("zlib_integrity_ok") if isinstance(flags, dict) else None),
            "restored_file": out.get("restored_file") if isinstance(out, dict) else None,
            "input_original": art.get("input_original") if isinstance(art, dict) else None,
        })
    return pd.DataFrame(rows)

def _md_kv(d: Dict[str, Any], title: str = "Summary") -> str:
    lines = [f"### {title}"]
    for k, v in d.items():
        lines.append(f"- **{k}**: `{v}`")
    return "\n".join(lines)

def _render_step1_report(report: Dict[str, Any]) -> str:
    if not report:
        return "No report."
    mode = report.get("mode")
    status = report.get("status")
    err = report.get("error")
    inp = report.get("input", {})
    rep = report.get("rep", {})
    zlib_s = report.get("zlib_stream", {})
    dna = report.get("dna", {})
    infl = report.get("inflate", {})
    flags = report.get("flags", {})
    out = report.get("output", {})

    summary = {
        "mode": mode,
        "status": status,
        "error": err,
        "input_bytes": inp.get("size_bytes"),
        "rep_bytes": rep.get("size_bytes"),
        "zlib_stream_bytes": zlib_s.get("size_bytes"),
        "dna_len_nt": dna.get("dna_len_nt"),
        "bits_per_nt_est": round(dna.get("bits_per_nt_est") or 0, 6) if dna.get("bits_per_nt_est") is not None else None,
        "zlib_eof": infl.get("eof"),
        "unused_tail_bytes": infl.get("unused_tail_len_bytes"),
        "dna_roundtrip_ok": flags.get("dna_roundtrip_ok"),
        "zlib_integrity_ok": flags.get("zlib_integrity_ok"),
        "magic_detect_ok": flags.get("magic_detect_ok"),
        "restored_file": out.get("restored_file"),
    }
    return _md_kv(summary, title="Step-1 Run")

def _read_dna_text(dna_text: str, dna_file) -> str:
    s = ""
    if dna_text and dna_text.strip():
        s = dna_text.strip()
    elif dna_file is not None:
        # dna_file is UploadedFile
        s = dna_file.getvalue().decode("utf-8", errors="ignore")
    else:
        return ""
    if hasattr(dna_codec, "clean_dna_text"):
        return dna_codec.clean_dna_text(s)
    return "".join([c for c in s.upper() if c in "ACGT"])

def _save_uploaded_file(uploaded_file) -> str:
    """Save Streamlit UploadedFile to a temp path and return filepath."""
    ensure_dir(".streamlit_tmp_uploads")
    name = uploaded_file.name
    # sanitize name
    safe = "".join([c if c.isalnum() or c in "._-" else "_" for c in name])
    tmp_dir = os.path.join(".streamlit_tmp_uploads", str(uuid.uuid4()))
    ensure_dir(tmp_dir)
    path = os.path.join(tmp_dir, safe)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path

def _auto_pick_mode_for_file(path: str, raw: bytes) -> str:
    """
    Auto: choose Mode 3 if domain in {image,audio,video,text}; else Mode 2.
    We rely on extension primarily; if unknown, fallback to magic heuristics.
    """
    ext = (os.path.splitext(path)[1] or "").lower()
    image_ext = {".png",".jpg",".jpeg",".webp",".bmp",".gif",".tif",".tiff"}
    audio_ext = {".wav",".mp3",".flac",".ogg",".opus",".m4a",".aac",".wma"}
    video_ext = {".mp4",".mov",".mkv",".avi",".webm",".m4v"}
    text_ext  = {".txt",".csv",".tsv",".json",".xml",".html",".md",".py",".m",".c",".cpp",".h",".java",".js",".ts",".yaml",".yml",".ini",".log"}

    if ext in image_ext | audio_ext | video_ext | text_ext:
        return "mode3_domain"

    # fallback: magic sniff (images, gzip, zip, pdf, ogg, mp4, etc.)
    mk = detect_magic(raw)
    if mk and mk.kind in {"png","jpeg","webp","gif","wav","flac","ogg","opus_ogg","mp4","pdf","zip","gzip"}:
        # treat as mode3 only for media/text; zip/pdf/docx/others remain mode2
        if mk.kind in {"png","jpeg","webp","gif","wav","flac","ogg","opus_ogg","mp4","gzip"}:
            return "mode3_domain"
    return "mode2_zip_store"

def _run_step1_streamlit(
    uploaded_file,
    mode_choice: str,
    # dna codec params
    scheme_name: str,
    codec_mode: str,
    seed: str,
    init_dimer: str,
    prepend_one: bool,
    whiten: bool,
    target_gc: float,
    w_gc: float,
    w_motif: float,
    ks: Tuple[int,int],
    zlib_policy: str,
    # mode1 zip
    zip_level: int,
    # mode3 policies
    allow_ffmpeg: bool,
    image_policy: str,
    webp_quality: int,
    text_policy: str,
    audio_policy: str,
    opus_bitrate_kbps: int,
    video_policy: str,
    video_crf: int,
    quality_mode: str = "Lossy",
    benchmark_best: bool = True,
) -> Tuple[str, Dict[str, Any], str]:
    """
    Returns: (job_dir, report, dna_preview)
    """
    input_path = _save_uploaded_file(uploaded_file)
    raw = open(input_path, "rb").read()

    common_kwargs = dict(
        scheme_name=scheme_name,
        mode_codec=codec_mode,
        seed=seed,
        init_dimer=init_dimer,
        prepend_one=prepend_one,
        remove_leading_one=True,
        whiten=whiten,
        target_gc=float(target_gc),
        w_gc=float(w_gc),
        w_motif=float(w_motif),
        ks=(int(ks[0]), int(ks[1])),
        zlib_policy=zlib_policy,
    )

    # Resolve selection
    if mode_choice.startswith("Auto"):
        picked = _auto_pick_mode_for_file(input_path, raw)
        if picked == "mode3_domain":
            if benchmark_best:
                job_dir, report = run_mode3_best(
                input_path,
                image_policy=image_policy,
                webp_quality=int(webp_quality),
                text_policy=text_policy,
                allow_external_ffmpeg=bool(allow_ffmpeg),
                audio_policy=audio_policy,
                opus_bitrate_kbps=int(opus_bitrate_kbps),
                video_policy=video_policy,
                video_crf=int(video_crf),
                **common_kwargs,
                    quality_mode=quality_mode,
                )
            else:
                if benchmark_best:
                    job_dir, report = run_mode3_best(
                input_path,
                image_policy=image_policy,
                webp_quality=int(webp_quality),
                text_policy=text_policy,
                allow_external_ffmpeg=bool(allow_ffmpeg),
                audio_policy=audio_policy,
                opus_bitrate_kbps=int(opus_bitrate_kbps),
                video_policy=video_policy,
                video_crf=int(video_crf),
                **common_kwargs,
                        quality_mode=quality_mode,
                    )
                else:
                    job_dir, report = run_mode3_domain(
                input_path,
                image_policy=image_policy,
                webp_quality=int(webp_quality),
                text_policy=text_policy,
                allow_external_ffmpeg=bool(allow_ffmpeg),
                audio_policy=audio_policy,
                opus_bitrate_kbps=int(opus_bitrate_kbps),
                video_policy=video_policy,
                video_crf=int(video_crf),
                **common_kwargs,
                    )
                
            
        else:
            job_dir, report = run_mode2_zip_store(input_path, **common_kwargs)

    elif mode_choice.startswith("Branch 1") or mode_choice == "mode0_raw":
        job_dir, report = run_mode0_raw(input_path, **common_kwargs)
    elif mode_choice.startswith("Mode 1") or mode_choice == "mode1_zip":
        job_dir, report = run_mode1_zip(input_path, zip_level=int(zip_level), **common_kwargs)
    elif mode_choice.startswith("Mode 2") or mode_choice == "mode2_zip_store":
        job_dir, report = run_mode2_zip_store(input_path, **common_kwargs)
    else:
        job_dir, report = run_mode3_domain(
            input_path,
            image_policy=image_policy,
            webp_quality=int(webp_quality),
            text_policy=text_policy,
            allow_external_ffmpeg=bool(allow_ffmpeg),
            audio_policy=audio_policy,
            opus_bitrate_kbps=int(opus_bitrate_kbps),
            video_policy=video_policy,
            video_crf=int(video_crf),
            **common_kwargs,
        )

    dna_preview = ""
    dna_path = report.get("artifacts", {}).get("dna_txt")
    if dna_path and os.path.exists(dna_path):
        dna_preview = open(dna_path, "r", encoding="utf-8", errors="ignore").read().strip()[:12000]

    return job_dir, report, dna_preview

def _decode_step1_streamlit(
    dna_text: str,
    dna_file,
    scheme_name: str,
    codec_mode: str,
    seed: str,
    init_dimer: str,
    whiten: bool,
    remove_leading_one: bool,
    target_gc: float,
    w_gc: float,
    w_motif: float,
    ks: Tuple[int,int],
    preferred_stem: str,
) -> Tuple[Dict[str, Any], Optional[str]]:
    dna = _read_dna_text(dna_text, dna_file)
    if not dna:
        return {"error": "Provide DNA text or dna.txt"}, None

    bits, _digits = dna_codec.decode_dna_to_bits(
        dna,
        scheme_name=scheme_name,
        mode=codec_mode,
        seed=seed,
        init_dimer=init_dimer,
        remove_leading_one=bool(remove_leading_one),
        whiten=bool(whiten),
        target_gc=float(target_gc),
        w_gc=float(w_gc),
        w_motif=float(w_motif),
        ks=(int(ks[0]), int(ks[1])),
    )
    decoded_buf, pad_bits = bitstring_to_bytes(bits, pad_to_byte=True)

    inner, infl = zlib_inflate_until_eof(decoded_buf)
    z_ok = bool(infl.get("eof")) and (infl.get("error") is None)

    out_dir = os.path.join("recovery_out", str(uuid.uuid4()))
    ensure_dir(out_dir)
    stem = safe_basename(preferred_stem or "restored", fallback="restored")
    restored_file, restore_meta = restore_rep(inner, out_dir=out_dir, preferred_stem=stem)

    mk = detect_magic(inner)

    stats = {
        "dna": {"nt": len(dna), "bits_per_nt_est": (len(bits) / len(dna)) if len(dna) else None},
        "decoded": {"bits_len": len(bits), "bytes_len": len(decoded_buf), "pad_bits_to_byte": pad_bits},
        "zlib": {
            "eof": bool(infl.get("eof")),
            "unused_tail_len_bytes": int(infl.get("unused_tail_len", 0) or 0),
            "error": infl.get("error"),
            "integrity_ok": z_ok,
        },
        "inner": {"bytes_len": len(inner), "magic": (mk.kind if mk else None)},
        "output": {"restored_file": restored_file, "restore_meta": restore_meta},
    }

    return stats, (restored_file if z_ok else None)

# ----------------------------
# UI
# ----------------------------
def render_designing():
    st.header("DNA Data Design & Pipeline Management")

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "history" not in st.session_state:
        st.session_state.history = []  # list of reports

    tab_enc, tab_wet, tab_dec, tab_ana = st.tabs([
        "1. Encoding & Design", "2. Wet-lab Simulation", "3. Decoding & Retrieval", "4. Analysis"
    ])

    # ========== TAB 1: Encoding ==========
    with tab_enc:
        sub_raw, sub_comp = st.tabs(["Raw Data (Branch 1)", "Compressed Data (Mode 1/2/3 + Auto)"])

        # ---- Shared config controls ----
        def codec_controls(prefix: str):
            with st.expander("DNA codec params (dna_codec.py)", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                scheme_name = c1.selectbox("scheme_name", ["R0_B9","R1_B12","R2_B15","RINF_B16"], index=3, key=f"{prefix}_scheme")
                codec_mode  = c2.selectbox("codec_mode", ["TABLE","ALGO"], index=0, key=f"{prefix}_codec", help="TABLE: simple mapping. ALGO: rule-based / scoring encoder.")
                seed        = c3.text_input("seed", value="rn", key=f"{prefix}_seed")
                init_dimer  = c4.text_input("init_dimer", value="TA", key=f"{prefix}_init")
                c5, c6, c7 = st.columns(3)
                prepend_one = c5.checkbox("prepend_one", value=True, key=f"{prefix}_prepend_one")
                whiten      = c6.checkbox("whiten", value=True, key=f"{prefix}_whiten")
                target_gc   = c7.slider("target_gc", 0.35, 0.65, value=0.50, step=0.01, key=f"{prefix}_tgc")
                c8, c9, c10 = st.columns(3)
                w_gc     = c8.slider("w_gc", 0.0, 5.0, value=2.0, step=0.1, key=f"{prefix}_wgc")
                w_motif  = c9.slider("w_motif", 0.0, 5.0, value=1.0, step=0.1, key=f"{prefix}_wmotif")
                ks0      = c10.slider("ks[0]", 2, 8, value=4, step=1, key=f"{prefix}_k0")
                ks1      = c10.slider("ks[1]", 2, 10, value=6, step=1, key=f"{prefix}_k1")
            with st.expander("Headerless framing (ZLIB wrapper)", expanded=False):
                zlib_policy = st.selectbox("zlib_policy", ["auto","stored","compress"], index=0, key=f"{prefix}_zlib")
            return scheme_name, codec_mode, seed, init_dimer, prepend_one, whiten, target_gc, w_gc, w_motif, (ks0, ks1), zlib_policy

        # ---- Raw branch ----
        with sub_raw:
            st.subheader("Branch 1: RAW (no domain compression; still headerless via ZLIB framing)")
            col_input, col_process = st.columns([1, 2])

            with col_input:
                st.info("Step 1: Input Source")
                up = st.file_uploader(
                    "Upload File",
                    type=None,
                    key=f"raw_u_{st.session_state.uploader_key}"
                )
                st.write("—")
                scheme_name, codec_mode, seed, init_dimer, prepend_one, whiten, target_gc, w_gc, w_motif, ks, zlib_policy = codec_controls("raw")

                if st.button("RUN ENCODING (Branch 1)", type="primary", use_container_width=True, disabled=(up is None)):
                    st.session_state["run_raw"] = True

                if st.button("RESET PIPELINE", use_container_width=True):
                    st.session_state.uploader_key += 1
                    st.session_state["run_raw"] = False
                    st.rerun()

            with col_process:
                st.info("Step 2: Results")
                report_box = st.container(border=True)
                dna_box = st.container(border=True)
                out_box = st.container(border=True)

            if up and st.session_state.get("run_raw"):
                try:
                    job_dir, report, dna_preview = _run_step1_streamlit(
                        up, "mode0_raw",
                        scheme_name, codec_mode, seed, init_dimer, prepend_one, whiten, target_gc, w_gc, w_motif, ks, zlib_policy,
                        zip_level=6,
                        allow_ffmpeg=False,
                        image_policy="keep",
                        webp_quality=80,
                        text_policy="keep",
                        audio_policy="keep",
                        opus_bitrate_kbps=64,
                        video_policy="keep",
                        video_crf=28,
                    )
                    st.session_state.history = [report]  # keep only latest job

                    with report_box:
                        st.markdown(_render_step1_report(report))
                        st.code(json.dumps(report, indent=2, ensure_ascii=False), language="json")

                    with dna_box:
                        st.write("DNA preview (first 12k nt)")
                        st.text_area("DNA", value=dna_preview, height=220)
                        dna_path = report.get("artifacts", {}).get("dna_txt")
                        rep_path = report.get("artifacts", {}).get("inner_bin")
                        rep = open(rep_path, "rb").read() if rep_path and os.path.exists(rep_path) else b""
                        st.download_button("Download dna.txt", data=open(dna_path,"rb").read() if dna_path and os.path.exists(dna_path) else b"", file_name="dna.txt")
                        st.download_button("Download report.json", data=json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"), file_name="report.json")
                        st.download_button("Download inner.bin", data=rep, file_name="inner.bin")

                    with out_box:
                        restored = report.get("output", {}).get("restored_file")
                        if restored and os.path.exists(restored):
                            ext = (os.path.splitext(restored)[1] or "").lower()
                            st.write(f"Restored file: `{restored}`")
                            st.download_button("Download restored file", data=open(restored,"rb").read(), file_name=os.path.basename(restored))
                            if ext in {".png",".jpg",".jpeg",".webp",".gif",".bmp",".tif",".tiff"}:
                                st.image(restored, caption="Preview", width=220)
                            elif ext in {".wav",".mp3",".flac",".ogg",".opus",".m4a",".aac"}:
                                st.audio(restored)
                            elif ext in {".mp4",".mov",".mkv",".avi",".webm"}:
                                st.video(restored)
                        else:
                            st.write("No restored output.")
                except Exception as e:
                    st.error(f"Error: {e}")

        # ---- Compressed modes + Auto ----
        with sub_comp:
            st.subheader("Step-1 Universal (Mode 1/2/3) + Auto")
            col_input, col_process = st.columns([1, 2])

            with col_input:
                st.info("Step 1: Input Source")
                up = st.file_uploader(
                    "Upload File",
                    type=None,
                    key=f"comp_u_{st.session_state.uploader_key}"
                )

                mode_choice = st.selectbox(
                    "Mode",
                    [
                        "Auto (Mode3 for image/audio/video/text; else Mode2)",
                        "Mode 1: ZIP (lossless baseline)",
                        "Mode 2: ZIP STORE (lossless self-describing)",
                        "Mode 3: Domain-aware (auto, can be lossy)",
                    ],
                    index=0,
                    key="comp_mode",
                )

                scheme_name, codec_mode, seed, init_dimer, prepend_one, whiten, target_gc, w_gc, w_motif, ks, zlib_policy = codec_controls("comp")

                with st.expander("Mode 1 settings (ZIP)", expanded=False):
                    zip_level = st.slider("zip_level", 0, 9, value=6, step=1, key="comp_zip_level")

                with st.expander("Mode 3 policies (domain-aware)", expanded=True):
                    allow_ffmpeg = st.checkbox("allow_ffmpeg (audio/video)", value=True, key="comp_allow_ffmpeg")
                    quality_mode_ui = st.radio("Compression mode", ["Lossless","Lossy"], horizontal=True, index=1, key="comp_quality_mode", help="Lossless: choose only lossless representations (e.g., PNG/FLAC). Lossy: may transcode (e.g., WebP/Opus/H.264) to reduce size.")
                    benchmark_best_ui = st.checkbox("Benchmark (try multiple codecs and pick smallest)", value=True, key="comp_benchmark_best", help="Tries multiple codecs for the detected domain and picks the smallest representation before DNA.")
                    c1, c2, c3 = st.columns(3)
                    image_policy = c1.selectbox("image_policy", ["webp_lossy","webp_lossless","png_lossless","keep"], index=0, key="comp_img_policy")
                    webp_quality = c2.slider("webp_quality (lossy)", 10, 100, value=80, step=1, key="comp_webp_q")
                    text_policy  = c3.selectbox("text_policy", ["gzip","keep"], index=0, key="comp_text_policy")
                    c4, c5, c6, c7 = st.columns(4)
                    audio_policy = c4.selectbox("audio_policy", ["opus_ogg","flac_lossless","keep"], index=0, key="comp_audio_policy")
                    opus_bitrate = c5.slider("opus_bitrate_kbps (lossy)", 16, 192, value=64, step=1, key="comp_opus_br")
                    video_policy = c6.selectbox("video_policy", ["mp4_h264","webm_vp9","mkv_av1","keep"], index=0, key="comp_video_policy", help="Used when Benchmark is OFF. When Benchmark is ON, the app tries multiple video codecs and picks the smallest.")
                    video_crf    = c7.slider("video_crf (lossy)", 18, 40, value=28, step=1, key="comp_video_crf")

                if st.button("RUN ENCODING (Universal)", type="primary", use_container_width=True, disabled=(up is None)):
                    st.session_state["run_comp"] = True

            with col_process:
                st.info("Step 2: Results")
                report_box = st.container(border=True)
                dna_box = st.container(border=True)
                out_box = st.container(border=True)

            if up and st.session_state.get("run_comp"):
                try:
                    job_dir, report, dna_preview = _run_step1_streamlit(
                        up, mode_choice,
                        scheme_name, codec_mode, seed, init_dimer, prepend_one, whiten, target_gc, w_gc, w_motif, ks, zlib_policy,
                        zip_level=zip_level,
                        allow_ffmpeg=allow_ffmpeg,
                        image_policy=image_policy,
                        webp_quality=webp_quality,
                        text_policy=text_policy,
                        audio_policy=audio_policy,
                        opus_bitrate_kbps=opus_bitrate,
                        video_policy=video_policy,
                        video_crf=video_crf,
                                            quality_mode=st.session_state.get("comp_quality_mode","Lossy"),
                        benchmark_best=st.session_state.get("comp_benchmark_best", True),
                    )
                    st.session_state.history = [report]  # keep only latest job

                    with report_box:
                        st.markdown(_render_step1_report(report))
                        st.code(json.dumps(report, indent=2, ensure_ascii=False), language="json")

                    with dna_box:
                        st.write("DNA preview (first 12k nt)")
                        st.text_area("DNA", value=dna_preview, height=220)
                        dna_path = report.get("artifacts", {}).get("dna_txt")
                        st.download_button("Download dna.txt", data=open(dna_path,"rb").read() if dna_path and os.path.exists(dna_path) else b"", file_name="dna.txt")
                        st.download_button("Download report.json", data=json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"), file_name="report.json")

                    with out_box:
                        restored = report.get("output", {}).get("restored_file")
                        if restored and os.path.exists(restored):
                            ext = (os.path.splitext(restored)[1] or "").lower()
                            st.write(f"Restored file: `{restored}`")
                            st.download_button("Download restored file", data=open(restored,"rb").read(), file_name=os.path.basename(restored))
                            if ext in {".png",".jpg",".jpeg",".webp",".gif",".bmp",".tif",".tiff"}:
                                st.image(restored, caption="Preview", width=260)
                            elif ext in {".wav",".mp3",".flac",".ogg",".opus",".m4a",".aac"}:
                                st.audio(restored)
                            elif ext in {".mp4",".mov",".mkv",".avi",".webm"}:
                                st.video(restored)
                        else:
                            st.write("No restored output.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ========== TAB 2: Wet-lab Simulation ==========
    with tab_wet:
        st.info("Wet-lab simulation will be integrated here (Step 2/3). For now, Step-1 is ready.")

    # ========== TAB 3: Decoding ==========
    with tab_dec:
        st.subheader("Decode: DNA → file (headerless via zlib framing + magic routing)")
        c1, c2 = st.columns([2,1])
        with c1:
            dna_text = st.text_area("Paste DNA (A/C/G/T)", value="", height=200, placeholder="Paste the single DNA string here…")
        with c2:
            dna_file = st.file_uploader("...or upload dna.txt", type=["txt","fasta","fa"], key=f"dna_u_{st.session_state.uploader_key}")
        with st.expander("DNA codec params (must match encoding)", expanded=False):
            d1, d2, d3, d4 = st.columns(4)
            scheme_name = d1.selectbox("scheme_name", ["R0_B9","R1_B12","R2_B15","RINF_B16"], index=3, key="dec_scheme")
            codec_mode  = d2.selectbox("codec_mode", ["TABLE","RT"], index=0, key="dec_codec")
            seed        = d3.text_input("seed", value="rn", key="dec_seed")
            init_dimer  = d4.text_input("init_dimer", value="TA", key="dec_init")
            d5, d6, d7 = st.columns(3)
            whiten      = d5.checkbox("whiten", value=True, key="dec_whiten")
            remove_one  = d6.checkbox("remove_leading_one", value=True, key="dec_remove_one")
            target_gc   = d7.slider("target_gc", 0.35, 0.65, value=0.50, step=0.01, key="dec_tgc")
            d8, d9, d10 = st.columns(3)
            w_gc     = d8.slider("w_gc", 0.0, 5.0, value=2.0, step=0.1, key="dec_wgc")
            w_motif  = d9.slider("w_motif", 0.0, 5.0, value=1.0, step=0.1, key="dec_wmotif")
            ks0      = d10.slider("ks[0]", 2, 8, value=4, step=1, key="dec_k0")
            ks1      = d10.slider("ks[1]", 2, 10, value=6, step=1, key="dec_k1")
        preferred_stem = st.text_input("preferred output name (stem)", value="restored", key="dec_stem")

        if st.button("RECOVER + STATS", type="primary"):
            try:
                stats, restored = _decode_step1_streamlit(
                    dna_text, dna_file,
                    scheme_name, codec_mode, seed, init_dimer,
                    whiten, remove_one,
                    target_gc, w_gc, w_motif,
                    (ks0, ks1),
                    preferred_stem,
                )
                st.code(json.dumps(stats, indent=2, ensure_ascii=False), language="json")
                if restored and os.path.exists(restored):
                    ext = (os.path.splitext(restored)[1] or "").lower()
                    st.success(f"Recovered: {restored}")
                    st.download_button("Download recovered file", data=open(restored,"rb").read(), file_name=os.path.basename(restored))
                    if ext in {".png",".jpg",".jpeg",".webp",".gif",".bmp",".tif",".tiff"}:
                        st.image(restored, caption="Preview", width=320)
                    elif ext in {".wav",".mp3",".flac",".ogg",".opus",".m4a",".aac"}:
                        st.audio(restored)
                    elif ext in {".mp4",".mov",".mkv",".avi",".webm"}:
                        st.video(restored)
                else:
                    st.warning("ZLIB integrity failed or no output file. Check stats['zlib'].")
            except Exception as e:
                st.error(f"Error: {e}")

    # ========== TAB 4: Analysis ==========
    with tab_ana:
        st.subheader("Analysis")
        st.caption("Shows: input bits and compressed bits (binary right before DNA), plus quality metrics when applicable. Keeps only the latest job.")
        
        if not st.session_state.history:
            st.info("No runs yet.")
        else:
            r = st.session_state.history[-1]
            job = str(r.get("job_uuid") or "")[:8]
            inp = (r.get("input", {}) or {})
            zlb = (r.get("zlib_stream", {}) or {})
            rep = (r.get("rep", {}) or {})
            rep_meta = (rep.get("meta", {}) or {}) if isinstance(rep, dict) else {}
            out = (r.get("output", {}) or {})
            art = (r.get("artifacts", {}) or {})
        
            input_bits = int(inp.get("size_bytes") or 0) * 8
            compressed_bits = int(zlb.get("size_bytes") or 0) * 8
            ratio = (compressed_bits / input_bits) if input_bits else None
        
            domain = rep_meta.get("domain") or rep_meta.get("detected_domain")
            lossy = rep_meta.get("lossy")
            mode = r.get("mode")
        
            st.markdown(f"**Latest Job {job}** | mode=`{mode}` | domain=`{domain}` | lossy=`{lossy}`")
        
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Input (bits)", f"{input_bits:,}")
            with c2:
                st.metric("Compressed binary before DNA (bits)", f"{compressed_bits:,}")
            with c3:
                st.metric("Ratio (Compressed/Input)", "N/A" if ratio is None else f"{ratio:.4f}")
        
            g1, g2 = st.columns([1.35, 1.0])
            with g1:
                st.plotly_chart(_plot_bits_grouped(job, input_bits, compressed_bits), use_container_width=True)
            with g2:
                st.plotly_chart(_plot_compression_dial(job, input_bits, compressed_bits), use_container_width=True)
        
            in_path = art.get("input_original")
            out_path = out.get("restored_file")
        
            if isinstance(in_path, str) and isinstance(out_path, str) and os.path.exists(in_path) and os.path.exists(out_path):
                ext_in = (os.path.splitext(in_path)[1] or "").lower()
                ext_out = (os.path.splitext(out_path)[1] or "").lower()
        
                img_ext = {".png",".jpg",".jpeg",".webp",".bmp",".gif",".tif",".tiff"}
                if ext_in in img_ext and ext_out in img_ext:
                    psnr, ssim_val = _compute_psnr_ssim(in_path, out_path)
        
                    st.markdown("### Image quality")
                    q1, q2 = st.columns(2)
                    with q1:
                        st.metric("PSNR (dB)", "N/A" if psnr is None else f"{psnr:.3f}")
                    with q2:
                        st.metric("SSIM", "N/A" if ssim_val is None else f"{ssim_val:.5f}")
        
                    if psnr is not None and ssim_val is not None:
                        st.plotly_chart(_plot_psnr_ssim_combo(job, float(psnr), float(ssim_val)), use_container_width=True)
                else:
                    text_ext = {".txt",".md",".csv",".json",".yaml",".yml",".py",".java",".cpp",".c",".js",".ts",".html",".css"}
                    if ext_in in text_ext and ext_out in text_ext:
                        try:
                            a = open(in_path, "rb").read().decode("utf-8", "replace")
                            b = open(out_path, "rb").read().decode("utf-8", "replace")
                            a_words = a.split()
                            b_words = b.split()
        
                            nA, nB = len(a_words), len(b_words)
                            nmax = max(nA, nB, 1)
                            same = sum(1 for i in range(min(nA, nB)) if a_words[i] == b_words[i])
                            acc = same / nmax
                            dist = _word_edit_distance(a_words, b_words)
                            wer = dist / max(1, nA)
        
                            st.markdown("### Text quality (word-by-word)")
                            t1, t2, t3 = st.columns(3)
                            with t1:
                                st.metric("Words (input)", f"{nA:,}")
                            with t2:
                                st.metric("Words (restored)", f"{nB:,}")
                            with t3:
                                st.metric("Word accuracy", f"{acc:.4f}")
                            st.write(f"Word edit distance: **{dist}**, WER: **{wer:.4f}**")
        
                            diffs = []
                            for i in range(min(nA, nB)):
                                if a_words[i] != b_words[i]:
                                    diffs.append({"idx": i, "input": a_words[i], "restored": b_words[i]})
                                    if len(diffs) >= 30:
                                        break
                            if diffs:
                                st.dataframe(pd.DataFrame(diffs), use_container_width=True)
                            else:
                                st.success("No word-level differences in the first aligned segment.")
                        except Exception as _e:
                            st.info(f"Text comparison not available: {_e}")
                    else:
                        av_ext = {".wav",".mp3",".flac",".ogg",".opus",".m4a",".aac",".mp4",".mov",".mkv",".avi",".webm"}
                        if ext_in in av_ext or ext_out in av_ext:
                            st.info("Audio/Video quality: PSNR/SSIM not applicable. Video metrics will be added later.")
            else:
                st.info("Quality metrics require both input_original and restored_file paths to exist for this job.")
