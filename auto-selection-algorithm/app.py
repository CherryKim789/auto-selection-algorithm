from __future__ import annotations

import bz2
import difflib
import gzip
import io
import json
import lzma
import math
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import numpy as np
except Exception:
    np = None

import dna_codec
from dna_storage_core import (
    bytes_to_bitstring,
    decode_dna_sequence,
    encode_file,
    read_bytes,
    sha256_bytes,
)
from ui_adapters import (
    display_file_preview,
    save_uploaded_file,
    validate_uploaded_file,
)

APP_TITLE = "Automatic Selection for Compression and DNA Design"
SEED = "ddss-seed"
STRATEGY = "hierarchical_full_automatic"

INPUT_IMAGE_WIDTH = 340
OUTPUT_IMAGE_WIDTH = 340
SHOW_IMAGE_FULL_WIDTH = False
DNA_PREVIEW_CHARS = 12000

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root {
    --green-50: #f0fdf4;
    --green-100: #dcfce7;
    --green-200: #bbf7d0;
    --green-300: #86efac;
    --green-500: #22c55e;
    --green-600: #16a34a;
    --green-700: #15803d;
    --green-800: #166534;
    --slate-50: #f8fafc;
    --slate-100: #f1f5f9;
    --slate-200: #e2e8f0;
    --slate-300: #cbd5e1;
    --slate-500: #64748b;
    --slate-700: #334155;
    --slate-900: #0f172a;
}

html, body, [class*="css"] {
    color: var(--slate-900);
}

.block-container {
    padding-top: 3.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1120px !important;
}

section[data-testid="stSidebar"] {
    display: none;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
    border: 1px solid rgba(22, 101, 52, 0.12) !important;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98)) !important;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05) !important;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(22, 101, 52, 0.10);
    border-radius: 14px;
    padding: 0.55rem 0.7rem;
    background: rgba(255,255,255,0.88);
}

div[data-testid="stMetricLabel"] p {
    font-size: 13px !important;
    color: var(--slate-500) !important;
}

div[data-testid="stMetricValue"] {
    font-size: 20px !important;
    font-weight: 780 !important;
}

.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    min-height: 2.8rem !important;
    width: 100% !important;
    border: 1px solid rgba(22, 101, 52, 0.18) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, var(--green-600), var(--green-700)) !important;
    color: white !important;
}

.stButton > button:hover {
    border-color: rgba(22, 101, 52, 0.30) !important;
}

div[data-testid="stFileUploader"] section {
    border: 2px dashed rgba(34, 197, 94, 0.48) !important;
    border-radius: 18px !important;
    background: linear-gradient(180deg, rgba(240,253,244,0.96), rgba(248,250,252,0.96)) !important;
    padding: 1rem !important;
}

div[data-testid="stFileUploader"] button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 0.45rem 1rem !important;
}

div[data-testid="stFileUploader"] small {
    color: var(--slate-500) !important;
}

div[data-testid="stFileUploaderFile"] {
    display: none !important;
}

div[data-testid="stFileUploader"] ul {
    display: none !important;
}

div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] {
    display: none !important;
}

.app-title-wrap {
    margin-top: 0.2rem;
    margin-bottom: 0.9rem;
    padding: 0.15rem 0 0.25rem 0;
}

.app-title {
    font-size: 2rem;
    line-height: 1.2;
    font-weight: 800;
    color: var(--slate-900);
    max-width: 780px;
    margin: 0;
    word-break: normal;
    overflow-wrap: break-word;
}

.app-title-accent {
    width: 110px;
    height: 4px;
    border-radius: 999px;
    margin-top: 0.55rem;
    background: linear-gradient(90deg, var(--green-500), var(--green-700));
}

.preview-shell {
    min-height: 122px;
    display: flex;
    align-items: stretch;
}

.preview-note,
.empty-box,
.file-name-box {
    border-radius: 14px;
    border: 1px solid rgba(22, 101, 52, 0.10);
    background: linear-gradient(180deg, rgba(240,253,244,0.96), rgba(236,253,245,0.88));
    padding: 0.9rem 1rem;
    color: var(--green-800);
}

.file-name-box {
    min-height: 52px;
    display: flex;
    align-items: center;
    background: rgba(241,245,249,0.82);
    border-color: rgba(15, 23, 42, 0.12);
    color: var(--slate-700);
    word-break: break-word;
    margin-top: 0.5rem;
}

.section-gap {
    margin-top: 0.55rem;
    margin-bottom: 0.45rem;
}

.decode-radio-wrap {
    padding-top: 0.15rem;
    padding-bottom: 0.15rem;
}

div[role="radiogroup"] {
    gap: 1rem !important;
}

hr.soft-divider {
    border: none;
    border-top: 1px solid rgba(22, 101, 52, 0.10);
    margin: 0.75rem 0 0.4rem 0;
}

.result-caption {
    color: var(--slate-500);
    font-size: 0.93rem;
}

.tight-subheader {
    margin-top: 0.2rem;
}

.top-card-run-gap {
    margin-top: 0.7rem;
}

.bottom-card {
    margin-top: 0.05rem;
}

div[data-testid="stMarkdownContainer"] p {
    word-break: break-word;
}
</style>
""",
    unsafe_allow_html=True,
)


def clean_dna_text(text: str) -> str:
    return re.sub(r"[^ACGTacgt]", "", text or "").upper()


def load_uploaded_text(uploaded: Any) -> str:
    if uploaded is None:
        return ""
    return uploaded.getvalue().decode("utf-8", errors="ignore")


def fmt_bytes(n: Any) -> str:
    if n is None:
        return "—"
    try:
        x = float(n)
    except Exception:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{int(x)} B" if unit == "B" else f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} TB"


def get_nested(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def method_from_report(report: dict[str, Any]) -> str:
    selected = report.get("compression", {}).get("selected", {}) or {}
    package = report.get("package", {}) or {}
    return (
        selected.get("method_name")
        or selected.get("method")
        or selected.get("policy")
        or package.get("method_name")
        or "—"
    )


def extension_from_report(report: dict[str, Any]) -> str:
    selected = report.get("compression", {}).get("selected", {}) or {}
    package = report.get("package", {}) or {}
    ext = (
        selected.get("restore_ext")
        or selected.get("ext")
        or package.get("restore_ext")
        or package.get("ext")
        or "—"
    )
    return str(ext)


def dna_rule_from_report(report: dict[str, Any]) -> str:
    selected = report.get("dna_design", {}).get("selected", {}) or {}
    rule = selected.get("rule_name") or selected.get("scheme_name") or "—"
    init = selected.get("init_dimer")
    if rule == "SIMPLE":
        rule = "Simple Mapping"
    if rule == "Simple Mapping":
        return "Simple Mapping"
    return f"{rule} / {init}" if init else str(rule)


def dna_score_from_report(report: dict[str, Any]) -> str:
    selected = report.get("dna_design", {}).get("selected", {}) or {}
    score = selected.get("score")
    return "—" if score is None else f"{float(score):.4f}"


def compression_score_from_report(report: dict[str, Any]) -> str:
    selected = report.get("compression", {}).get("selected", {}) or {}
    score = selected.get("score")
    return "—" if score is None else f"{float(score):.4f}"


def encoded_file_size_from_report(report: dict[str, Any]) -> str:
    selected = report.get("compression", {}).get("selected", {}) or {}
    package = report.get("package", {}) or {}
    size = (
        selected.get("size_bytes")
        or selected.get("rep_size_bytes")
        or selected.get("representation_len_bytes")
        or package.get("representation_len_bytes")
    )
    return fmt_bytes(size)


def payload_binary_from_report(report: dict[str, Any]) -> tuple[str, str]:
    artifacts = report.get("artifacts", {}) or {}
    payload_path = artifacts.get("representation_bin")
    if payload_path and Path(payload_path).exists():
        payload_bytes = read_bytes(Path(payload_path))
        return bytes_to_bitstring(payload_bytes), "selected_payload_bits.txt"
    return "", "selected_payload_bits.txt"


def dna_stats(dna: str, report: dict[str, Any] | None = None) -> dict[str, Any]:
    report = report or {}
    selected = report.get("dna_design", {}).get("selected", {}) or {}
    hp = selected.get("homopolymer") or dna_codec.homopolymer_stats(dna)
    gc = selected.get("gc_fraction")
    if gc is None:
        gc = dna_codec.gc_content(dna)
    return {
        "dna_len": len(dna or ""),
        "gc": float(gc or 0.0),
        "hp_longest": int(hp.get("longest", 0) or 0),
        "hp_count": int(hp.get("count_ge2", hp.get("homo_count", 0)) or 0),
    }


def reset_decode_state() -> None:
    for key in (
        "decode_result",
        "restored_path",
        "decode_error",
        "uploaded_dna_clean",
        "uploaded_dna_name",
    ):
        st.session_state.pop(key, None)


def run_encoding(input_path: Path) -> None:
    work_root = Path(tempfile.mkdtemp(prefix="dna_clean_ui_"))
    out_dir = work_root / "run_output"

    result = encode_file(
        input_path=input_path,
        output_dir=out_dir,
        operation_mode="content_preserving",
        strategy=STRATEGY,
        seed=SEED,
    )

    st.session_state["work_root"] = str(work_root)
    st.session_state["input_path"] = str(input_path)
    st.session_state["encode_result"] = result
    st.session_state["encode_report"] = result.get("report", {})
    st.session_state["encoded_dna"] = result.get("dna_sequence", "")
    st.session_state.pop("encode_error", None)
    reset_decode_state()


def run_decoding(dna_text: str, preferred_stem: str = "restored") -> None:
    work_root = Path(
        st.session_state.get("work_root")
        or tempfile.mkdtemp(prefix="dna_clean_ui_decode_")
    )
    decoded_dir = work_root / "decoded_output"

    result = decode_dna_sequence(
        dna_text=dna_text,
        output_dir=decoded_dir,
        preferred_stem=preferred_stem,
        seed=SEED,
        enable_blind_fallback=True,
        verify_only=False,
    )

    restored_path = (
        result.get("output_path")
        or get_nested(result, "report.restored.path")
        or get_nested(result, "report.output_path")
    )
    if not restored_path:
        raise RuntimeError("Decode finished but no output path was returned.")

    st.session_state["decode_result"] = result
    st.session_state["restored_path"] = str(Path(restored_path))
    st.session_state.pop("decode_error", None)


def metric_or_blank(label: str, value: Any) -> None:
    st.metric(label, "—" if value is None or value == "" else value)


def info_box(message: str) -> None:
    st.markdown(f"<div class='preview-note'>{message}</div>", unsafe_allow_html=True)


def filename_box(message: str) -> None:
    st.markdown(f"<div class='file-name-box'>{message}</div>", unsafe_allow_html=True)


def _read_restored_comparable_bytes(path: str | Path) -> tuple[bytes, str]:
    path = Path(path)
    data = read_bytes(path)
    suffix = path.suffix.lower()

    try:
        if suffix == ".gz":
            return gzip.decompress(data), "gunzip"
        if suffix == ".bz2":
            return bz2.decompress(data), "bunzip2"
        if suffix == ".xz":
            return lzma.decompress(data), "unxz"
        if suffix == ".zip":
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                members = [m for m in zf.namelist() if not m.endswith("/")]
                if members:
                    return zf.read(members[0]), f"unzip:{members[0]}"
    except Exception:
        pass

    return data, "direct"


def image_quality_metrics(original_path: str | Path, restored_path: str | Path) -> dict[str, Any]:
    if Image is None or np is None:
        return {"available": False, "reason": "Pillow or NumPy is not available."}

    try:
        original = Image.open(original_path).convert("RGB")
        restored = Image.open(restored_path).convert("RGB")

        if restored.size != original.size:
            restored = restored.resize(original.size)

        a = np.asarray(original).astype(np.float64)
        b = np.asarray(restored).astype(np.float64)

        mse = float(np.mean((a - b) ** 2))
        psnr = 99.0 if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse))

        scores: list[float] = []
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        for ch in range(3):
            x = a[:, :, ch]
            y = b[:, :, ch]
            mux = float(np.mean(x))
            muy = float(np.mean(y))
            vx = float(np.var(x))
            vy = float(np.var(y))
            cov = float(np.mean((x - mux) * (y - muy)))
            den = (mux * mux + muy * muy + c1) * (vx + vy + c2)
            ssim = 1.0 if abs(den) < 1e-12 else ((2 * mux * muy + c1) * (2 * cov + c2)) / den
            scores.append(float(ssim))

        return {
            "available": True,
            "psnr": float(psnr),
            "mse": float(mse),
            "ssim": float(sum(scores) / len(scores)),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def text_diff_preview(a: str, b: str, max_lines: int = 80) -> str:
    lines = list(
        difflib.unified_diff(
            a.splitlines(),
            b.splitlines(),
            fromfile="original",
            tofile="restored",
            lineterm="",
        )
    )
    return "\n".join(lines[:max_lines]) if lines else "No textual differences detected."


if "encode_result" not in st.session_state:
    st.session_state["encode_result"] = None
if "encode_report" not in st.session_state:
    st.session_state["encode_report"] = {}
if "encoded_dna" not in st.session_state:
    st.session_state["encoded_dna"] = ""

st.markdown(
    """
<div class="app-title-wrap">
    <div class="app-title">🧬 Automatic Selection for Compression and DNA Design</div>
    <div class="app-title-accent"></div>
</div>
""",
    unsafe_allow_html=True,
)

top_left, top_right = st.columns(2, gap="large")
bottom_left, bottom_right = st.columns(2, gap="large")

with top_left:
    with st.container(border=True):
        st.header("Encoding")

        input_path_str = st.session_state.get("input_path")
        preview_col = st.container()
        with preview_col:
            if input_path_str and Path(input_path_str).exists():
                display_file_preview(
                    input_path_str,
                    title="Input preview",
                    image_width=INPUT_IMAGE_WIDTH,
                    image_use_container_width=SHOW_IMAGE_FULL_WIDTH,
                )
            else:
                info_box("Upload a file to start.")

        if input_path_str and Path(input_path_str).exists():
            filename_box(Path(input_path_str).name)
        else:
            filename_box("No file selected")

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.subheader("Browse")

        uploaded_file = st.file_uploader(
            "Upload a file",
            label_visibility="collapsed",
            type=None,
            accept_multiple_files=False,
            key="encode_upload_file",
        )

        if uploaded_file is not None:
            validation = validate_uploaded_file(uploaded_file)
            if validation["accepted"]:
                st.caption(
                    f"{validation['filename']} · {validation['media_kind']} · {validation['size_mb']} MB"
                )
            else:
                for err in validation["errors"]:
                    st.error(err)
                for warn in validation["warnings"]:
                    st.warning(warn)

        st.markdown("<div class='top-card-run-gap'></div>", unsafe_allow_html=True)
        run_encode = st.button("Run", key="run_encode", type="primary", use_container_width=True)

        if run_encode:
            if uploaded_file is None:
                st.error("Please upload a file first.")
            else:
                validation = validate_uploaded_file(uploaded_file)
                if not validation["accepted"]:
                    st.error("Uploaded file is not accepted. Fix the validation errors first.")
                else:
                    try:
                        upload_root = Path(tempfile.mkdtemp(prefix="dna_ui_upload_"))
                        input_path = save_uploaded_file(uploaded_file, upload_root)
                        run_encoding(input_path)
                        st.rerun()
                    except Exception as exc:
                        st.session_state["encode_error"] = str(exc)
                        st.error(f"Encoding failed: {exc}")

        if st.session_state.get("encode_error"):
            st.error(st.session_state["encode_error"])

with top_right:
    with st.container(border=True):
        st.header("Decoding")

        restored_path = st.session_state.get("restored_path")
        if restored_path and Path(restored_path).exists():
            display_file_preview(
                restored_path,
                title="Decoded preview",
                image_width=OUTPUT_IMAGE_WIDTH,
                image_use_container_width=SHOW_IMAGE_FULL_WIDTH,
            )
        else:
            info_box("Decoded preview will appear here.")

        current_decode_source = st.session_state.get("decode_source_clean_ui", "From encoding")
        if current_decode_source == "From encoding":
            encoded_dna = st.session_state.get("encoded_dna", "")
            if encoded_dna:
                filename_box(f"{len(encoded_dna):,} nt available from encoding")
            else:
                filename_box("No encoded DNA available")
        else:
            uploaded_name = st.session_state.get("uploaded_dna_name")
            uploaded_clean = st.session_state.get("uploaded_dna_clean", "")
            if uploaded_name and uploaded_clean:
                filename_box(f"{uploaded_name} · {len(uploaded_clean):,} nt")
            elif uploaded_name:
                filename_box(f"{uploaded_name} · No valid A/C/G/T characters found")
            else:
                filename_box("No DNA file selected")

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.subheader("Browse")

        st.markdown("<div class='decode-radio-wrap'>", unsafe_allow_html=True)
        decode_source = st.radio(
            "Decode source",
            ["From encoding", "User Upload"],
            horizontal=True,
            label_visibility="collapsed",
            key="decode_source_clean_ui",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        decode_dna_text = ""
        preferred_stem = "restored"

        if decode_source == "From encoding":
            encoded_dna = st.session_state.get("encoded_dna", "")
            if encoded_dna:
                decode_dna_text = encoded_dna
                input_path = st.session_state.get("input_path")
                if input_path:
                    preferred_stem = Path(input_path).stem or "restored"
        else:
            uploaded_dna_file = st.file_uploader(
                "Upload DNA file",
                label_visibility="collapsed",
                type=["txt", "fasta", "fa", "dna"],
                accept_multiple_files=False,
                key="decode_upload_dna",
            )
            if uploaded_dna_file is not None:
                raw_text = load_uploaded_text(uploaded_dna_file)
                cleaned = clean_dna_text(raw_text)
                st.session_state["uploaded_dna_clean"] = cleaned
                st.session_state["uploaded_dna_name"] = uploaded_dna_file.name
                preferred_stem = Path(uploaded_dna_file.name).stem or "restored"
                if cleaned:
                    decode_dna_text = cleaned

        st.markdown("<div class='top-card-run-gap'></div>", unsafe_allow_html=True)
        run_decode = st.button("Run", key="run_decode", type="primary", use_container_width=True)

        if run_decode:
            if not decode_dna_text:
                st.error("Please provide DNA input first.")
            else:
                try:
                    run_decoding(decode_dna_text, preferred_stem=preferred_stem)
                    st.rerun()
                except Exception as exc:
                    st.session_state["decode_error"] = str(exc)
                    st.error(f"Decoding failed: {exc}")

        if st.session_state.get("decode_error"):
            st.error(st.session_state["decode_error"])

with bottom_left:
    with st.container(border=True):
        st.header("Encoding result")

        encode_result = st.session_state.get("encode_result")
        encode_report = st.session_state.get("encode_report", {})
        encoded_dna = st.session_state.get("encoded_dna", "")

        if not encode_result or not encoded_dna:
            info_box("Run encoding to show selected compressor and DNA design.")
        else:
            stats = dna_stats(encoded_dna, encode_report)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                metric_or_blank("Compressor", method_from_report(encode_report))
            with m2:
                metric_or_blank("Restore ext", extension_from_report(encode_report))
            with m3:
                metric_or_blank("DNA design", dna_rule_from_report(encode_report))
            with m4:
                metric_or_blank("DNA length", f"{stats['dna_len']:,} nt")

            m5, m6, m7, m8 = st.columns(4)
            with m5:
                metric_or_blank("Compression score", compression_score_from_report(encode_report))
            with m6:
                metric_or_blank("DNA score", dna_score_from_report(encode_report))
            with m7:
                metric_or_blank("GC content", f"{stats['gc'] * 100:.2f}%")
            with m8:
                metric_or_blank("Longest homopolymer", stats["hp_longest"])

            st.caption(f"Selected payload size: {encoded_file_size_from_report(encode_report)}")

            artifacts = encode_report.get("artifacts", {}) or {}
            dna_path = artifacts.get("dna_text")
            if dna_path and Path(dna_path).exists():
                st.download_button(
                    "Download DNA",
                    data=Path(dna_path).read_text(encoding="utf-8", errors="ignore"),
                    file_name=Path(dna_path).name,
                    mime="text/plain",
                    use_container_width=True,
                )
            else:
                st.download_button(
                    "Download DNA",
                    data=encoded_dna,
                    file_name="encoded_dna.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with st.expander("DNA preview", expanded=False):
                st.text_area(
                    "DNA sequence",
                    value=encoded_dna[:DNA_PREVIEW_CHARS],
                    height=240,
                    disabled=True,
                    label_visibility="collapsed",
                )

            payload_bits, payload_name = payload_binary_from_report(encode_report)
            if payload_bits:
                st.download_button(
                    "Download selected payload bits",
                    data=payload_bits,
                    file_name=payload_name,
                    mime="text/plain",
                    use_container_width=True,
                )

            with st.expander("Raw report", expanded=False):
                st.code(json.dumps(encode_report, indent=2, ensure_ascii=False), language="json")

with bottom_right:
    with st.container(border=True):
        st.header("Decoding result")

        decode_result = st.session_state.get("decode_result")
        restored_path_str = st.session_state.get("restored_path")
        input_path_str = st.session_state.get("input_path")

        if not decode_result or not restored_path_str or not Path(restored_path_str).exists():
            info_box("Run decoding to show restored file result.")
        else:
            restored_path = Path(restored_path_str)
            restored_bytes = read_bytes(restored_path)
            restored_sha = sha256_bytes(restored_bytes)

            n1, n2, n3 = st.columns(3)
            with n1:
                metric_or_blank("Restored file", restored_path.name)
            with n2:
                metric_or_blank("Size", fmt_bytes(len(restored_bytes)))
            with n3:
                metric_or_blank("SHA-256", f"{restored_sha[:12]}…")

            st.download_button(
                "Download restored file",
                data=restored_bytes,
                file_name=restored_path.name,
                mime="application/octet-stream",
                use_container_width=True,
            )

            display_file_preview(
                restored_path,
                title="Restored preview",
                image_width=OUTPUT_IMAGE_WIDTH,
                image_use_container_width=SHOW_IMAGE_FULL_WIDTH,
            )

            if input_path_str and Path(input_path_str).exists():
                original_path = Path(input_path_str)
                original_bytes = read_bytes(original_path)
                original_sha = sha256_bytes(original_bytes)

                c1, c2, c3 = st.columns(3)
                with c1:
                    metric_or_blank("Original size", fmt_bytes(len(original_bytes)))
                with c2:
                    metric_or_blank("Original SHA-256", f"{original_sha[:12]}…")
                with c3:
                    metric_or_blank("Exact bytes", "Yes" if original_sha == restored_sha else "No")

                if original_path.suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"
                }:
                    img_metrics = image_quality_metrics(original_path, restored_path)
                    if img_metrics.get("available"):
                        q1, q2, q3 = st.columns(3)
                        with q1:
                            metric_or_blank("PSNR", f"{img_metrics['psnr']:.3f}")
                        with q2:
                            metric_or_blank("SSIM", f"{img_metrics['ssim']:.4f}")
                        with q3:
                            metric_or_blank("MSE", f"{img_metrics['mse']:.3f}")

                try:
                    restored_cmp, mode = _read_restored_comparable_bytes(restored_path)
                    if original_bytes != restored_cmp and original_path.suffix.lower() in {
                        ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".py"
                    }:
                        original_text = original_bytes.decode("utf-8", errors="ignore")
                        restored_text = restored_cmp.decode("utf-8", errors="ignore")
                        with st.expander(f"Text diff ({mode})", expanded=False):
                            st.code(text_diff_preview(original_text, restored_text), language="diff")
                except Exception:
                    pass

            with st.expander("Raw decode report", expanded=False):
                st.code(json.dumps(decode_result, indent=2, ensure_ascii=False), language="json")