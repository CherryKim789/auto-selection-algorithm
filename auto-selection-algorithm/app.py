from __future__ import annotations

import json
import re
import tempfile
import difflib
import gzip
import bz2
import lzma
import zipfile
import io
import math
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
from dna_storage_core import decode_dna_sequence, encode_file, read_bytes, sha256_bytes, bytes_to_bitstring

from ui_adapters import (
    display_file_preview,
    save_uploaded_file,
    validate_uploaded_file,
    infer_media_kind,
)

# ============================================================
# Simple 4-block UI
# ============================================================

APP_TITLE = "DNA Storage Pipeline"
SEED = "ddss-seed"
STRATEGY = "hierarchical_full_automatic"

INPUT_IMAGE_WIDTH = 360
OUTPUT_IMAGE_WIDTH = 360
SHOW_IMAGE_FULL_WIDTH = False


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}
h1 {
    font-size: 30px !important;
    font-weight: 800 !important;
    margin-bottom: 0.4rem !important;
}
h2, h3 {
    font-weight: 760 !important;
}
section[data-testid="stSidebar"] {
    display: none;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
    border: 1px solid rgba(15, 23, 42, 0.12) !important;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.055);
}
div[data-testid="stMetric"] {
    border: 1px solid rgba(15, 23, 42, 0.10);
    border-radius: 14px;
    padding: 0.55rem 0.7rem;
    background: rgba(255,255,255,0.85);
}
div[data-testid="stMetricLabel"] p {
    font-size: 13px !important;
    color: #64748b !important;
}
div[data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 780 !important;
}
.stButton button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    height: 2.75rem;
}
.file-name-box {
    border: 1px solid rgba(15, 23, 42, 0.14);
    border-radius: 12px;
    padding: 0.65rem 0.8rem;
    background: rgba(241,245,249,0.80);
    font-size: 14px;
    word-break: break-all;
    margin: 0.35rem 0 0.8rem 0;
}
.muted {
    color: #64748b;
    font-size: 13px;
}
.result-box {
    border-radius: 14px;
    border: 1px solid rgba(15, 23, 42, 0.10);
    background: rgba(255,255,255,0.70);
    padding: 0.8rem 0.9rem;
}
/* Upload box */
div[data-testid="stFileUploader"] section {
    border: 2px dashed rgba(14, 165, 233, 0.45) !important;
    border-radius: 18px !important;
    background: linear-gradient(180deg, rgba(240,249,255,0.95), rgba(248,250,252,0.95)) !important;
    padding: 1rem !important;
}

/* Change Browse files button to Browse */
div[data-testid="stFileUploader"] button {
    font-size: 0 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}

div[data-testid="stFileUploader"] button::after {
    content: "Browse";
    font-size: 14px !important;
    font-weight: 700 !important;
}

/* Hide some default uploader instruction text if needed */
div[data-testid="stFileUploader"] small {
    color: #64748b !important;
}
/* Hide Streamlit default uploaded file row: filename + size */
div[data-testid="stFileUploaderFile"] {
    display: none !important;
}

/* Extra fallback for some Streamlit versions */
div[data-testid="stFileUploader"] ul {
    display: none !important;
}

div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] {
    display: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================

def clean_dna_text(text: str) -> str:
    return re.sub(r"[^ACGTacgt]", "", text or "").upper()


def load_uploaded_text(uploaded) -> str:
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


def payload_binary_from_report(report: dict[str, Any]) -> tuple[str, str]:
    """Return the selected compressed/self-describing payload as a 0/1 bitstring.

    This is the exact byte stream that is sent into DNA design, not the original file.
    """
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
    for key in ("decode_result", "restored_path", "decode_error"):
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
    reset_decode_state()


def run_decoding(dna_text: str, preferred_stem: str = "restored") -> None:
    work_root = Path(st.session_state.get("work_root") or tempfile.mkdtemp(prefix="dna_clean_ui_decode_"))
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


def _read_restored_comparable_bytes(path: str | Path) -> tuple[bytes, str]:
    """Return bytes suitable for original-vs-restored comparison.

    The decoded file is the self-describing payload. For plain image/text files,
    this is already directly comparable. For common lossless containers
    generated from text/binary data, this function unwraps the container before
    comparing with the original input.
    """
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
    """Compute lightweight PSNR/SSIM between original image and restored image."""
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

        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        ssim_scores = []

        for ch in range(3):
            x = a[:, :, ch]
            y = b[:, :, ch]

            mux = float(np.mean(x))
            muy = float(np.mean(y))
            vx = float(np.var(x))
            vy = float(np.var(y))
            cov = float(np.mean((x - mux) * (y - muy)))

            num = (2 * mux * muy + c1) * (2 * cov + c2)
            den = (mux * mux + muy * muy + c1) * (vx + vy + c2)
            ssim_scores.append(1.0 if abs(den) < 1e-12 else float(num / den))

        ssim = max(-1.0, min(1.0, float(np.mean(ssim_scores))))

        return {
            "available": True,
            "psnr": float(psnr),
            "ssim": float(ssim),
            "mse": float(mse),
            "reason": "",
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def text_accuracy_metrics(original_path: str | Path, restored_path: str | Path) -> dict[str, Any]:
    """Compare original text-like file with decoded/restored content.

    For compressed decoded payloads, common containers are decompressed first.
    """
    try:
        original_bytes = read_bytes(Path(original_path))
        restored_bytes, restore_mode = _read_restored_comparable_bytes(restored_path)

        original_text = original_bytes.decode("utf-8", errors="ignore")
        restored_text = restored_bytes.decode("utf-8", errors="ignore")

        exact_bytes = original_bytes == restored_bytes
        exact_text = original_text == restored_text
        similarity = difflib.SequenceMatcher(None, original_text, restored_text).ratio()

        if original_text:
            matching_chars = sum(1 for a, b in zip(original_text, restored_text) if a == b)
            char_accuracy = matching_chars / max(len(original_text), len(restored_text), 1)
        else:
            char_accuracy = 1.0 if not restored_text else 0.0

        return {
            "available": True,
            "restore_mode": restore_mode,
            "exact_bytes": bool(exact_bytes),
            "exact_text": bool(exact_text),
            "similarity": float(similarity),
            "char_accuracy": float(char_accuracy),
            "original_chars": len(original_text),
            "restored_chars": len(restored_text),
            "reason": "",
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "restore_mode": "unavailable",
            "exact_bytes": False,
            "exact_text": False,
            "similarity": 0.0,
            "char_accuracy": 0.0,
            "original_chars": 0,
            "restored_chars": 0,
        }


def render_recovery_analysis(input_path_value: str | None, restored_path: Path, decoded_ok: bool | None) -> None:
    """Render domain-specific quality/accuracy metrics after decoding."""
    st.markdown("#### Recovery Analysis")

    if not input_path_value or not Path(input_path_value).exists():
        st.info("Original input file is unavailable, so recovery metrics cannot be computed.")
        return

    input_path = Path(input_path_value)
    kind = infer_media_kind(input_path)

    if kind == "image":
        metrics = image_quality_metrics(input_path, restored_path)
        if metrics.get("available"):
            c1, c2, c3 = st.columns(3)
            c1.metric("PSNR", f"{metrics['psnr']:.2f} dB")
            c2.metric("SSIM", f"{metrics['ssim']:.4f}")
            c3.metric("MSE", f"{metrics['mse']:.2f}")
            # st.caption(
            #     "PSNR and SSIM compare the original input image with the decoded/restored image. "
            #     "For lossy compression, DNA decoding can be successful even when the restored image is not byte-identical to the original."
            # )
        else:
            st.info(f"Image metrics unavailable: {metrics.get('reason', 'unknown error')}")
        return

    if kind == "text":
        metrics = text_accuracy_metrics(input_path, restored_path)
        if metrics.get("available"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Text Accuracy", f"{metrics['char_accuracy'] * 100:.2f}%")
            c2.metric("Text Similarity", f"{metrics['similarity']:.4f}")
            c3.metric("Exact Text Match", "Passed" if metrics["exact_text"] else "Failed")

            c1, c2 = st.columns(2)
            c1.metric("Original Chars", f"{metrics['original_chars']:,}")
            c2.metric("Restored Chars", f"{metrics['restored_chars']:,}")

            st.caption(
                f"Text comparison mode: {metrics['restore_mode']}. "
                "For lossless text compression, exact text match should pass after decoding."
            )
        else:
            st.info(f"Text metrics unavailable: {metrics.get('reason', 'unknown error')}")
        return

    # Generic validation for audio/video/document/binary.
    restored_bytes = read_bytes(restored_path)
    input_bytes = read_bytes(input_path)
    exact_original = sha256_bytes(input_bytes) == sha256_bytes(restored_bytes)

    c1, c2, c3 = st.columns(3)
    c1.metric("Original Size", fmt_bytes(len(input_bytes)))
    c2.metric("Restored Size", fmt_bytes(len(restored_bytes)))
    c3.metric("Original SHA256", "Passed" if exact_original else "Transformed")

    st.caption(
        "For non-image/non-text files, this panel reports file-level validation. "
        "The primary DNA validation remains payload integrity."
    )


# ============================================================
# Header
# ============================================================

st.title("🧬 DNA Storage Pipeline")
# st.caption("Hierarchical pipeline: compression is selected first, then DNA design is selected from Simple Mapping and rule-based mappings with four initial dimers. Whitening is disabled.")


# ============================================================
# State
# ============================================================

if "input_path" not in st.session_state:
    st.session_state["input_path"] = None


# ============================================================
# Layout: four simple blocks
# ============================================================

top_left, top_right = st.columns(2, gap="large")
bottom_left, bottom_right = st.columns(2, gap="large")


# ============================================================
# Top-left: Encoding
# ============================================================

with top_left:
    with st.container(border=True):
        st.subheader("Encoding")

        input_path_value = st.session_state.get("input_path")
        if input_path_value and Path(input_path_value).exists():
            display_file_preview(
                Path(input_path_value),
                title="Input preview",
                image_width=INPUT_IMAGE_WIDTH,
                image_use_container_width=SHOW_IMAGE_FULL_WIDTH,
                show_file_meta=False,
                text_height=180,
            )
        else:
            st.info("Upload a file to start.")

        if input_path_value and Path(input_path_value).exists():
            input_file = Path(input_path_value)

            file_name = input_file.name
            file_size = fmt_bytes(input_file.stat().st_size)
            file_kind = infer_media_kind(input_file)
            file_ext = input_file.suffix.lower() or "no extension"

            st.markdown(
                f"""
                <div class='file-name-box'>
                    <div style='font-size:18px; font-weight:800; margin-bottom:6px;'>
                        {file_name}
                    </div>
                    <div style='font-size:15px; line-height:1.7;'>
                        <span class='muted'><b>Type:</b> {file_kind.capitalize()}</span><br>
                        <span class='muted'><b>Extension:</b> {file_ext.upper()}</span><br>
                        <span class='muted'><b>Uploaded Size:</b> {file_size}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class='file-name-box'>
                    No file selected
                </div>
                """,
                unsafe_allow_html=True,
            )        
        st.markdown("#### Browse")
        uploaded = st.file_uploader(
            "Upload input file",
            type=None,
            accept_multiple_files=False,
            key="encoding_browse",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            upload_sig = f"{uploaded.name}_{uploaded.size}"

            if st.session_state.get("last_upload_sig") != upload_sig:
                validation = validate_uploaded_file(uploaded)

                for err in validation.get("errors", []):
                    st.error(err)
                for msg in validation.get("warnings", []):
                    st.warning(msg)

                if validation.get("accepted", True):
                    work_root = Path(tempfile.mkdtemp(prefix="dna_clean_ui_upload_"))
                    input_path = save_uploaded_file(uploaded, work_root / "input")

                    st.session_state["input_path"] = str(input_path)
                    st.session_state["last_upload_sig"] = upload_sig

                    # Clear old results when a new file is uploaded
                    for key in (
                        "encode_result",
                        "encode_report",
                        "encoded_dna",
                        "decode_result",
                        "restored_path",
                        "decode_error",
                    ):
                        st.session_state.pop(key, None)
        run_encode = st.button(
            "Run",
            type="primary",
            use_container_width=True,
            key="btn_run_encoding_clean_ui",
        )

        if run_encode:
            input_path_value = st.session_state.get("input_path")
            if not input_path_value:
                st.error("Please browse and upload a file first.")
            else:
                try:
                    with st.spinner("Running compression and DNA design scoring..."):
                        run_encoding(Path(input_path_value))
                    st.success("Encoding completed.")
                except Exception as exc:
                    st.exception(exc)


# ============================================================
# Top-right: Decoding
# ============================================================

with top_right:
    with st.container(border=True):
        st.subheader("Decoding")

        if st.session_state.get("decode_success_message"):
            st.success(st.session_state.pop("decode_success_message"))

        restored_path_value = st.session_state.get("restored_path")
        if restored_path_value and Path(restored_path_value).exists():
            display_file_preview(
                Path(restored_path_value),
                title="Decoded preview",
                image_width=OUTPUT_IMAGE_WIDTH,
                image_use_container_width=SHOW_IMAGE_FULL_WIDTH,
                show_file_meta=False,
                text_height=180,
            )
        else:
            st.info("Decoded preview will appear here.")

        decode_source = st.radio(
            "Input",
            ["From encoding", "Browse DNA file"],
            horizontal=True,
            key="decode_source_clean_ui",
        )

        dna_to_decode = ""
        if decode_source == "From encoding":
            dna_to_decode = st.session_state.get("encoded_dna", "")
            if dna_to_decode:
                st.markdown(f"<div class='file-name-box'>Loaded from encoding · {len(dna_to_decode):,} nt</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='file-name-box'>No encoded DNA available</div>", unsafe_allow_html=True)
        else:
            uploaded_dna = st.file_uploader(
                "Browse",
                type=["txt", "dna", "fasta", "fa"],
                accept_multiple_files=False,
                key="decode_browse_dna",
            )
            if uploaded_dna is not None:
                dna_to_decode = clean_dna_text(load_uploaded_text(uploaded_dna))
                st.markdown(f"<div class='file-name-box'>{uploaded_dna.name} · {len(dna_to_decode):,} nt</div>", unsafe_allow_html=True)

        run_decode = st.button(
            "Run",
            type="primary",
            use_container_width=True,
            key="btn_run_decoding_clean_ui",
        )

        if run_decode:
            if not dna_to_decode:
                st.error("No DNA sequence is available for decoding.")
            else:
                try:
                    input_name = Path(st.session_state.get("input_path") or "restored").stem
                    with st.spinner("Decoding DNA and restoring file..."):
                        run_decoding(dna_to_decode, preferred_stem=input_name)
                    # Force one rerun so the decoded preview block above is refreshed immediately.
                    #st.session_state["decode_success_message"] = "Decoding completed."
                    st.rerun()
                except Exception as exc:
                    st.session_state["decode_error"] = str(exc)
                    st.exception(exc)


# ============================================================
# Bottom-left: Encoding results
# ============================================================

with bottom_left:
    with st.container(border=True):
        st.subheader("Encoding result")

        result = st.session_state.get("encode_result")
        report = st.session_state.get("encode_report", {})
        dna = st.session_state.get("encoded_dna", "")

        if not result:
            st.info("Run encoding to show selected compressor and DNA design.")
        else:
            stats = dna_stats(dna, report)

 

            c1, c2 = st.columns(2)

            with c1:
                metric_or_blank("Compression Method", method_from_report(report))

            with c2:
                metric_or_blank("Encoded File Type", extension_from_report(report))

            c1, c2 = st.columns(2)
            with c1:
                metric_or_blank("DNA design rule", dna_rule_from_report(report))
            with c2:
                metric_or_blank("DNA score", dna_score_from_report(report))

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_or_blank("Encoded File Size", encoded_file_size_from_report(report))
            with c2:
                metric_or_blank("Compression score", compression_score_from_report(report))
            with c3:
                metric_or_blank("DNA length", f"{stats['dna_len']:,} nt")

            c1, c2, c3 = st.columns(3)
            c1.metric("GC content", f"{stats['gc']:.4f}")
            c2.metric("HP count", f"{stats['hp_count']:,}")
            c3.metric("HP length", f"{stats['hp_longest']} nt")

            st.download_button(
                "Download DNA string",
                data=dna.encode("utf-8"),
                file_name="dna_sequence.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_dna_clean_ui",
            )

            binary_bits, binary_filename = payload_binary_from_report(report)
            if binary_bits:
                st.download_button(
                    "Download binary string",
                    data=binary_bits.encode("utf-8"),
                    file_name=binary_filename,
                    mime="text/plain",
                    use_container_width=True,
                    key="download_binary_bits_clean_ui",
                )

            st.download_button(
                "Download report",
                data=json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="encode_report.json",
                mime="application/json",
                use_container_width=True,
                key="download_report_clean_ui",
            )


# ============================================================
# Bottom-right: Decoding results
# ============================================================

with bottom_right:
    with st.container(border=True):
        st.subheader("Decoding result")

        decode_result = st.session_state.get("decode_result")
        restored_path_value = st.session_state.get("restored_path")
        error = st.session_state.get("decode_error")

        if error:
            st.error(error)

        if not decode_result or not restored_path_value or not Path(restored_path_value).exists():
            st.info("Run decoding to show restored file result.")
        else:
            restored_path = Path(restored_path_value)
            restored_bytes = read_bytes(restored_path)

            decode_report = decode_result.get("report", {}) or {}
            rule_name = (
                get_nested(decode_report, "payload.rule_name")
                or get_nested(decode_report, "decode_rule_used.rule_name")
                or get_nested(decode_report, "selected_rule.rule_name")
                or "auto"
            )

            rule_name_map = {
                "SIMPLE": "Simple Mapping",
                "Simple Mapping": "Simple Mapping",
                "RINF_B16": "RINF_B16",
                "R2_B15": "R2_B15",
                "R1_B12": "R1_B12",
                "R0_B9": "R0_B9",
            }

            rule = rule_name_map.get(str(rule_name), str(rule_name))

            c1, c2 = st.columns(2)
            c1.metric("Detected rule", rule)
            c2.metric("Output size", fmt_bytes(len(restored_bytes)))

            encode_report = st.session_state.get("encode_report", {}) or {}
            expected_sha = (
                get_nested(encode_report, "dna_payload.sha256")
                or get_nested(encode_report, "zlib_payload.sha256")
                or get_nested(encode_report, "package.zlib_sha256")
            )
            actual_sha = sha256_bytes(restored_bytes)
            decoded_ok = None
            if expected_sha:
                decoded_ok = expected_sha == actual_sha
                st.metric("DNA Decoding", "Successful" if decoded_ok else "Failed")
                # st.caption(
                #     "This validates whether the selected payload before DNA design was recovered exactly after DNA decoding."
                # )
            else:
                st.metric("DNA Decoding", "Unavailable")
                st.caption("No reference payload hash was found for validation.")

            input_path_value = st.session_state.get("input_path")
            render_recovery_analysis(input_path_value, restored_path, decoded_ok)

            st.download_button(
                "Download decoded file",
                data=restored_bytes,
                file_name=restored_path.name,
                mime="application/octet-stream",
                use_container_width=True,
                key="download_restored_clean_ui",
            )
