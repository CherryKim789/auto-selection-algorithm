
from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ============================================================
# UI CUSTOMIZATION DEFAULTS
# Edit these values if you want fixed defaults without using sidebar controls.
# ============================================================
UI_DEFAULTS: dict[str, Any] = {
    "input_image_width": 320,
    "output_image_width": 320,
    "preview_layout": "stacked",  # "stacked" or "side_by_side"
    "table_height": 360,
    "dna_preview_chars": 6000,
    "dna_line_chunk": 120,
    "show_image_full_width": False,
    "text_preview_height": 260,
    "text_preview_chars": 8000,
}

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".json", ".csv", ".py", ".xml", ".yaml", ".yml",
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff",
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac",
    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v",
    ".pdf", ".docx", ".pptx", ".xlsx",
    ".zip", ".gz", ".xz", ".bz2",
}

ALLOWED_MIME_PREFIXES = (
    "text/",
    "image/",
    "audio/",
    "video/",
    "application/pdf",
    "application/zip",
    "application/gzip",
    "application/x-xz",
    "application/x-bzip2",
    "application/vnd.openxmlformats-officedocument",
    "application/json",
    "application/xml",
)

SIZE_LIMIT_MB_BY_KIND = {
    "text": 20,
    "image": 20,
    "audio": 50,
    "video": 100,
    "binary": 25,
}

VALIDATION_HELP = (
    "The app validates file extension, MIME guess, and file size before encoding. "
    "The DNA payload uses minimal DCP1 compressor metadata inside zlib and blind DNA-rule decoding."
)

# Streamlit widget keys must be unique within one script run.
# display_file_preview() can be called multiple times for the same text/PDF file
# in different tabs or sections, so path/title alone is not always enough.
_PREVIEW_WIDGET_COUNTER = 0


def _next_preview_widget_counter() -> int:
    global _PREVIEW_WIDGET_COUNTER
    _PREVIEW_WIDGET_COUNTER += 1
    return _PREVIEW_WIDGET_COUNTER


def _stable_widget_key(prefix: str, *parts: Any) -> str:
    """Create a stable unique Streamlit widget key from context."""
    raw = "|".join(str(x) for x in parts)
    digest = hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _get_nested(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def save_uploaded_file(uploaded_file: Any, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    path = destination_dir / uploaded_file.name
    path.write_bytes(uploaded_file.getbuffer())
    return path


def infer_media_kind(path: str | Path) -> str:
    path = Path(path)
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("audio/"):
            return "audio"
        if mime.startswith("video/"):
            return "video"
        if mime.startswith("text/"):
            return "text"
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}:
        return "image"
    if suffix in {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}:
        return "audio"
    if suffix in {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}:
        return "video"
    if suffix in {".txt", ".md", ".json", ".csv", ".py", ".yaml", ".yml", ".xml", ".html", ".htm"}:
        return "text"
    if suffix == ".pdf":
        return "pdf"
    return "binary"


def _mime_allowed(mime: str | None) -> bool:
    if not mime:
        return False
    return any(mime.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES)


def validate_uploaded_file(uploaded_file: Any) -> dict[str, Any]:
    filename = uploaded_file.name
    suffix = Path(filename).suffix.lower()
    mime, _ = mimetypes.guess_type(filename)
    size_bytes = int(uploaded_file.size)
    size_mb = round(size_bytes / (1024 * 1024), 4)
    media_kind = infer_media_kind(filename)
    size_limit_mb = SIZE_LIMIT_MB_BY_KIND.get(media_kind, SIZE_LIMIT_MB_BY_KIND["binary"])
    extension_allowed = suffix in SUPPORTED_EXTENSIONS
    mime_allowed = _mime_allowed(mime) or suffix in {".docx", ".pptx", ".xlsx", ".gz", ".xz", ".bz2"}
    size_allowed = size_mb <= size_limit_mb
    errors: list[str] = []
    warnings: list[str] = []
    if not extension_allowed:
        errors.append(f"Unsupported extension: {suffix or '[none]'}.")
    if not mime_allowed:
        warnings.append(f"MIME type is unknown or unusual: {mime}.")
    if not size_allowed:
        errors.append(f"File too large for demo mode: {size_mb} MB > {size_limit_mb} MB.")
    return {
        "filename": filename,
        "extension": suffix,
        "mime": mime,
        "media_kind": media_kind,
        "size_bytes": size_bytes,
        "size_mb": size_mb,
        "size_limit_mb": size_limit_mb,
        "extension_allowed": extension_allowed,
        "mime_allowed": mime_allowed,
        "size_allowed": size_allowed,
        "accepted": extension_allowed and size_allowed,
        "errors": errors,
        "warnings": warnings,
    }


def load_text_preview(path: str | Path, max_chars: int | None = None) -> str:
    path = Path(path)
    if max_chars is None:
        max_chars = int(UI_DEFAULTS["text_preview_chars"])
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text[:max_chars]
    except Exception:
        return ""


def display_file_preview(
    path: str | Path,
    title: str = "Preview",
    *,
    image_width: int | None = None,
    image_use_container_width: bool | None = None,
    text_height: int | None = None,
    text_max_chars: int | None = None,
    show_file_meta: bool = True,
) -> None:
    """Display a file preview with controllable image sizing.

    Change defaults in UI_DEFAULTS, or pass image_width from app.py.
    image_use_container_width=True overrides image_width and fills the container.
    """
    path = Path(path)
    preview_instance = _next_preview_widget_counter()
    st.markdown(f"#### {title}")

    if not path.exists():
        st.info("Preview file does not exist.")
        return

    kind = infer_media_kind(path)
    if show_file_meta:
        st.caption(f"{path.name} · {path.suffix.lower() or 'no extension'} · {path.stat().st_size:,} bytes")

    if kind == "image":
        if image_use_container_width is None:
            image_use_container_width = bool(UI_DEFAULTS["show_image_full_width"])
        if image_width is None:
            image_width = int(UI_DEFAULTS["input_image_width"])
        if image_use_container_width:
            st.image(str(path), use_container_width=True)
        else:
            st.image(str(path), width=int(image_width))
        return

    if kind == "audio":
        st.audio(path.read_bytes())
        return

    if kind == "video":
        st.video(path.read_bytes())
        return

    if kind == "text":
        if text_height is None:
            text_height = int(UI_DEFAULTS["text_preview_height"])
        preview = load_text_preview(path, max_chars=text_max_chars)
        if preview:
            st.text_area(
                "Content preview",
                value=preview,
                height=int(text_height),
                disabled=True,
                key=_stable_widget_key("text_preview", preview_instance, str(path.resolve()), title, int(text_height), text_max_chars),
            )
        else:
            st.info("Text preview unavailable.")
        return

    if kind == "pdf":
        st.info("Inline PDF preview is disabled. Download the file to inspect it.")
        st.download_button(
            "Download PDF",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/pdf",
            use_container_width=True,
            key=_stable_widget_key("pdf_download", preview_instance, str(path.resolve()), title),
        )
        return

    st.info(f"No inline preview for this file type: {path.suffix.lower() or 'unknown'}")


def metrics_to_df(data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for k, v in (data or {}).items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        rows.append({"metric": k, "value": v})
    return pd.DataFrame(rows)


def render_metrics_table(data: dict[str, Any], *, height: int | None = None, hide_index: bool = True) -> None:
    if height is None:
        height = int(UI_DEFAULTS["table_height"])
    df = metrics_to_df(data)
    st.dataframe(df, use_container_width=True, hide_index=hide_index, height=min(height, 420))


def data_profile_df(report: dict[str, Any]) -> pd.DataFrame:
    return metrics_to_df(report.get("input_profile", {}))


def compression_candidates_df(report: dict[str, Any]) -> pd.DataFrame:
    rows = report.get("compression", {}).get("candidates", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["score", "size_ratio", "runtime_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    sort_cols = [c for c in ["score", "size_ratio", "zlib_size_bytes", "size_bytes"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=True)
    preferred = [
        "method_name", "policy", "domain", "lossy", "lossless", "size_bytes",
        "zlib_size_bytes", "size_ratio", "score", "runtime_sec", "selected_backend_bytes",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols].reset_index(drop=True)


def dna_candidates_df(report: dict[str, Any]) -> pd.DataFrame:
    rows = report.get("dna_design", {}).get("candidates", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "init_dimer" not in df.columns:
        df["init_dimer"] = ""
    if "whiten" not in df.columns:
        df["whiten"] = False
    df["rule_label"] = df.get("rule_name", "").astype(str) + " | " + df["init_dimer"].astype(str) + " | w=" + df["whiten"].astype(str)
    for col in ["score", "dna_len_nt", "bits_per_nt", "gc_fraction", "runtime_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    sort_cols = [c for c in ["selected", "score", "dna_len_nt"] if c in df.columns]
    if "selected" in df.columns:
        df["_selected_sort"] = ~df["selected"].astype(bool)
        df = df.sort_values(by=["_selected_sort"] + [c for c in ["score", "dna_len_nt"] if c in df.columns])
        df = df.drop(columns=["_selected_sort"])
    elif sort_cols:
        df = df.sort_values(by=sort_cols)
    preferred = [
        "selected", "rule_label", "rule_name", "mode", "init_dimer", "whiten", "dna_len_nt",
        "bits_per_nt", "gc_fraction", "score", "runtime_sec", "motif_count", "homopolymer",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols].reset_index(drop=True)


def decode_summary_df(report: dict[str, Any]) -> pd.DataFrame:
    restored = report.get("restored", {})
    verification = report.get("verification", {})
    payload = report.get("payload", {})
    package = report.get("package", {})
    merged = {
        "restored_path": restored.get("path"),
        "restored_size_bytes": restored.get("size_bytes"),
        "restored_sha256": restored.get("sha256"),
        **verification,
        **payload,
        "package_method": package.get("method_name"),
        "package_restore_ext": package.get("restore_ext"),
    }
    return metrics_to_df(merged)


def extract_key_metrics(report: dict[str, Any]) -> dict[str, Any]:
    selected_comp = report.get("compression", {}).get("selected", {})
    selected_dna = report.get("dna_design", {}).get("selected", {})
    verification = report.get("verification", {})
    package = report.get("package", {})
    bits_per_nt = selected_dna.get("bits_per_nt", 0.0)
    try:
        bits_per_nt = round(float(bits_per_nt), 4)
    except Exception:
        bits_per_nt = None
    return {
        "Input bytes": _get_nested(report, "input.size_bytes"),
        "Method": selected_comp.get("method_name"),
        "Rep bytes": package.get("representation_len_bytes"),
        "DCP1 header": package.get("package_header_len_bytes"),
        "Zlib bytes": package.get("zlib_size_bytes"),
        "DNA rule": selected_dna.get("rule_name"),
        "Final DNA nt": report.get("dna_design", {}).get("final_dna_len_nt"),
        "Bits/nt": bits_per_nt,
        "CRC ok": verification.get("package_crc32_ok", verification.get("payload_crc32_ok")),
    }


def render_metric_cards(metrics: dict[str, Any], columns: int = 5) -> None:
    items = list(metrics.items())
    cols = st.columns(columns)
    for idx, (k, v) in enumerate(items):
        cols[idx % columns].metric(k, "—" if v is None else v)


def render_dataframe(
    df: pd.DataFrame,
    *,
    height: int | None = None,
    hide_index: bool = True,
    column_config: dict[str, Any] | None = None,
) -> None:
    if height is None:
        height = int(UI_DEFAULTS["table_height"])
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=hide_index,
        height=int(height),
        column_config=column_config,
    )


def render_matplotlib_bar(df: pd.DataFrame, x: str, y: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_df = df.copy()
    if plot_df.empty or x not in plot_df or y not in plot_df:
        return fig
    ax.bar(plot_df[x].astype(str), pd.to_numeric(plot_df[y], errors="coerce"))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    return fig


def render_matplotlib_scatter(df: pd.DataFrame, x: str, y: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_df = df.copy()
    if plot_df.empty or x not in plot_df or y not in plot_df:
        return fig
    ax.scatter(pd.to_numeric(plot_df[x], errors="coerce"), pd.to_numeric(plot_df[y], errors="coerce"))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    return fig


def format_dna_preview(dna: str, chunk: int | None = None, max_chars: int | None = None) -> str:
    if chunk is None:
        chunk = int(UI_DEFAULTS["dna_line_chunk"])
    if max_chars is None:
        max_chars = int(UI_DEFAULTS["dna_preview_chars"])
    trimmed = dna[:max_chars]
    parts = [trimmed[i:i + chunk] for i in range(0, len(trimmed), chunk)]
    out = "\n".join(parts)
    if len(dna) > max_chars:
        out += f"\n... [truncated, total length={len(dna)} nt]"
    return out


def flatten_dict(data: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for key, value in (data or {}).items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, list):
            items[new_key] = json.dumps(value, ensure_ascii=False)[:2000]
        else:
            items[new_key] = value
    return items


def build_artifact_zip(output_dir: str | Path) -> bytes:
    output_dir = Path(output_dir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in output_dir.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=file.relative_to(output_dir))
    buf.seek(0)
    return buf.getvalue()
