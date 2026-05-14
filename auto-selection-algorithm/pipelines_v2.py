# pipelines_v2.py
# Unified v5 pipeline wrapper for the older app/API.
# Same storage architecture as app_simple:
#   compact binary metadata header + representation bytes -> best zlib -> DNA
# Decode uses finite blind search over supported DNA mappings; DNA mapping metadata is not stored separately.

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

from dna_storage_core import encode_file, decode_dna_sequence, ensure_dir, write_text


def _new_job_dir(root: str = "jobs") -> str:
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(root, job_id)
    ensure_dir(job_dir)
    return job_dir


def _write_legacy_report(job_dir: str, result: Dict[str, Any]) -> Dict[str, Any]:
    report = dict(result["report"])
    artifacts = dict(report.get("artifacts", {}))
    artifacts["report_json"] = os.path.join(job_dir, "report.json")
    report["artifacts"] = artifacts
    report["job_uuid"] = os.path.basename(job_dir)
    report["engine_version"] = report.get("engine")
    report["legacy_wrapper"] = "pipelines_v2_unified_v5"
    write_text(artifacts["report_json"], json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _run_unified(
    input_path: str,
    *,
    mode_name: str,
    operation_mode: str = "content_preserving",
    strategy: str = "hierarchical_full_automatic",
    seed: str = "ddss-seed",
    **kwargs: Any,
) -> Tuple[str, Dict[str, Any]]:
    job_dir = _new_job_dir("jobs")
    result = encode_file(
        input_path=Path(input_path),
        output_dir=Path(job_dir),
        operation_mode=operation_mode,
        strategy=strategy,
        seed=seed,
    )
    report = _write_legacy_report(job_dir, result)
    report["mode"] = mode_name
    return job_dir, report


def run_mode0_raw(input_path: str, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
    return _run_unified(input_path, mode_name="mode0_raw_unified_v5", operation_mode="exact_recovery", strategy="fixed_baseline", **kwargs)


def run_mode1_zip(input_path: str, zip_level: int = 6, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
    return _run_unified(input_path, mode_name="mode1_zip_unified_v5", operation_mode="exact_recovery", strategy="hierarchical_full_automatic", **kwargs)


def run_mode2_zip_store(input_path: str, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
    return _run_unified(input_path, mode_name="mode2_zip_store_unified_v5", operation_mode="exact_recovery", strategy="hierarchical_full_automatic", **kwargs)


def run_mode3_domain(input_path: str, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
    # For exact recovery, lossy candidates are not selected. For content preserving image demo,
    # pass operation_mode="content_preserving" from the caller.
    operation_mode = kwargs.pop("operation_mode", "content_preserving")
    return _run_unified(input_path, mode_name="mode3_domain_unified_v5", operation_mode=operation_mode, strategy="hierarchical_full_automatic", **kwargs)


def run_mode3_best(input_path: str, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
    operation_mode = kwargs.pop("operation_mode", "content_preserving")
    return _run_unified(input_path, mode_name="mode3_best_unified_v5", operation_mode=operation_mode, strategy="hierarchical_full_automatic", **kwargs)


__all__ = [
    "run_mode0_raw", "run_mode1_zip", "run_mode2_zip_store", "run_mode3_domain", "run_mode3_best", "decode_dna_sequence",
]
