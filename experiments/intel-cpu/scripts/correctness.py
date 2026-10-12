"""Correctness checks for benchmark outputs.

1. Ground truth (OCR off only): compare outputs/<doc>.ocr-off.compact.md and
   .json with the repo ground truth in tests/data/pdf/groundtruth. The repo
   ground truth was produced with do_ocr=False, table structure on, cell
   matching on, device CPU, which is the "ocr off" benchmark configuration.

       python correctness.py groundtruth <results dir> [--repo-check]

   --repo-check also re-converts each document with the ground-truth settings
   and runs the repo's own strict check (tests/verify_utils.py
   check_conversion_result_v2: text, tables, bboxes, pictures, doctags).

2. Reference (any mode): compare a candidate run with a reference run, for
   example experiment B against baseline A.

       python correctness.py reference <reference dir> <candidate dir>

Results are written to <dir>/correctness-<kind>.json and .md.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import Any

from common import CORPUS_FILE, REPO_ROOT, load_json, write_json


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "")


def compare_md(expected: str, actual: str) -> dict[str, Any]:
    exp, act = _norm(expected).splitlines(), _norm(actual).splitlines()
    first_diff = next(
        (i for i, (a, b) in enumerate(zip(exp, act)) if a != b),
        None if len(exp) == len(act) else min(len(exp), len(act)),
    )
    return {
        "exact": exp == act,
        "line_similarity": difflib.SequenceMatcher(
            None, exp, act, autojunk=False
        ).ratio(),
        "lines_expected": len(exp),
        "lines_actual": len(act),
        "first_diff_line": first_diff,
    }


def doc_signature(path: Path) -> dict[str, Any]:
    from docling_core.types.doc import DoclingDocument

    doc = DoclingDocument.load_from_json(path)
    tables = [
        {
            "shape": [t.data.num_rows, t.data.num_cols],
            "cells": [c.text for c in t.data.table_cells],
        }
        for t in doc.tables
    ]
    return {
        "texts": [(str(t.label.value), t.text) for t in doc.texts],
        "tables": tables,
        "pictures": len(doc.pictures),
    }


def compare_json(expected: Path, actual: Path) -> dict[str, Any]:
    exp, act = doc_signature(expected), doc_signature(actual)
    text_ratio = difflib.SequenceMatcher(
        None, exp["texts"], act["texts"], autojunk=False
    ).ratio()
    shapes_equal = [t["shape"] for t in exp["tables"]] == [
        t["shape"] for t in act["tables"]
    ]
    cells_equal = [t["cells"] for t in exp["tables"]] == [
        t["cells"] for t in act["tables"]
    ]
    return {
        "texts_expected": len(exp["texts"]),
        "texts_actual": len(act["texts"]),
        "text_items_similarity": text_ratio,
        "texts_equal": exp["texts"] == act["texts"],
        "tables_expected": len(exp["tables"]),
        "tables_actual": len(act["tables"]),
        "table_shapes_equal": shapes_equal,
        "table_cells_equal": cells_equal,
        "pictures_expected": exp["pictures"],
        "pictures_actual": act["pictures"],
    }


def repo_check(doc_path: Path, gt_base: Path) -> list[str]:
    sys.path.insert(0, str(REPO_ROOT))
    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from tests.groundtruth_paths import GroundTruthPaths
    from tests.verify_utils import check_conversion_result_v2

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    opts.table_structure_options.do_cell_matching = True
    opts.accelerator_options.device = AcceleratorDevice.CPU
    opts.generate_parsed_pages = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    result = converter.convert(doc_path)
    gt = GroundTruthPaths(
        pages_meta=gt_base.with_name(gt_base.name + ".pages.meta.json"),
        doc_json=gt_base.with_name(gt_base.name + ".json"),
        md=gt_base.with_name(gt_base.name + ".md"),
        doctags=gt_base.with_name(gt_base.name + ".doctags.txt"),
    )
    verify_doctags = gt.doctags.exists() and doc_path.name not in {
        "2203.01017v2.pdf",  # skipped in tests/test_e2e_conversion.py too
    }
    return [
        str(f)
        for f in check_conversion_result_v2(
            gt=gt, doc_result=result, generate=False, verify_doctags=verify_doctags
        )
    ]


def write_report(out_dir: Path, kind: str, rows: list[dict[str, Any]]) -> None:
    write_json(out_dir / f"correctness-{kind}.json", rows)
    lines = [
        f"# Correctness ({kind})",
        "",
        "| doc | ocr | md exact | md line sim | text items sim | tables exp/act "
        "| table shapes eq | table cells eq | repo check |",
        "|---|---|---|---:|---:|---|---|---|---|",
    ]
    for r in rows:
        md, js = r.get("md") or {}, r.get("json") or {}
        repo = r.get("repo_check")
        repo_txt = (
            "-" if repo is None else ("PASS" if not repo else f"FAIL ({len(repo)})")
        )
        lines.append(
            f"| {r['doc_id']} | {r['ocr']} | {md.get('exact', '-')} "
            f"| {md.get('line_similarity', float('nan')):.4f} "
            f"| {js.get('text_items_similarity', float('nan')):.4f} "
            f"| {js.get('tables_expected', '-')}/{js.get('tables_actual', '-')} "
            f"| {js.get('table_shapes_equal', '-')} | {js.get('table_cells_equal', '-')} "
            f"| {repo_txt} |"
        )
    (out_dir / f"correctness-{kind}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


def cmd_groundtruth(results: Path, with_repo_check: bool) -> None:
    rows = []
    for doc in load_json(CORPUS_FILE)["documents"]:
        if not doc.get("groundtruth"):
            continue
        gt_base = REPO_ROOT / doc["groundtruth"]
        stem = results / "outputs" / f"{doc['id']}.ocr-off"
        row: dict[str, Any] = {"doc_id": doc["id"], "ocr": "off"}
        md_out = stem.with_name(stem.name + ".compact.md")
        if md_out.exists():
            row["md"] = compare_md(
                gt_base.with_name(gt_base.name + ".md").read_text(encoding="utf-8"),
                md_out.read_text(encoding="utf-8"),
            )
            row["json"] = compare_json(
                gt_base.with_name(gt_base.name + ".json"),
                stem.with_name(stem.name + ".json"),
            )
        else:
            row["missing_output"] = str(md_out)
        if with_repo_check:
            row["repo_check"] = repo_check(REPO_ROOT / doc["path"], gt_base)
        rows.append(row)
    write_report(results, "groundtruth", rows)


def cmd_reference(reference: Path, candidate: Path) -> None:
    rows = []
    for ref_md in sorted((reference / "outputs").glob("*.compact.md")):
        stem = ref_md.name.removesuffix(".compact.md")
        doc_id, _, mode = stem.rpartition(".ocr-")
        cand_md = candidate / "outputs" / ref_md.name
        row: dict[str, Any] = {"doc_id": doc_id, "ocr": mode}
        if not cand_md.exists():
            row["missing_output"] = str(cand_md)
        else:
            row["md"] = compare_md(
                ref_md.read_text(encoding="utf-8"), cand_md.read_text(encoding="utf-8")
            )
            row["json"] = compare_json(
                reference / "outputs" / f"{stem}.json",
                candidate / "outputs" / f"{stem}.json",
            )
        rows.append(row)
    write_report(candidate, "reference", rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="kind", required=True)
    g = sub.add_parser("groundtruth")
    g.add_argument("results", type=Path)
    g.add_argument("--repo-check", action="store_true")
    r = sub.add_parser("reference")
    r.add_argument("reference", type=Path)
    r.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.kind == "groundtruth":
        cmd_groundtruth(args.results, args.repo_check)
    else:
        cmd_reference(args.reference, args.candidate)


if __name__ == "__main__":
    main()
