#!/usr/bin/env python3
"""Audit the full, anonymous, and supplementary KBS manuscript builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz


SCIENTIFIC_SOURCES = (
    "main.tex",
    "supplementary.tex",
    "01-introduction.tex",
    "02-related-work.tex",
    "03-method.tex",
    "04-experiments.tex",
    "05-conclusion.tex",
    "appendix-a.tex",
    "appendix-scorer-table.tex",
    "references.bib",
    "highlights.txt",
)

KEY_PHRASES = (
    "Schema-Grounded Structured Chain-of-Thought",
    "93.8%",
    "subtype-level compositional novelty",
    "schema-driven instantiation transfers to ACE 2005",
    "0.197",
    "0.155",
    "0.351",
    # Every AI tool that assisted must stay named in the disclosure; dropping one
    # would understate the assistance actually used.
    "OpenAI Codex and Anthropic Claude",
)

BAD_LOG_PATTERNS = (
    re.compile(r"Citation .* undefined"),
    re.compile(r"Reference .* undefined"),
    re.compile(r"There were undefined (?:citations|references)"),
    re.compile(r"Overfull \\hbox"),
    re.compile(r"Underfull \\hbox"),
)

SUBMISSION_PLACEHOLDERS = {
    "main.tex": (
        "Surname et al.",
        "Funding or project information can be placed here.",
        "First Author",
        "Second Author",
        "0000-0000-0000-0000",
        "first.author@example.com",
        "second.author@example.com",
        "School or Institute Name",
        "Department or Lab Name",
        "% TODO: add repository URL before submission.",
        "% TODO before submission: add the funding statement",
    ),
    "cover-letter.md": (
        "[Date]",
        "[repository URL]",
        "[Corresponding author, affiliation, email]",
    ),
    "cover-letter.txt": (
        "[Date]",
        "[repository URL]",
        "[Corresponding author, affiliation, email]",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle_sha256(paper_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in SCIENTIFIC_SOURCES:
        path = paper_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def pdf_text(document: fitz.Document) -> str:
    return "\n".join(page.get_text() for page in document)


def identity_markers(main_source: str) -> list[str]:
    patterns = (
        r"\\author(?:\[[^\]]*\])?\{([^{}]+)\}",
        r"\\ead\{([^{}]+)\}",
        r"organization=\{([^{}]+)\}",
        # Postal-address fields are as identifying as the organization name once a
        # real affiliation is filled in, so they must be scanned too.
        r"addressline=\{([^{}]+)\}",
        r"postcode=\{([^{}]+)\}",
        r"state=\{([^{}]+)\}",
        r"orcid=([0-9-]+)",
        r"\\tnotetext\[[^\]]+\]\{([^{}]+)\}",
    )
    markers: set[str] = set()
    for pattern in patterns:
        markers.update(match.strip() for match in re.findall(pattern, main_source))
    # "China" alone appears in ordinary prose (dataset provenance, related work),
    # so country= is deliberately not scanned; the city/postcode/street are what
    # actually identify the group.
    return sorted(
        marker
        for marker in markers
        if marker and marker.lower() not in {"anonymous", "anonymous author", "anonymous authors"}
    )


def bad_log_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [line for line in lines if any(pattern.search(line) for pattern in BAD_LOG_PATTERNS)]


def metadata_placeholders(paper_dir: Path) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for name, placeholders in SUBMISSION_PLACEHOLDERS.items():
        text = (paper_dir / name).read_text(encoding="utf-8")
        present = [placeholder for placeholder in placeholders if placeholder in text]
        if present:
            hits[name] = present
    return hits


def audit(args: argparse.Namespace) -> dict:
    paper_dir = args.paper_dir.resolve()
    full_pdf = (paper_dir / args.full_pdf).resolve()
    anonymous_pdf = (paper_dir / args.anonymous_pdf).resolve()
    supplementary_pdf = (paper_dir / args.supplementary_pdf).resolve()
    full_log = (paper_dir / args.full_log).resolve()
    anonymous_log = (paper_dir / args.anonymous_log).resolve()
    supplementary_log = (paper_dir / args.supplementary_log).resolve()

    errors: list[str] = []

    wrapper = (paper_dir / "main-anonymous.tex").read_text(encoding="utf-8")
    wrapper_lines = [line.strip() for line in wrapper.splitlines() if line.strip() and not line.lstrip().startswith("%")]
    expected_wrapper = [r"\def\anonymoussubmission{1}", r"\input{main.tex}"]
    wrapper_is_shared_source = wrapper_lines == expected_wrapper
    if not wrapper_is_shared_source:
        errors.append("main-anonymous.tex is not the two-line shared-source wrapper")

    main_source = (paper_dir / "main.tex").read_text(encoding="utf-8")
    if r"\documentclass[a4paper,fleqn,doubleblind]{cas-sc}" not in main_source:
        errors.append("anonymous branch does not use cas-sc doubleblind mode")

    full_doc = fitz.open(full_pdf)
    anonymous_doc = fitz.open(anonymous_pdf)
    supplementary_doc = fitz.open(supplementary_pdf)
    full_text = pdf_text(full_doc)
    anonymous_text = pdf_text(anonymous_doc)
    supplementary_text = pdf_text(supplementary_doc)

    # The anonymous variant suppresses the CRediT statement (cas-sc hides it under
    # \theblind), so once real CRediT roles are present the author-bearing PDF may
    # legitimately run one page longer. Anything beyond that is a real divergence.
    page_delta = full_doc.page_count - anonymous_doc.page_count
    credit_in_full = "CRediT authorship contribution statement" in full_text
    allowed_delta = 1 if credit_in_full else 0
    if not 0 <= page_delta <= allowed_delta:
        errors.append(
            "full and anonymous PDFs have different page counts "
            f"(full {full_doc.page_count}, anonymous {anonymous_doc.page_count}; "
            f"allowed extra pages {allowed_delta})"
        )
    if "Supplementary Material" not in supplementary_text:
        errors.append("supplementary PDF is missing its title")

    missing_key_phrases: dict[str, list[str]] = {}
    for label, text in (("full", full_text), ("anonymous", anonymous_text)):
        missing = [phrase for phrase in KEY_PHRASES if phrase not in text]
        if missing:
            missing_key_phrases[label] = missing
            errors.append(f"{label} PDF is missing {len(missing)} frozen key phrase(s)")

    markers = identity_markers(main_source) + list(args.forbid)
    anonymous_lower = anonymous_text.lower()
    leaks = sorted({marker for marker in markers if marker.lower() in anonymous_lower})
    if leaks:
        errors.append(f"anonymous PDF leaks {len(leaks)} identity marker(s)")

    anonymous_metadata_author = (anonymous_doc.metadata.get("author") or "").strip()
    if anonymous_metadata_author:
        errors.append("anonymous PDF metadata has a non-empty author field")

    log_issues = {
        "full": bad_log_lines(full_log),
        "anonymous": bad_log_lines(anonymous_log),
        "supplementary": bad_log_lines(supplementary_log),
    }
    for label, issues in log_issues.items():
        if issues:
            errors.append(f"{label} log has {len(issues)} blocking warning(s)")

    placeholder_hits = metadata_placeholders(paper_dir)
    metadata_complete = not placeholder_hits
    if args.require_complete_metadata and not metadata_complete:
        count = sum(len(items) for items in placeholder_hits.values())
        errors.append(f"submission metadata still has {count} placeholder occurrence type(s)")

    return {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "shared_scientific_source": wrapper_is_shared_source,
        "scientific_source_bundle_sha256": source_bundle_sha256(paper_dir),
        "scientific_source_files": list(SCIENTIFIC_SOURCES),
        "full_pdf": {
            "path": str(full_pdf),
            "pages": full_doc.page_count,
            "bytes": full_pdf.stat().st_size,
            "sha256": sha256_file(full_pdf),
        },
        "anonymous_pdf": {
            "path": str(anonymous_pdf),
            "pages": anonymous_doc.page_count,
            "bytes": anonymous_pdf.stat().st_size,
            "sha256": sha256_file(anonymous_pdf),
            "metadata_author": anonymous_metadata_author,
            "identity_markers_checked": markers,
            "identity_leaks": leaks,
        },
        "supplementary_pdf": {
            "path": str(supplementary_pdf),
            "pages": supplementary_doc.page_count,
            "bytes": supplementary_pdf.stat().st_size,
            "sha256": sha256_file(supplementary_pdf),
        },
        "missing_key_phrases": missing_key_phrases,
        "blocking_log_lines": log_issues,
        "metadata_complete": metadata_complete,
        "metadata_placeholders": placeholder_hits,
        "strict_metadata_required": args.require_complete_metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=Path("thesis/paper-kbs"))
    parser.add_argument("--full-pdf", default="main.pdf")
    parser.add_argument("--anonymous-pdf", default="main-anonymous.pdf")
    parser.add_argument("--supplementary-pdf", default="supplementary.pdf")
    parser.add_argument("--full-log", default="main.log")
    parser.add_argument("--anonymous-log", default="main-anonymous.log")
    parser.add_argument("--supplementary-log", default="supplementary.log")
    parser.add_argument("--forbid", action="append", default=[], help="additional identity string forbidden in the anonymous PDF")
    parser.add_argument("--require-complete-metadata", action="store_true", help="fail while manuscript or cover-letter placeholders remain")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
