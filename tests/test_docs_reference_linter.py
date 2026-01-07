from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MissingReference:
    source_file: Path
    line_no: int
    raw_ref: str
    resolved_path: Path
    suggested_path: str | None = None


_ALLOWED_PREFIXES: tuple[str, ...] = (
    "docs/",
    "src/",
    "tests/",
    ".github/workflows/",
)

_LINK_TARGET_RE = re.compile(r"\]\(([^)]+)\)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")

_PATH_TOKEN_RE = re.compile(
    r"(?P<path>(?:docs/|src/|tests/|\.github/workflows/)[A-Za-z0-9_./\-]+)"
)


def _iter_markdown_files() -> list[Path]:
    top_level = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / "architecture.md",
        REPO_ROOT / "CHANGELOG.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "SUMMARY.md",
        REPO_ROOT / "prompts.md",
    ]

    roots = [
        REPO_ROOT / "docs",
        REPO_ROOT / "src",
        REPO_ROOT / "reports",
        REPO_ROOT / "data",
        REPO_ROOT / "stubs",
    ]

    files: list[Path] = [p for p in top_level if p.exists()]
    for root in roots:
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("*.md")))

    # Deterministic order for stable test output.
    return sorted({p.resolve() for p in files})


def _strip_markdown_target(raw: str) -> str:
    target = raw.strip()

    # Remove surrounding <...> used in some markdown autolinks.
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()

    # Drop common wrappers.
    target = target.strip("\"'")

    # Ignore anchors-only.
    if target.startswith("#"):
        return ""

    # Ignore external URLs / mailto.
    if re.match(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*):", target):
        return ""

    # Normalize leading ./
    if target.startswith("./"):
        target = target[2:]

    # Remove query / anchor parts.
    target = target.split("?", 1)[0]
    target = target.split("#", 1)[0]

    # pytest node ids like tests/test_x.py::TestClass::test_name
    target = target.split("::", 1)[0]

    # Trim trailing punctuation that often follows inline refs.
    target = target.rstrip(").,;:>")

    return target


def _looks_like_repo_path(target: str) -> bool:
    return any(target.startswith(prefix) for prefix in _ALLOWED_PREFIXES)


def _extract_paths_from_snippet(snippet: str) -> list[str]:
    paths: list[str] = []

    for m in _PATH_TOKEN_RE.finditer(snippet):
        token = m.group("path")

        # Ignore common templates like tests/test_<name>.py where the token would
        # be matched as tests/test_ (i.e. immediately followed by a placeholder).
        end = m.end("path")
        next_char = snippet[end : end + 1]
        if next_char in {"<", "{", "*"}:
            continue
        if token.endswith("_"):
            continue

        paths.append(token)

    return paths


def _extract_candidate_refs(line: str) -> list[str]:
    refs: list[str] = []

    # 1) Markdown link targets: [text](target)
    for m in _LINK_TARGET_RE.finditer(line):
        refs.append(m.group(1))

    # 2) Inline code spans: `...`
    for m in _INLINE_CODE_RE.finditer(line):
        refs.extend(_extract_paths_from_snippet(m.group(1)))

    # 3) Autolinks: <docs/...> etc.
    for m in re.finditer(r"<(docs/|src/|tests/|\.github/workflows/)[^>]+>", line):
        refs.extend(_extract_paths_from_snippet(m.group(0)[1:-1]))

    return refs


def _check_markdown_file(path: Path) -> list[MissingReference]:
    missing: list[MissingReference] = []

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # If a doc isn't UTF-8 we skip it; this repo's docs should be UTF-8.
        return missing

    for idx, line in enumerate(text.splitlines(), start=1):
        line_lower = line.lower()
        if "docs-lint:ignore" in line_lower:
            continue
        for raw_ref in _extract_candidate_refs(line):
            target = _strip_markdown_target(raw_ref)
            if not target:
                continue
            if not _looks_like_repo_path(target):
                continue

            # Allow directory references with or without a trailing slash.
            normalized_target = target.rstrip("/")

            candidate = (REPO_ROOT / normalized_target)
            resolved = candidate.resolve()
            if not resolved.exists():
                if "docs-lint:planned" in line_lower:
                    continue
                missing.append(
                    MissingReference(
                        source_file=path,
                        line_no=idx,
                        raw_ref=raw_ref,
                        resolved_path=candidate,
                    )
                )
                continue

            # Catch case/path mismatches that only fail on case-sensitive FS
            # (Linux CI) but can pass locally on macOS/Windows.
            try:
                real_rel = resolved.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                # Not a repo-local path; nothing actionable.
                continue

            if real_rel != normalized_target:
                if "docs-lint:planned" in line_lower:
                    continue
                missing.append(
                    MissingReference(
                        source_file=path,
                        line_no=idx,
                        raw_ref=raw_ref,
                        resolved_path=candidate,
                        suggested_path=real_rel,
                    )
                )

    return missing


def _extract_runbook_rollback_procedure(runbook_path: Path) -> str | None:
    try:
        text = runbook_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    # Find closing front matter delimiter.
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return None

    for line in lines[1:end_idx]:
        if not line.lstrip().startswith("rollback_procedure:"):
            continue

        _, value = line.split(":", 1)
        value = value.strip()
        if not value:
            return None

        return value

    return None


def test_docs_references_are_resolvable() -> None:
    missing: list[MissingReference] = []

    for md_file in _iter_markdown_files():
        missing.extend(_check_markdown_file(md_file))

    if missing:
        details = "\n".join(
            (
                f"- {m.source_file.relative_to(REPO_ROOT)}:{m.line_no} -> {m.raw_ref!r} "
                f"(missing: {m.resolved_path.relative_to(REPO_ROOT)})"
                + (" (did you mean: " + m.suggested_path + ")" if m.suggested_path else "")
            )
            for m in missing[:50]
        )
        more = "" if len(missing) <= 50 else f"\n… plus {len(missing) - 50} more"
        raise AssertionError(
            "Broken docs references found (docs/src/tests/.github/workflows):\n"
            + details
            + more
        )


def test_runbooks_have_valid_rollback_procedure_links() -> None:
    runbooks_dir = REPO_ROOT / "docs" / "runbooks"
    if not runbooks_dir.exists():
        return

    missing: list[str] = []
    for runbook in sorted(runbooks_dir.glob("*.md")):
        rollback = _extract_runbook_rollback_procedure(runbook)
        if rollback is None:
            continue

        rollback_target = _strip_markdown_target(rollback)
        if not rollback_target:
            continue
        if not _looks_like_repo_path(rollback_target):
            # We only support repo-root-relative rollback paths.
            missing.append(
                f"{runbook.relative_to(REPO_ROOT)}: rollback_procedure is not a repo path: {rollback!r}"
            )
            continue

        resolved = (REPO_ROOT / rollback_target).resolve()
        if not resolved.exists():
            missing.append(
                f"{runbook.relative_to(REPO_ROOT)}: rollback_procedure missing: {rollback_target}"
            )

    if missing:
        raise AssertionError(
            "Invalid runbook rollback_procedure references:\n" + "\n".join(missing)
        )
