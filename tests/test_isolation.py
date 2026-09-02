"""Subsystem isolation guard.

These tests fail the build if any subsystem imports from a sibling
subsystem's package. The orchestrator (`src/pipeline/`) is the only
module allowed to import across subsystem boundaries, and even the
``interface.py`` of each subsystem must not import from another
subsystem.

The forbidden imports are checked statically by scanning every
``*.py`` file under the offending subsystem for the sibling package
name (e.g. ``src.localization_sensor_fusion`` inside
``src/visual_perception/``).

When the test suite is run, violations produce a clear error message
naming the offending file and the sibling subsystem it tried to
reach.
"""

from __future__ import annotations

import pathlib
import re
from typing import Iterable, Tuple

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SUBSYSTEMS = (
    "src/visual_perception",
    "src/localization_sensor_fusion",
    "src/reconstruction",
    "src/georeferencing_validation",
    "src/application_deployment",
)

# All ordered pairs (A, B) where A must NOT import from B.
FORBIDDEN: Tuple[Tuple[str, str], ...] = tuple(
    (a, b)
    for i, a in enumerate(SUBSYSTEMS)
    for b in SUBSYSTEMS[i + 1:]
) + tuple(
    (b, a)
    for i, a in enumerate(SUBSYSTEMS)
    for b in SUBSYSTEMS[i + 1:]
)


def _module_name(subsystem_path: str) -> str:
    """Convert ``src/visual_perception`` to ``src.visual_perception``."""
    return subsystem_path.replace("/", ".")


def _iter_python_files(subsystem_path: str) -> Iterable[pathlib.Path]:
    root = REPO_ROOT / subsystem_path
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _find_sibling_imports(
    subsystem_path: str, sibling_module: str
) -> list[Tuple[pathlib.Path, int, str]]:
    """Return a list of (file, line_no, line_text) that import ``sibling_module``."""
    hits: list[Tuple[pathlib.Path, int, str]] = []
    pattern = re.compile(rf"(^|\s)(from\s+{re.escape(sibling_module)}(\.|\s+import)|import\s+{re.escape(sibling_module)}(\.|\s|$))")
    for f in _iter_python_files(subsystem_path):
        text = f.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append((f, line_no, line.strip()))
    return hits


@pytest.mark.parametrize("sub,other", FORBIDDEN)
def test_subsystem_does_not_import_sibling(sub: str, other: str) -> None:
    """No module under ``sub`` may import from ``other``."""
    sibling_module = _module_name(other)
    hits = _find_sibling_imports(sub, sibling_module)
    if hits:
        formatted = "\n".join(
            f"  {f.relative_to(REPO_ROOT)}:{ln}: {txt}"
            for f, ln, txt in hits
        )
        pytest.fail(
            f"Subsystem '{sub}' must not import from sibling '{other}':\n{formatted}"
        )


@pytest.mark.parametrize("sub", SUBSYSTEMS)
def test_subsystem_internal_namespace_is_private(sub: str) -> None:
    """Subsystem internals live under ``_internal/`` and stay there.

    This is a structural sanity check: ``_internal/`` directories are
    the agreed-upon namespace for hidden modules. If a subsystem
    currently has no ``_internal/`` directory, the refactor is
    incomplete and this test fails loudly.
    """
    internal_dir = REPO_ROOT / sub / "_internal"
    assert internal_dir.is_dir(), (
        f"Subsystem '{sub}' must have an _internal/ directory to keep "
        f"its implementation private. Create one and move non-public "
        f"modules under it."
    )


def test_pipeline_is_only_cross_subsystem_importer() -> None:
    """The orchestrator is the only module allowed to import across subsystems.

    Any other top-level module outside ``src/pipeline/`` that touches a
    sibling subsystem package name is reported.
    """
    pipeline_dir = REPO_ROOT / "src" / "pipeline"
    for sub in SUBSYSTEMS:
        sub_module = _module_name(sub)
        # Search everything under src/ except pipeline/ for the sibling.
        for path in (REPO_ROOT / "src").rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            # Skip the subsystem's own files (they're checked by
            # test_subsystem_does_not_import_sibling above).
            try:
                path.relative_to(REPO_ROOT / sub)
                continue
            except ValueError:
                pass
            try:
                path.relative_to(pipeline_dir)
                continue  # orchestrator is allowed
            except ValueError:
                pass

            text = path.read_text(encoding="utf-8")
            for other in SUBSYSTEMS:
                other_module = _module_name(other)
                if other_module == sub_module:
                    continue
                pattern = re.compile(
                    rf"(^|\s)(from\s+{re.escape(other_module)}(\.|\s+import)|import\s+{re.escape(other_module)}(\.|\s|$))"
                )
                if pattern.search(text):
                    pytest.fail(
                        f"Only src/pipeline/ may import across subsystems. "
                        f"Offending file: {path.relative_to(REPO_ROOT)} "
                        f"imports '{other_module}'."
                    )


def test_interface_does_not_import_sibling() -> None:
    """Each subsystem's ``interface.py`` must not import from a sibling.

    The single integration symbol is built only from the subsystem's
    own ``_internal/`` namespace.
    """
    for sub in SUBSYSTEMS:
        interface_path = REPO_ROOT / sub / "interface.py"
        if not interface_path.is_file():
            pytest.fail(f"Subsystem '{sub}' is missing interface.py")
        text = interface_path.read_text(encoding="utf-8")
        for other in SUBSYSTEMS:
            if other == sub:
                continue
            other_module = _module_name(other)
            if re.search(
                rf"(^|\s)(from\s+{re.escape(other_module)}(\.|\s+import)|import\s+{re.escape(other_module)}(\.|\s|$))",
                text,
            ):
                pytest.fail(
                    f"{interface_path.relative_to(REPO_ROOT)} must not "
                    f"import from sibling '{other_module}'."
                )
