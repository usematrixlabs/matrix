"""S5 — Application & Deployment

Bundles and summarizes the final pipeline outputs into a portable
``s5_output/`` directory. This is a minimal, orchestrator-friendly
implementation; richer application logic (UI / API) can be layered on top
later without changing the pipeline contract.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class FinalOutput:
    """Portable summary of the end-to-end Matrix pipeline result."""

    success: bool
    scene_id: str
    output_dir: str
    stage_status: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Finalizer:
    """Bundles per-subsystem outputs into a single portable ``s5_output/`` directory.

    The orchestrator is expected to call :meth:`bundle` after S1–S4 have
    completed. This intentionally performs no algorithmic work — it only
    copies metadata pointers together and writes a top-level summary.
    """

    SCHEMA_VERSION = "1.0.0"

    def __init__(self, output_dir: Union[str, Path]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def bundle(
        self,
        scene_id: str,
        success: bool,
        stage_status: Dict[str, str],
        artifacts: Optional[Dict[str, str]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> FinalOutput:
        """Persist the final pipeline summary and return a :class:`FinalOutput`."""
        final = FinalOutput(
            success=success,
            scene_id=scene_id,
            output_dir=str(self.output_dir.resolve()),
            stage_status=dict(stage_status),
            artifacts=dict(artifacts or {}),
            summary=dict(summary or {}),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        manifest_path = self.output_dir / "final_output.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    **final.to_dict(),
                },
                f,
                indent=2,
            )

        return final


__all__ = ["Finalizer", "FinalOutput"]
