"""S2 Exporter for serializing localization outputs to S2PayloadOutput contracts."""

import json
from pathlib import Path
from typing import List, Union
from ..schemas.contracts import (
    S2ObservationOutput,
    S2PayloadOutput,
    Units,
)


class S2Exporter:
    def __init__(self, coordinate_frame: str = "NED", units: str = "meters"):
        self.coordinate_frame = coordinate_frame
        self.units = units

    def create_payload(
        self,
        observations: List[S2ObservationOutput],
        schema_version: str = "1.0.0",
    ) -> S2PayloadOutput:
        """Wraps localized observations into a standardized S2PayloadOutput contract."""
        unit_payload = self.units
        if isinstance(unit_payload, str):
            unit_payload = {"position": unit_payload, "rotation": "quaternion"}
        elif isinstance(unit_payload, Units):
            unit_payload = unit_payload.model_dump()

        return S2PayloadOutput(
            schema_version=schema_version,
            coordinate_frame=self.coordinate_frame,
            units=unit_payload,
            observations=observations,
        )

    def export_to_json(
        self,
        payload: S2PayloadOutput,
        output_path: Union[str, Path],
    ) -> Path:
        """Serializes S2PayloadOutput to a JSON file on disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(payload.model_dump_json(indent=2))

        return path