"""S4 Georeferencer

Transforms local 3D reconstruction into geographic coordinates.
"""


class Georeferencer:
    """Georeference the reconstructed 3D scene."""

    def __init__(self, reconstruction_data, coordinate_reference: dict):
        self.reconstruction_data = reconstruction_data
        self.coordinate_reference = coordinate_reference

    def georeference(self):
        """Transform reconstruction to geographic coordinates.

        Returns georeferenced scene with coordinate reference info.
        """
        # TODO: Implement georeferencing
        return {
            "geo_point_cloud": [],
            "geographic_reference": {},
            "metrics": {},
        }