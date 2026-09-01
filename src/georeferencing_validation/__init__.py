"""S4 — Georeferencing & Validation

Transforms reconstruction into geographically meaningful representation
and evaluates spatial quality.
"""
from .georeferencer import Georeferencer
from .validator import GeoreferencingValidator as Validator

__all__ = ["Georeferencer", "Validator", "GeoreferencingValidator"]
