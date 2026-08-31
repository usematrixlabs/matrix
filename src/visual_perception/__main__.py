"""S1 CLI Module Entrypoint.

Allows running the visual perception subsystem directly via:
    python -m src.visual_perception
"""

from .pipeline import main

if __name__ == "__main__":
    main()

