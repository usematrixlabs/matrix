import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
tests_path = Path(__file__).parent
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(tests_path))