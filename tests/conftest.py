import sys
from pathlib import Path

print(f"CONFTEST: Adding src to path: {Path(__file__).parent.parent / 'src'}")
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
print("CONFTEST: Done")