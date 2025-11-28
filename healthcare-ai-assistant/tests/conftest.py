import sys
from pathlib import Path

# Add the project root to sys.path so tests can import main, ingest, etc.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
