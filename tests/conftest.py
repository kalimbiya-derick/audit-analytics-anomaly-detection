"""
Ensures the project root is on sys.path so tests can `from modules.x import y`
regardless of where pytest is invoked from.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
