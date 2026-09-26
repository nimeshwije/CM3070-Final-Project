"""Makes the project root importable so pytest works no matter where it's run from."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
