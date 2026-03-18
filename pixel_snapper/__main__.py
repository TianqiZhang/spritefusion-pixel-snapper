"""Allow running pixel_snapper as a module: python -m pixel_snapper"""
import sys
from .cli import main

sys.exit(main(sys.argv))
