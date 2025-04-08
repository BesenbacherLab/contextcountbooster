import sys
from contextcountbooster.cli import main as ccb


def main() -> None:
    sys.exit(ccb(sys.argv[1:]))
