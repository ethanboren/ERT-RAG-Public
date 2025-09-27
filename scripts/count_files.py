"""
Features
--------
• Prompts the user for a folder path.
• Walks through that folder recursively.
• Counts the total number of files.
• Counts how many of them have the “.md” extension (case-insensitive).
• Prints both numbers to the console.
"""

from pathlib import Path
import sys


def count_files(folder: Path) -> tuple[int, int]:
    """Return (total_files, md_files) inside *folder*."""
    total_files = 0
    md_files = 0

    for path in folder.rglob('*'):
        if path.is_file():
            total_files += 1
            if path.suffix.lower() == '.md':
                md_files += 1
    return total_files, md_files


def main() -> None:
    folder_input = input("Folder to analyse: ").strip()
    folder = Path(folder_input).expanduser().resolve()

    if not folder.is_dir():
        print(f"Error: {folder} is not an existing directory.", file=sys.stderr)
        sys.exit(1)

    total, md_total = count_files(folder)
    print(f"Total files found : {total}")
    print(f"Total '.md' files : {md_total}")


if __name__ == "__main__":
    main()
