"""Associate synthetic variants with the slide IDs of their reference patches."""

import argparse
import csv
import re
from pathlib import Path

VARIANT = re.compile(r"cond_(\d+)_(\d+)\.(?:jpe?g|png|tiff?|bmp)", re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--synthetic_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.references.open(newline="") as f:
        references = list(csv.DictReader(f))
    columns = ["image_paths", "slide_id"]
    if not references or any(not row.get(key) for row in references for key in columns):
        parser.error("References require nonempty image_paths and slide_id columns.")
    if not args.synthetic_dir.is_dir():
        parser.error("--synthetic_dir must point to an image directory.")
    if args.references.resolve() == args.output.resolve():
        parser.error("Use a separate output path for the training manifest.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + ".tmp")
    generated = 0
    try:
        with temporary_output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in references:
                writer.writerow({key: row[key] for key in columns})
            for path in args.synthetic_dir.iterdir():
                match = VARIANT.fullmatch(path.name)
                if not path.is_file() or match is None:
                    continue
                index = int(match[1])
                if index >= len(references):
                    raise ValueError(f"{path.name} refers to an index outside the reference CSV")
                writer.writerow({"image_paths": str(path), "slide_id": references[index]["slide_id"]})
                generated += 1
            if not generated:
                raise ValueError("No generated cond_<reference index>_<variant index> images found")
        temporary_output.replace(args.output)
    finally:
        temporary_output.unlink(missing_ok=True)
    print(f"Wrote {len(references):,} real and {generated:,} synthetic paths to {args.output}")


if __name__ == "__main__":
    main()
