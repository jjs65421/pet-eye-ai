"""Read original annotations; write the confirmed index inside the workspace."""
import argparse
import json
import os
from collections import Counter
from pathlib import Path
from screening_data import DEFAULT_DATA_ROOT, MANIFEST_PATH, DISEASES, annotation_record


def build(data_root=DEFAULT_DATA_ROOT, output=MANIFEST_PATH):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pending.jsonl")
    totals, visited = Counter(), set()
    old_index = Path(data_root) / "labels_manifest.jsonl"

    def candidates():
        if old_index.is_file():
            # The old index locates files; its binary targets are never reused.
            with old_index.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    if row["disease"] in DISEASES:
                        yield Path(row["image_path"]).with_suffix(".json"), row["split"]
        else:
            for split in ("training", "validation"):
                for path in (Path(data_root) / split / "labels").rglob("*.json"):
                    yield path, split

    with temporary.open("w", encoding="utf-8") as destination:
        for index, (path, split) in enumerate(candidates(), 1):
            if str(path) in visited:
                continue
            visited.add(str(path))
            try:
                row = annotation_record(path, split)
                if row and row["disease"] in DISEASES:
                    destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                    totals[f"{split}|{row['disease']}|{row['device']}|{row['abnormal']}"] += 1
                else:
                    totals["excluded"] += 1
            except (OSError, UnicodeError, ValueError, KeyError) as error:
                totals["read_errors"] += 1
                if totals["read_errors"] <= 5:
                    print(f"annotation_error={path}: {error}", flush=True)
            if index % 10000 == 0:
                print(f"annotations_read={index}", flush=True)
    os.replace(temporary, output)
    output.with_suffix(".summary.json").write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"confirmed_manifest={output}; counts={dict(totals)}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    build(args.data_root, args.output)


if __name__ == "__main__":
    main()
