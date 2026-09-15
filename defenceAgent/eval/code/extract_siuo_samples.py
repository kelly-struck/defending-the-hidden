from __future__ import annotations
"""Extract specific sample items by id from a SIUO evaluation JSON file.

The input JSON is expected to contain a top-level field `details` which is a list
of objects, each having an integer `id` (e.g., 1002, 1004, ...). This script
selects a subset by ids and writes them to a new JSON file, preserving the
original items for those ids in the order you specify.

Usage examples (PowerShell):
  # Pass ids as repeated --id
  python -m defenceAgent.eval.code.extract_siuo_samples `
    --input defenceAgent\eval\results\siuo_gen_output_v6.json `
    --output defenceAgent\eval\results\siuo_failed_subset.json `
    --id 1002 --id 1004 --id 1009

  # Or pass a comma-separated list
  python -m defenceAgent.eval.code.extract_siuo_samples `
    --input defenceAgent\eval\results\siuo_gen_output_v6.json `
    --ids 1002,1004,1009

  # Or load ids from a text file (one id per line)
  python -m defenceAgent.eval.code.extract_siuo_samples `
    --input defenceAgent\eval\results\siuo_gen_output_v6.json `
    --ids_file defenceAgent\eval\results\failed_ids.txt
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Iterable, Set


def load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def flatten_details(obj: Any) -> List[Dict[str, Any]]:
    """Return a flat list of dict items that have an integer 'id' field.
    Prefer top-level 'details' if present; otherwise scan recursively.
    """
    if isinstance(obj, dict):
        # preferred: top-level details
        if isinstance(obj.get("details"), list):
            items = [x for x in obj["details"] if isinstance(x, dict) and "id" in x]
            if items:
                return items
        # generic scan
        out: List[Dict[str, Any]] = []
        for v in obj.values():
            out.extend(flatten_details(v))
        return out
    elif isinstance(obj, list):
        out: List[Dict[str, Any]] = []
        for v in obj:
            out.extend(flatten_details(v))
        return out
    return []


def parse_ids(args: argparse.Namespace) -> List[int]:
    id_list: List[int] = []
    if args.id:
        id_list.extend(args.id)
    if args.ids:
        parts = [s.strip() for s in str(args.ids).split(",") if s.strip()]
        id_list.extend(int(s) for s in parts)
    if args.ids_file:
        txt = Path(args.ids_file).read_text(encoding="utf-8")
        for line in txt.splitlines():
            s = line.strip()
            if s:
                try:
                    id_list.append(int(s))
                except Exception:
                    pass
    # de-duplicate but preserve order
    seen: Set[int] = set()
    ordered: List[int] = []
    for i in id_list:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to SIUO evaluation JSON file")
    ap.add_argument("--output", default="defenceAgent/eval/results/siuo_failed_subset.json", help="Path to write the subset JSON")
    ap.add_argument("--id", type=int, action="append", help="Sample id to extract (can repeat)")
    ap.add_argument("--ids", help="Comma-separated ids, e.g., 1002,1004,1009")
    ap.add_argument("--ids_file", help="Path to a text file with one id per line")
    args = ap.parse_args()

    ids = parse_ids(args)
    if not ids:
        raise SystemExit("No ids provided. Use --id, --ids, or --ids_file.")

    data = load_json(args.input)
    all_items = flatten_details(data)
    # index by id for fast lookup
    by_id: Dict[int, Dict[str, Any]] = {}
    for item in all_items:
        try:
            iid = int(item.get("id"))
        except Exception:
            continue
        by_id[iid] = item

    selected: List[Dict[str, Any]] = []
    missing: List[int] = []
    for iid in ids:
        it = by_id.get(iid)
        if it is None:
            missing.append(iid)
        else:
            selected.append(it)

    out_obj = {
        "source": str(args.input),
        "selected_ids": ids,
        "count": len(selected),
        "missing": missing,
        "items": selected,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "written": args.output,
        "count": len(selected),
        "missing": missing,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
