#!/usr/bin/env python3
"""Search the local datasets/ folder for data matching a topic.

Scans all files in the datasets/ directory (JSONL, JSON, CSV, Parquet, TXT)
and reports what's available, with sample previews and relevance hints.

Usage:
    python search_local.py --topic "math reasoning"
    python search_local.py --topic "code generation" --datasets-dir /path/to/datasets
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)

SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".csv", ".parquet", ".txt", ".tsv"}


def get_file_info(filepath):
    """Get basic info about a data file."""
    path = Path(filepath)
    size_mb = round(path.stat().st_size / (1024 * 1024), 2)
    ext = path.suffix.lower()

    info = {
        "path": str(path),
        "name": path.name,
        "extension": ext,
        "size_mb": size_mb,
        "samples": [],
        "record_count": None,
        "columns": None,
    }

    try:
        if ext in (".jsonl",):
            count = 0
            samples = []
            with open(path) as f:
                for i, line in enumerate(f):
                    if line.strip():
                        count += 1
                        if i < 3:
                            samples.append(json.loads(line.strip()))
            info["record_count"] = count
            info["samples"] = samples
            if samples:
                info["columns"] = list(samples[0].keys())

        elif ext in (".json",):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                info["record_count"] = len(data)
                info["samples"] = data[:3]
                if data:
                    info["columns"] = list(data[0].keys()) if isinstance(data[0], dict) else None
            elif isinstance(data, dict):
                info["columns"] = list(data.keys())

        elif ext in (".csv", ".tsv"):
            import csv
            delimiter = "\t" if ext == ".tsv" else ","
            with open(path) as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                info["columns"] = reader.fieldnames
                samples = []
                for i, row in enumerate(reader):
                    if i >= 3:
                        break
                    samples.append(dict(row))
                info["samples"] = samples
            # Count lines
            with open(path) as f:
                info["record_count"] = sum(1 for _ in f) - 1  # minus header

        elif ext in (".parquet",):
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(path)
                info["record_count"] = table.num_rows
                info["columns"] = table.column_names
                df = table.to_pandas().head(3)
                info["samples"] = df.to_dict(orient="records")
            except ImportError:
                info["note"] = "pyarrow not installed, can't read parquet"

        elif ext in (".txt",):
            with open(path) as f:
                lines = f.readlines()
            info["record_count"] = len(lines)
            info["samples"] = [{"text": l.strip()} for l in lines[:3]]

    except Exception as e:
        info["error"] = str(e)

    return info


def keyword_relevance(file_info, topic_keywords):
    """Score how relevant a file might be to the topic based on name, columns, and sample content."""
    score = 0
    searchable = ""

    # Check filename
    searchable += file_info["name"].lower() + " "

    # Check column names
    if file_info.get("columns"):
        searchable += " ".join(str(c).lower() for c in file_info["columns"]) + " "

    # Check sample content
    for sample in file_info.get("samples", []):
        if isinstance(sample, dict):
            for v in sample.values():
                searchable += str(v).lower()[:200] + " "

    for kw in topic_keywords:
        if kw.lower() in searchable:
            score += 1

    return score


def main():
    parser = argparse.ArgumentParser(description="Search local datasets folder")
    parser.add_argument("--topic", required=True, help="Topic to search for (e.g., 'math reasoning')")
    parser.add_argument("--datasets-dir", default=None, help="Override datasets directory")
    args = parser.parse_args()

    datasets_dir = args.datasets_dir or os.path.join(PROJECT_ROOT, "datasets")

    if not os.path.isdir(datasets_dir):
        logger.info(f"Datasets directory not found: {datasets_dir}")
        print(json.dumps({"found": [], "datasets_dir": datasets_dir, "topic": args.topic}))
        return

    # Find all data files
    data_files = []
    for root, dirs, files in os.walk(datasets_dir):
        # Skip hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                data_files.append(os.path.join(root, fname))

    if not data_files:
        logger.info(f"No data files found in {datasets_dir}")
        print(json.dumps({"found": [], "datasets_dir": datasets_dir, "topic": args.topic}))
        return

    logger.info(f"Found {len(data_files)} data files in {datasets_dir}")

    # Analyze each file
    topic_keywords = args.topic.lower().split()
    results = []

    for fpath in data_files:
        info = get_file_info(fpath)
        info["relevance_score"] = keyword_relevance(info, topic_keywords)
        results.append(info)

    # Sort by relevance (highest first), then by size
    results.sort(key=lambda x: (-x["relevance_score"], -x.get("record_count", 0)))

    # Print summary
    for r in results:
        rel = f"[relevance={r['relevance_score']}]" if r["relevance_score"] > 0 else ""
        count = f"{r['record_count']} records" if r["record_count"] else "unknown size"
        cols = f", columns: {r['columns']}" if r.get("columns") else ""
        logger.info(f"  {r['name']}: {r['size_mb']}MB, {count}{cols} {rel}")

    output = {
        "topic": args.topic,
        "datasets_dir": datasets_dir,
        "total_files": len(results),
        "relevant_files": [r for r in results if r["relevance_score"] > 0],
        "all_files": [{
            "path": r["path"],
            "name": r["name"],
            "extension": r["extension"],
            "size_mb": r["size_mb"],
            "record_count": r["record_count"],
            "columns": r["columns"],
            "relevance_score": r["relevance_score"],
        } for r in results],
    }

    print(f"__LOCAL_SEARCH__:{json.dumps(output, default=str)}")


if __name__ == "__main__":
    main()
