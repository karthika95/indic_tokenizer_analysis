#!/usr/bin/env python3
import re
from pathlib import Path
import shutil
import sys

def combine_clusters(cluster_def_path, source_dir, dest_dir):
    cluster_def_path = Path(cluster_def_path)
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True, parents=True)

    cluster_re = re.compile(r"Cluster\s+(\d+)\s*:\s*(.+)", re.I)

    clusters = {}
    with cluster_def_path.open(encoding="utf-8") as f:
        for line in f:
            m = cluster_re.match(line.strip())
            if not m:
                continue
            cid, langs = m.groups()
            lang_list = [l.strip() for l in langs.split(",") if l.strip()]
            clusters[int(cid)] = lang_list

    for cid, langs in clusters.items():
        out_path = dest_dir / f"cluster_{cid}.txt"
        with out_path.open("wb") as fout:
            for lang in langs:
                in_path = source_dir / f"{lang}.txt"
                if not in_path.exists():
                    print(f"[WARN] {in_path} missing – skipped.")
                    continue
                with in_path.open("rb") as fin:
                    shutil.copyfileobj(fin, fout)
        print(f"[✓] Wrote {out_path} (cluster {cid} ↔ {', '.join(langs)})")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <cluster_def.txt> <source_dir> <dest_dir>")
        sys.exit(1)
    
    cluster_def = sys.argv[1]
    source_dir = sys.argv[2]
    dest_dir = sys.argv[3]
    
    combine_clusters(cluster_def, source_dir, dest_dir)