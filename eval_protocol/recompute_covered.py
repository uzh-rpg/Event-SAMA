import csv, sys, glob, os

run_dir = sys.argv[1]
csv_path = glob.glob(os.path.join(run_dir, "revos_result.csv"))[0]
log_path = glob.glob(os.path.join(os.path.dirname(run_dir), "..", "revos_*.out"))
# out log lives one level up, named revos_<jobid>.out where jobid is the run_dir suffix
jobid = run_dir.rstrip("/").split("_")[-1]
log_path = os.path.join(os.path.dirname(run_dir.rstrip("/")), "..", f"revos_{jobid}.out")
log_path = os.path.normpath(log_path)

missing = set()
with open(log_path, errors="ignore") as f:
    for line in f:
        if " not found" in line:
            vid = line.split(" not found")[0].strip().split()[-1]
            missing.add(vid)

rows = list(csv.DictReader(open(csv_path)))
covered_rows = [r for r in rows if r["video_name"] not in missing]

def summarize(rs, label):
    if not rs:
        print(label, "NO ROWS"); return
    j = sum(float(r["J"]) for r in rs) / len(rs)
    f = sum(float(r["F"]) for r in rs) / len(rs)
    jf = sum(float(r["JF"]) for r in rs) / len(rs)
    print(f"{label}: n={len(rs)} J={j:.2f} F={f:.2f} J&F={jf:.2f}")

print(f"run_dir={run_dir}")
print(f"missing unique video ids: {len(missing)}")
print(f"total csv rows: {len(rows)}")
summarize(rows, "ALL rows (incl. missing-data zeros)")
summarize(covered_rows, "COVERED-ONLY rows (real event data present)")
print(f"covered unique videos: {len(set(r['video_name'] for r in covered_rows))} / {len(set(r['video_name'] for r in rows))}")
