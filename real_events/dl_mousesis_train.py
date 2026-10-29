# Download MouseSIS train split (public GT) from the official Google Drive folder, one file at a time.
import json, os, gdown
ids = json.load(open("mousesis_train_ids.json"))
out = "mousesis_train_raw"
for k, fid in sorted(ids.items()):
    dst = os.path.join(out, os.path.basename(k))
    if os.path.exists(dst):
        continue
    gdown.download(id=fid, output=dst + ".part", quiet=True)
    os.rename(dst + ".part", dst)
    print(k, round(os.path.getsize(dst) / 1e9, 2), "GB", flush=True)
print("DONE", flush=True)
