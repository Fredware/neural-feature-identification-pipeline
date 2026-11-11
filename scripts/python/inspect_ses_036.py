"""Print the structure of the session-36 backup so the figure scripts use
the right names. Reads only metadata (keys, shapes, dtypes), not the data.

Usage (from anywhere, in the project's conda env):
    python inspect_session36.py ~/sfn_session36
"""
from pathlib import Path
import sys

import h5py

root = Path(sys.argv[1] if len(sys.argv) > 1 else "~/sfn_session36").expanduser()


def show_h5(path, max_items=25):
    print(f"\n=== {path.relative_to(root)} ===")
    count = 0

    def visit(name, obj):
        nonlocal count
        if count >= max_items:
            return
        count += 1
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"  {name}/  (group, {len(obj)} items)")
        for k, v in list(obj.attrs.items())[:5]:
            print(f"      attr {k} = {str(v)[:80]}")

    with h5py.File(path, "r") as f:
        for k, v in list(f.attrs.items())[:5]:
            print(f"  root attr {k} = {str(v)[:80]}")
        f.visititems(visit)
    if count >= max_items:
        print(f"  ... (stopped after {max_items} items)")


for p in [root / "events.h5", root / "kinematics.h5",
          *sorted((root / "features").glob("*.h5"))[:1]]:
    show_h5(p)

for cfg in ["3DOF", "6DOF", "8DOF"]:
    for sub in ["decode", "selected"]:
        d = root / cfg / sub
        files = sorted(d.iterdir()) if d.exists() else []
        print(f"\n--- {cfg}/{sub}: {len(files)} files")
        for f in files[:8]:
            print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")

# Peek inside one prediction file and one selection file (first match)
for pattern in ["3DOF/decode/*", "3DOF/selected/*"]:
    hits = [p for p in sorted(root.glob(pattern)) if p.suffix in {".h5", ".hdf5"}]
    if hits:
        show_h5(hits[0])
    else:
        others = sorted(root.glob(pattern))[:1]
        for o in others:
            print(f"\n=== {o.relative_to(root)} (first 15 lines) ===")
            try:
                print("".join(o.open().readlines()[:15]))
            except UnicodeDecodeError:
                print("  (binary, not h5)")


