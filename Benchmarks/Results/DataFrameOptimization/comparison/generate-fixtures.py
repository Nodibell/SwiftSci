"""Generate the recorded deterministic CSV fixtures and verify their bytes."""
from pathlib import Path
import hashlib
hashes = {100000: "e9918fca3c64cd23df830a7f48b1aeff0e41a06d6eaaff4a1530371bdf03060d",
          1000000: "a72e18ea5a198ba174a2097686ade2472de671fbd10b9f270bf9e25535c44cf9"}
folder = Path(__file__).resolve().parent / "data"
folder.mkdir(exist_ok=True)
for n, expected in hashes.items():
    path = folder / f"data-{n}.csv"
    with path.open("w", newline="\n") as f:
        f.write("id,group,value\n")
        for i in range(n):
            value = "" if i % 101 == 0 else str(((i * 7919) % 100003) / 100.0)
            f.write(f"{i},{i % 100},{value}\n")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Fixture mismatch for {path.name}: {actual}")
    print(f"{path.name}: {actual}")
