"""Move experiment artifacts beside the existing HDD results; keep their project paths."""

import argparse
import ctypes
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("analysis", "figs", "figs_outlier", "activation_dumps_clean",
        "activation_dump", "snapshots", "logs")
LOCKS = ("dlm_sandwich_flute_wait.lock", "dlm_sandwich_flute_2gpu.lock",
         "dlm_sandwich_paired_2gpu.lock", "flute_shape_ab.lock")
# These legacy writers call savefig/torch.save directly on a filename in cwd.
FIXED = ({f"fig_{tag}_{metric}.png"
          for tag in ("baseline", "diff", "exp12", "exp13", "exp13_diff")
          for metric in ("max", "norm", "var")}
         | {"paper_style_activation.png", "Layer1_Attn_q_proj.png",
            "Layer27_Attn_out.png", "Layer31_FFN_ff_out.png",
            "layer1_q_proj.pt", "layer31_ff_proj.pt", "layer31_up_proj.pt"})
GLOBS = ("*.log", "*.png", "*.csv", "*.jsonl", "layer*_*.pt", "nohup.out",
         "final_metrics_summary.txt", "final_log_summary.txt*", "module_names.txt",
         "llada_after_exp62.txt", "llada_generate_stable_part.txt")


def hdd_base(root):
    bases = []
    for name in ("results", "step_logs"):
        src = root / name
        if not src.is_symlink() or not src.is_dir():
            raise RuntimeError(f"Expected an existing HDD directory link: {src}")
        target = src.resolve(strict=True)
        if target.name != name or target.stat().st_dev == root.stat().st_dev:
            raise RuntimeError(f"Unexpected destination or same filesystem: {src} -> {target}")
        bases.append(target.parent)
    if bases[0] != bases[1]:
        raise RuntimeError("results and step_logs point to different output roots")
    return bases[0]


def plan(root, base):
    loose = base / "root_outputs"
    if loose.is_symlink() or (loose.exists() and not loose.is_dir()):
        raise RuntimeError(f"Not an ordinary destination directory: {loose}")
    names = FIXED | {p.name for pattern in GLOBS for p in root.glob(pattern)}
    entries = [(root / n, base / n, True) for n in DIRS]
    entries += [(root / n, loose / n, False) for n in sorted(names)]
    pending = []
    for src, dst, directory in entries:
        if dst.is_symlink():
            raise RuntimeError(f"Unexpected symlink at destination: {dst}")
        if src.is_symlink():
            if src.resolve() != dst or (directory and not dst.is_dir()) or (dst.exists() and dst.is_dir() != directory):
                raise RuntimeError(f"Unexpected link: {src} -> {os.readlink(src)}")
            continue
        if dst.exists():
            raise RuntimeError(f"Destination already exists; not merging or overwriting: {dst}")
        if src.exists() and not (src.is_dir() if directory else src.is_file()):
            raise RuntimeError(f"Unexpected source type: {src}")
        if src.is_dir():
            # An escaping relative link would change meaning after moving its parent.
            for child in src.rglob("*"):
                if child.is_symlink():
                    to = Path(os.readlink(child))
                    if not to.is_absolute() and not Path(os.path.abspath(child.parent / to)).is_relative_to(src):
                        raise RuntimeError(f"Review external relative symlink before moving: {child}")
        pending.append((src, dst, directory))
    return pending


def hold_locks(logs, stack):
    names = set(LOCKS) | {p.name for p in logs.glob("*.lock")}
    for name in sorted(names):
        f = stack.enter_context((logs / name).open("a"))
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"An experiment or waiter holds {name}; retry after it ends") from None


def check_writers(paths):
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            for fd in (proc / "fd").iterdir():
                try:
                    target = Path(os.readlink(fd))
                    # Protocol lock descriptors are checked with flock on both
                    # inode sets; merely opening a lock file is not a data write.
                    if target.suffix == ".lock" and target.parent.name == "logs":
                        continue
                        continue
                    if not any(target == p or p in target.parents for p in paths):
                        continue
                    info = (proc / "fdinfo" / fd.name).read_text()
                    flags = next(int(l.split()[1], 8) for l in info.splitlines() if l.startswith("flags:"))
                    if flags & os.O_ACCMODE:
                        raise RuntimeError(f"PID {proc.name} still has {target} open for writing")
                except (FileNotFoundError, ProcessLookupError):
                    continue
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def rename_linux(a, b, flags):
    # NOREPLACE=1 for publishing staged data; EXCHANGE=2 to switch without a gap.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.renameat2(-100, os.fsencode(a), -100, os.fsencode(b), flags) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), f"{a} -> {b}")


def remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def relocate(src, dst, directory, stack):
    dst.parent.mkdir(exist_ok=True)
    if not src.exists():
        if directory:
            dst.mkdir()
        # For a not-yet-produced root file, create only the route, never dummy data.
        src.symlink_to(dst)
        return
    print(f"Copying and verifying: {src.name} -> {dst}", flush=True)
    with tempfile.TemporaryDirectory(dir=dst.parent, prefix=".qdlm-stage-") as tmp:
        stage = Path(tmp) / "data"
        subprocess.run(["cp", "-a", "--", str(src), str(stage)], check=True)
        if src.name == "logs":
            # Hold the copied lock inodes too, across the directory handoff.
            hold_locks(stage, stack)
        check_writers([src])
        subprocess.run(["diff", "-qr", "--no-dereference", "--", str(src), str(stage)], check=True)
        rename_linux(stage, dst, 1)
    backup_dir = Path(tempfile.mkdtemp(dir=src.parent, prefix=".qdlm-handoff-"))
    old = backup_dir / src.name
    old.symlink_to(dst)
    try:
        rename_linux(src, old, 2)
    except BaseException:
        print(f"Source intact; verified HDD copy at {dst}", flush=True)
        raise
    try:
        # Detect writes/changes during the copy or switch; keep the original on failure.
        check_writers([old])
        subprocess.run(["diff", "-qr", "--no-dereference", "--", str(old), str(dst)], check=True)
    except BaseException:
        print(f"Original retained at {old}; HDD copy at {dst}. Inspect both.", flush=True)
        raise
    remove(old)
    backup_dir.rmdir()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="Perform the migration; default is preview")
    args = ap.parse_args()
    base = hdd_base(ROOT)
    print(f"HDD output root: {base}", flush=True)
    pending = plan(ROOT, base)
    if not args.apply:
        for src, dst, _ in pending:
            print(f"{'Move' if src.exists() else 'Route future writes'}: {src.name} -> {dst}")
        return
    if pending:
        with ExitStack() as stack:
            f = stack.enter_context((base / ".qdlm-output-migration.lock").open("a"))
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            hold_locks(ROOT / "logs", stack)
            pending = plan(ROOT, base)
            check_writers([s for s, _, _ in pending])
            for src, dst, directory in pending:
                relocate(src, dst, directory, stack)
    if plan(ROOT, base):
        raise RuntimeError("A new output appeared during migration; rerun after writers finish")
    print(f"Verified HDD output links: {', '.join(('results', 'step_logs') + DIRS)}")
    print(f"Loose outputs and legacy fixed filenames: {base / 'root_outputs'}")
    print("Migration complete.")


if __name__ == "__main__":
    main()
