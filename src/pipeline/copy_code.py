"""Copy the code tree into a package/test root (filtered, no caches/data)."""
import os
import shutil

CODE_EXTS = {".py", ".md", ".txt", ".json", ".ps1", ".yaml", ".yml", ".cfg", ".toml"}
SKIP_DIRS = {"__pycache__", "cache", "out", "tmp", "reports"}
CODE_DIRS = ["lib", "task1", "task2", "task3", "task4", "task2_seq", "task2_cnn",
             "pipeline", "tools", "setup", "report"]
FULL_DIRS = ["task2/models", "task2_seq/models", "task1/unet/models_deploy",
             "task1/unet/deploy_npz"]


def copy_code(src_work, dst_root):
    """Copy code files from src_work into dst_root/work preserving layout."""
    n = 0
    for d in CODE_DIRS:
        root = os.path.join(src_work, d)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS and not x.startswith("cache")]
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() not in CODE_EXTS:
                    continue
                if fn.startswith("cache_") or fn.endswith(".log"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn), src_work)
                dst = os.path.join(dst_root, "work", rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(dirpath, fn), dst)
                n += 1
    return n


def copy_full_dirs(src_work, dst_root):
    n = 0
    for d in FULL_DIRS:
        root = os.path.join(src_work, d)
        if not os.path.isdir(root):
            continue
        dst = os.path.join(dst_root, "work", d)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for fn in filenames:
                src = os.path.join(dirpath, fn)
                rel = os.path.relpath(src, src_work)
                out = os.path.join(dst_root, "work", rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copy2(src, out)
                n += 1
    return n

if __name__ == "__main__":
    import sys
    src = sys.argv[1]
    dst = sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    a = copy_code(src, dst)
    b = copy_full_dirs(src, dst)
    print(f"copied {a} code files and {b} model/npz files")
