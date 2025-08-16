import os


def fix_path(workdir: str, path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.join(workdir, path)
