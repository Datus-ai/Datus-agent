"""Temporary, one-use sealing of the three authorized legacy Nightly configs.

Only ciphertext leaves the runner. Remove this helper after migration.
https://docs.github.com/en/rest/guides/encrypting-secrets-for-the-rest-api
"""

import base64
import json
import os
import re
import stat
import sys
from pathlib import Path

SOURCE_ROOT = Path("/home/datus_ci/ci_template")
CONFIGS = {
    "agent.yml": "DATUS_NIGHTLY_AGENT_CONFIG",
    "ci.env": "DATUS_NIGHTLY_CI_ENV",
    "nightly.env": "DATUS_NIGHTLY_SERVICE_ENV",
}
MAX_BYTES = 48 * 1024


def seal_configs(public_key: str, key_id: str, output: Path, *, source_root: Path = SOURCE_ROOT) -> None:
    """The source override is internal for synthetic tests, never a CLI input."""
    from nacl.public import PublicKey, SealedBox

    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("Configuration source must be a real directory")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", key_id):
        raise ValueError("Invalid public key identifier")
    key = PublicKey(base64.b64decode(public_key, validate=True))
    values = []
    # Validate the complete input set before creating any output file.
    for filename, name in CONFIGS.items():
        fd = os.open(source_root / filename, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("Configuration must be a regular file")
            value = handle.read(MAX_BYTES + 1)
        if not value or len(value) > MAX_BYTES or not value.decode("utf-8").strip():
            raise ValueError("Configuration must be nonempty UTF8 within the GitHub secret limit")
        values.append((name, value))
    box = SealedBox(key)
    rows = [
        {"name": name, "encrypted_value": base64.b64encode(box.encrypt(value)).decode("ascii"), "key_id": key_id}
        for name, value in values
    ]
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    output.parent.chmod(0o700)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def main() -> int:
    try:
        if len(sys.argv) != 1:
            raise ValueError("This helper accepts no command-line arguments")
        seal_configs(
            os.environ["NIGHTLY_SEAL_PUBLIC_KEY"],
            os.environ["NIGHTLY_SEAL_PUBLIC_KEY_ID"],
            Path(os.environ["RUNNER_TEMP"]) / "nightly-config-sealed" / "sealed-config.json",
        )
    except Exception:
        # Never render exception text: decoding errors can contain input bytes.
        print("Nightly configuration sealing failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
