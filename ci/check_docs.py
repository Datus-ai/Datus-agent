"""Build both documented locales with MkDocs strict validation."""

import os
import subprocess
import sys
import tempfile


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="datus-docs-") as directory:
        for locale in ("en", "zh"):
            subprocess.run(
                [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", f"{directory}/{locale}"],
                env={**os.environ, "DOCS_LOCALE": locale},
                check=True,
            )


if __name__ == "__main__":
    main()
