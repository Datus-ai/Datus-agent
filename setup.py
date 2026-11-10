"""Generate dependency-free tool metadata for wheels, editable installs and sdists."""

import runpy
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


def compile_tool_catalog() -> None:
    """Compile declarations without importing the Datus runtime in build isolation."""
    package = Path(__file__).resolve().parent / "datus"
    compiler = runpy.run_path(str(package / "tools" / "registry" / "catalog_builder.py"))
    compiler["write_catalog"](package)


class BuildPy(build_py):
    """Include a complete generated inventory in every Python package build."""

    def run(self):
        """Generate the inventory before copying package files into the wheel."""
        compile_tool_catalog()
        super().run()


class Sdist(sdist):
    """Ship matching source declarations and inventory in source distributions."""

    def run(self):
        """Generate the inventory before collecting source distribution files."""
        compile_tool_catalog()
        super().run()


setup(cmdclass={"build_py": BuildPy, "sdist": Sdist})
