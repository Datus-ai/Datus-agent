.PHONY: help clean build install install-dist test check upload-test upload all publish dev-install

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean: ## Clean build artifacts
	python build_scripts/build_pypi_package.py clean

build: ## Build the package
	python build_scripts/build_pypi_package.py build

install: ## Install package locally (editable mode)
	python build_scripts/build_pypi_package.py install

install-dist: ## Install from built distribution
	python build_scripts/build_pypi_package.py install-dist

test: ## Test the installation
	python build_scripts/build_pypi_package.py test

check: ## Check package before upload
	python build_scripts/build_pypi_package.py check

upload-test: ## Upload to Test PyPI
	python build_scripts/build_pypi_package.py upload-test

upload: ## Upload to PyPI
	python build_scripts/build_pypi_package.py upload

all: ## Clean, build, check, and test
	python build_scripts/build_pypi_package.py all

publish: ## Clean, build, check, and upload to PyPI
	python build_scripts/build_pypi_package.py publish

dev-install: ## Install development dependencies
	uv pip install -e ".[dev]"

setup-dev: ## Setup development environment
	uv pip install build twine
	uv pip install -e ".[dev]"

# Quick commands
quick-build: clean build ## Quick build (clean + build)
quick-test: build test ## Quick test (build + test)
quick-publish: clean build check upload ## Quick publish (clean + build + check + upload) 
