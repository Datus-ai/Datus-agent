# Datus Agent Release Guide

## Local Installation

### Development Mode (Recommended for Development)
```bash
# Install development dependencies
make setup-dev

# Or install directly
pip install -e ".[dev]"
```

### Install from Source
```bash
# Build the package
make build

# Install from the built package
make install-dist
```

### Install from PyPI
```bash
pip install datus-agent
```

## Publishing to PyPI

### 1. Preparation

1. **Register a PyPI Account**
   - Visit https://pypi.org/account/register/
   - Create an account and verify your email

2. **Get an API Token**
   - Log in to PyPI, go to Account Settings
   - Create an API Token (recommended: project-specific token)

3. **Configure Authentication**
   ```bash
   # Copy the config template
   cp .pypirc.template .pypirc
   
   # Edit the config file and fill in your token
   vim .pypirc
   ```

### 2. Publishing Workflow

#### Test Publish (Recommended First)
```bash
# Publish to Test PyPI
make upload-test
```

#### Official Publish
```bash
# Full publish workflow (clean, build, check, upload)
make publish

# Or step by step
make clean
make build
make check
make upload
```

### 3. Verify the Release

After publishing, you can verify with:

```bash
# Create a new virtual environment for testing
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
# or test_env\Scripts\activate  # Windows

# Install the package
pip install datus-agent

# Test import
python -c "import datus; print(datus.__version__)"
```

## Version Management

### Update Version Number

1. Change the version in `datus/__init__.py`
2. Change the version in `pyproject.toml`
3. Commit and tag the change

```bash
# After updating the version
git add .
git commit -m "Bump version to 0.1.6"
git tag v0.1.6
git push origin main --tags
```

### Versioning Convention

Follow [Semantic Versioning](https://semver.org/):

- **Major**: Incompatible API changes
- **Minor**: Backwards-compatible new features
- **Patch**: Backwards-compatible bug fixes

## Common Issues

### 1. Build Fails
```bash
# Make sure build tools are installed
pip install build twine

# Clean and rebuild
make clean
make build
```

### 2. Upload Fails
- Check `.pypirc` configuration
- Make sure your token has upload permission
- Check if the package name is already taken

### 3. Dependency Issues
- Make sure all dependencies are declared in `pyproject.toml`
- Check for version compatibility

### 4. Package Content Issues
- Make sure `MANIFEST.in` or `pyproject.toml` `package-data` is correct
- Check if all necessary files are included

## Automated Release

You can use GitHub Actions for automated release:

1. Set `PYPI_TOKEN` secret in your GitHub repository
2. Create a `.github/workflows/release.yml` workflow
3. Push a tag to trigger the release

## Useful Commands

```bash
# Show help
make help

# Quick build
make quick-build

# Quick test
make quick-test

# Setup development environment
make setup-dev

# Check package content
python -m twine check dist/*

# Show package info
pip show datus-agent
``` 