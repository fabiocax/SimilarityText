# Publishing Guide - PyPI

Complete guide to publish SimilarityText on PyPI (Python Package Index).

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Pre-Publishing Checklist](#pre-publishing-checklist)
- [Step-by-Step Guide](#step-by-step-guide)
- [First Time Publishing](#first-time-publishing)
- [Updating Existing Package](#updating-existing-package)
- [Testing Before Publishing](#testing-before-publishing)
- [Troubleshooting](#troubleshooting)
- [Post-Publishing](#post-publishing)

---

## Prerequisites

### 1. Install Required Tools

```bash
# Install build tools
pip install --upgrade pip setuptools wheel twine

# Verify installation
python -m pip --version
python -m twine --version
```

### 2. Create PyPI Accounts

**PyPI (Production)**:
- Visit: https://pypi.org/account/register/
- Verify your email
- Enable 2FA (recommended)

**TestPyPI (Testing)**:
- Visit: https://test.pypi.org/account/register/
- Verify your email
- Use for testing before production

### 3. Configure API Tokens (Recommended)

**Create API Token on PyPI**:
1. Go to https://pypi.org/manage/account/
2. Scroll to "API tokens"
3. Click "Add API token"
4. Name: "SimilarityText-upload"
5. Scope: "Project: SimilarityText" (after first upload) or "Entire account"
6. Copy the token (starts with `pypi-`)

**Create `.pypirc` file**:

```bash
# Create config file
nano ~/.pypirc
```

Add this content:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

Secure the file:
```bash
chmod 600 ~/.pypirc
```

---

## Pre-Publishing Checklist

### ✅ Required Files

- [x] `setup.py` - Package configuration
- [x] `requirements.txt` - Dependencies
- [x] `README.md` - Main documentation
- [x] `LICENSE` - License file
- [x] `MANIFEST.in` - Include non-Python files
- [x] `similarity/__init__.py` - Main code
- [x] `CHANGELOG.md` - Version history

### ✅ Version Checklist

- [ ] Update version in `setup.py`
- [ ] Update version in `CHANGELOG.md`
- [ ] Add release notes to `CHANGELOG.md`
- [ ] Update `README.md` if needed
- [ ] Commit all changes
- [ ] Create git tag

### ✅ Code Quality

- [ ] Code tested and working
- [ ] Examples run without errors
- [ ] No sensitive information (API keys, passwords)
- [ ] Dependencies correctly specified
- [ ] Documentation up to date

---

## Step-by-Step Guide

### Step 1: Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf similarity.egg-info/
rm -rf SimilarityText.egg-info/

# Verify clean state
ls -la
```

### Step 2: Update Version Number

Edit `setup.py`:

```python
setup(
    name="SimilarityText",
    version="0.3.0",  # Update this!
    # ...
)
```

### Step 3: Build Distribution Packages

```bash
# Build source distribution and wheel
python setup.py sdist bdist_wheel

# Verify build
ls -la dist/
# Should see:
# - SimilarityText-0.3.0.tar.gz (source)
# - SimilarityText-0.3.0-py3-none-any.whl (wheel)
```

### Step 4: Check Package Quality

```bash
# Check distribution
python -m twine check dist/*

# Should output:
# Checking dist/SimilarityText-0.3.0.tar.gz: PASSED
# Checking dist/SimilarityText-0.3.0-py3-none-any.whl: PASSED
```

### Step 5: Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Install from TestPyPI to test
pip install --index-url https://test.pypi.org/simple/ --no-deps SimilarityText

# Test installation
python -c "from similarity import Similarity, Classification; print('Success!')"
```

### Step 6: Upload to PyPI (Production)

```bash
# Upload to PyPI
python -m twine upload dist/*

# Enter credentials if not using API token
# Username: __token__
# Password: your-api-token
```

### Step 7: Verify on PyPI

Visit: https://pypi.org/project/SimilarityText/

Check:
- ✅ Package appears
- ✅ Version is correct
- ✅ README renders properly
- ✅ Links work
- ✅ Dependencies listed

### Step 8: Test Installation

```bash
# Create new virtual environment
python -m venv test_env
source test_env/bin/activate

# Install from PyPI
pip install SimilarityText

# Test
python -c "from similarity import Similarity; print('Success!')"

# Test with transformers
pip install SimilarityText[transformers]
```

### Step 9: Create Git Tag

```bash
# Create annotated tag
git tag -a v0.3.0 -m "Release version 0.3.0 - Advanced AI Support"

# Push tag to GitHub
git push origin v0.3.0

# Push all tags
git push --tags
```

---

## First Time Publishing

If this is your **first time** publishing SimilarityText:

### 1. Choose Package Name

Verify name availability:
- Visit: https://pypi.org/project/SimilarityText/
- If taken, choose different name in `setup.py`

### 2. Initial Upload

```bash
# Clean build
rm -rf dist/ build/ *.egg-info

# Build
python setup.py sdist bdist_wheel

# Check
python -m twine check dist/*

# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ SimilarityText

# If all works, upload to PyPI
python -m twine upload dist/*
```

### 3. Configure API Token for Project

After first upload:
1. Go to https://pypi.org/manage/project/SimilarityText/settings/
2. Create project-specific API token
3. Update `.pypirc` with project token

---

## Updating Existing Package

### For New Versions (e.g., 0.3.0 → 0.3.1)

```bash
# 1. Update version
# Edit setup.py: version="0.3.1"

# 2. Update changelog
# Edit CHANGELOG.md

# 3. Commit changes
git add setup.py CHANGELOG.md
git commit -m "Bump version to 0.3.1"

# 4. Clean and build
rm -rf dist/ build/ *.egg-info
python setup.py sdist bdist_wheel

# 5. Check
python -m twine check dist/*

# 6. Upload
python -m twine upload dist/*

# 7. Tag release
git tag -a v0.3.1 -m "Release version 0.3.1"
git push && git push --tags
```

### Automation Script

Save as `publish.sh`:

```bash
#!/bin/bash

# Publishing script for SimilarityText

set -e  # Exit on error

echo "🚀 SimilarityText Publishing Script"
echo "===================================="

# Check for version argument
if [ -z "$1" ]; then
    echo "❌ Error: Version number required"
    echo "Usage: ./publish.sh 0.3.1"
    exit 1
fi

VERSION=$1

echo "📦 Publishing version: $VERSION"

# 1. Clean old builds
echo "🧹 Cleaning old builds..."
rm -rf dist/ build/ *.egg-info

# 2. Update version in setup.py (manual check)
echo "⚠️  Make sure setup.py has version='$VERSION'"
read -p "Press enter to continue..."

# 3. Build
echo "🔨 Building distribution packages..."
python setup.py sdist bdist_wheel

# 4. Check
echo "✅ Checking package..."
python -m twine check dist/*

# 5. Test on TestPyPI (optional)
read -p "Upload to TestPyPI first? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Uploading to TestPyPI..."
    python -m twine upload --repository testpypi dist/*
    echo "✅ Uploaded to TestPyPI"
    echo "Test with: pip install --index-url https://test.pypi.org/simple/ SimilarityText"
    read -p "Press enter to continue to PyPI..."
fi

# 6. Upload to PyPI
read -p "Upload to PyPI? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Uploading to PyPI..."
    python -m twine upload dist/*
    echo "✅ Published to PyPI!"
    echo "🎉 Version $VERSION is now live!"

    # 7. Git tag
    read -p "Create git tag? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git tag -a "v$VERSION" -m "Release version $VERSION"
        git push origin "v$VERSION"
        echo "✅ Git tag created and pushed"
    fi
else
    echo "❌ Cancelled upload to PyPI"
fi

echo "✨ Done!"
```

Make executable:
```bash
chmod +x publish.sh
```

Usage:
```bash
./publish.sh 0.3.1
```

---

## Testing Before Publishing

### Local Installation Test

```bash
# Install in editable mode
pip install -e .

# Run examples
python example/example.py
python example/example_advanced.py

# Test imports
python -c "from similarity import Similarity, Classification"
```

### Test Build Locally

```bash
# Build
python setup.py sdist bdist_wheel

# Install from local build
pip install dist/SimilarityText-0.3.0-py3-none-any.whl

# Test
python -c "from similarity import Similarity; print('OK')"
```

### TestPyPI Full Test

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Create clean environment
python -m venv test_env
source test_env/bin/activate

# Install from TestPyPI (with dependencies from PyPI)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ SimilarityText

# Run tests
python example/example.py
```

---

## Troubleshooting

### Error: "File already exists"

**Problem**: Version already uploaded to PyPI
**Solution**: Bump version number (can't overwrite existing versions)

```bash
# Update version in setup.py
version="0.3.1"  # Increment version
```

### Error: "Invalid distribution"

**Problem**: Missing required files or metadata
**Solution**: Check setup.py and run twine check

```bash
python -m twine check dist/*
```

### Error: "Authentication failed"

**Problem**: Wrong credentials or API token
**Solution**: Check `.pypirc` or regenerate token

```bash
# Test authentication
python -m twine upload --repository testpypi dist/* --verbose
```

### Error: "README rendering failed"

**Problem**: Invalid Markdown or reStructuredText
**Solution**: Validate README on GitHub first, or check locally

```bash
# Install readme-renderer
pip install readme-renderer

# Check README
python -m readme_renderer README.md
```

### Warning: "Unknown field in setup()"

**Problem**: Unsupported parameter in setup.py
**Solution**: Remove or update deprecated parameters

### Large Package Size

**Problem**: Build includes unnecessary files
**Solution**: Create `.pypiignore` or update `MANIFEST.in`

```bash
# Check package contents
tar -tzf dist/SimilarityText-0.3.0.tar.gz

# Exclude files in MANIFEST.in
```

---

## Post-Publishing

### 1. Announce Release

**GitHub Release**:
1. Go to: https://github.com/fabiocax/SimilarityText/releases
2. Click "Draft a new release"
3. Tag: `v0.3.0`
4. Title: "SimilarityText v0.3.0 - Advanced AI Support"
5. Description: Copy from CHANGELOG.md
6. Publish release

**Social Media** (optional):
- Twitter/X
- LinkedIn
- Reddit (r/Python, r/MachineLearning)
- Dev.to
- Hacker News

### 2. Update Documentation Links

Verify all links work:
- PyPI page: https://pypi.org/project/SimilarityText/
- GitHub repo: https://github.com/fabiocax/SimilarityText
- Documentation in README

### 3. Monitor

**First 24 hours**:
- Check PyPI page renders correctly
- Monitor download stats
- Watch for GitHub issues
- Check installation works on different platforms

**Tools**:
- PyPI Stats: https://pypistats.org/packages/similaritytext
- Libraries.io: https://libraries.io/pypi/SimilarityText

### 4. Support Users

- Respond to GitHub issues
- Answer questions
- Fix urgent bugs with patch releases (0.3.1)

---

## Version Numbering Guide

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.3.0): New features, backwards-compatible
- **PATCH** (0.3.1): Bug fixes, backwards-compatible

**Examples**:
- `0.3.0` → `0.3.1`: Bug fixes
- `0.3.0` → `0.4.0`: New features
- `0.3.0` → `1.0.0`: Breaking changes

---

## Quick Reference

### One-Line Commands

```bash
# Clean, build, check, upload
rm -rf dist/ build/ *.egg-info && python setup.py sdist bdist_wheel && python -m twine check dist/* && python -m twine upload dist/*

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ SimilarityText

# Install from PyPI
pip install SimilarityText
```

---

## Security Notes

⚠️ **Never commit**:
- `.pypirc` file
- API tokens
- Passwords

✅ **Always**:
- Use API tokens (not passwords)
- Use 2FA on PyPI
- Set correct permissions: `chmod 600 ~/.pypirc`
- Regenerate tokens if compromised

---

## Additional Resources

- **PyPI Help**: https://pypi.org/help/
- **Packaging Guide**: https://packaging.python.org/
- **Twine Documentation**: https://twine.readthedocs.io/
- **setuptools Documentation**: https://setuptools.pypa.io/

---

## Checklist: First Release

```
Pre-Publishing:
[ ] All code tested
[ ] Documentation complete
[ ] LICENSE file added
[ ] MANIFEST.in created
[ ] Version number set
[ ] CHANGELOG updated
[ ] Git committed

Building:
[ ] Old builds cleaned
[ ] Package built (sdist + wheel)
[ ] Build checked with twine
[ ] TestPyPI upload successful
[ ] TestPyPI install tested

Publishing:
[ ] PyPI upload successful
[ ] PyPI page checked
[ ] Installation from PyPI tested
[ ] Git tag created
[ ] GitHub release created

Post-Publishing:
[ ] Installation instructions verified
[ ] README renders on PyPI
[ ] Links working
[ ] Downloads monitored
```

---

**Ready to publish?** Follow the [Step-by-Step Guide](#step-by-step-guide) above!

**Questions?** Open an issue on GitHub or email: fabiocax@gmail.com
