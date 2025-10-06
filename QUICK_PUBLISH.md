# Quick Publishing Guide

**Fast guide to publish SimilarityText to PyPI**

---

## 🚀 Quick Steps

### 1. First Time Setup (One-time only)

```bash
# Install tools
pip install --upgrade pip setuptools wheel twine

# Create PyPI account
# Visit: https://pypi.org/account/register/

# Create API token
# Visit: https://pypi.org/manage/account/
# Copy token (starts with pypi-)
```

### 2. Configure API Token

Create `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE
```

Secure it:
```bash
chmod 600 ~/.pypirc
```

---

## 📦 Publishing (Every Release)

### Option A: Using Script (Recommended)

```bash
# Make script executable (first time only)
chmod +x publish.sh

# Publish version 0.3.0
./publish.sh 0.3.0

# Test on TestPyPI only
./publish.sh 0.3.0 --test-only

# Skip TestPyPI, go directly to PyPI
./publish.sh 0.3.0 --skip-test
```

### Option B: Manual Steps

```bash
# 1. Update version in setup.py
# version="0.3.0"

# 2. Clean and build
rm -rf dist/ build/ *.egg-info
python setup.py sdist bdist_wheel

# 3. Check
python -m twine check dist/*

# 4. Test on TestPyPI (optional)
python -m twine upload --repository testpypi dist/*

# 5. Upload to PyPI
python -m twine upload dist/*

# 6. Create git tag
git tag -a v0.3.0 -m "Release version 0.3.0"
git push origin v0.3.0
```

---

## ✅ Pre-Publishing Checklist

```
[ ] Code tested and working
[ ] Version updated in setup.py
[ ] CHANGELOG.md updated
[ ] All changes committed to git
[ ] README.md reflects latest changes
```

---

## 🧪 Testing Installation

```bash
# After publishing
pip install --upgrade SimilarityText

# Test
python -c "from similarity import Similarity; print('OK')"
```

---

## 🔗 Important Links

- **PyPI Page**: https://pypi.org/project/SimilarityText/
- **Create Token**: https://pypi.org/manage/account/
- **TestPyPI**: https://test.pypi.org/
- **Full Guide**: See PUBLISHING.md

---

## 🆘 Troubleshooting

**Error: "File already exists"**
- Increment version number (can't overwrite)

**Error: "Invalid credentials"**
- Check ~/.pypirc token
- Regenerate token on PyPI

**Build fails**
- Run: `pip install --upgrade setuptools wheel`

---

## 📞 Need Help?

Full documentation: See `PUBLISHING.md`

**Happy Publishing!** 🎉
