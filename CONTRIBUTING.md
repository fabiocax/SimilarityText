# Contributing to SimilarityText

Thank you for your interest in contributing to SimilarityText! This document provides guidelines and instructions for contributing to the project.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Documentation](#documentation)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for everyone. We pledge to respect all people who contribute through reporting issues, posting feature requests, updating documentation, submitting pull requests, and other activities.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Harassment, trolling, or insulting comments
- Public or private harassment
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

---

## How Can I Contribute?

### 1. Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**How to submit a bug report:**

1. **Use a clear and descriptive title**
2. **Describe the exact steps to reproduce the problem**
3. **Provide specific examples** (code snippets, input data)
4. **Describe the behavior you observed** and what you expected
5. **Include system information**: Python version, OS, library versions

**Bug Report Template:**

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Initialize with '...'
2. Call method '...'
3. Observe error '...'

**Expected behavior**
What you expected to happen.

**Code Example**
```python
from similarity import Similarity
sim = Similarity()
# Your code here
```

**Environment:**
- OS: [e.g., Ubuntu 22.04, Windows 11]
- Python version: [e.g., 3.9.7]
- SimilarityText version: [e.g., 0.3.0]
- Dependencies: [e.g., sentence-transformers 2.2.0]

**Additional context**
Any other information about the problem.
```

### 2. Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues.

**How to submit an enhancement:**

1. **Use a clear and descriptive title**
2. **Provide a detailed description** of the proposed functionality
3. **Explain why this enhancement would be useful**
4. **Provide examples** of how it would be used
5. **Consider backwards compatibility**

**Enhancement Template:**

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Example usage**
```python
# How would users use this feature?
from similarity import NewFeature
feature = NewFeature()
result = feature.do_something()
```

**Additional context**
Any other context or screenshots.
```

### 3. Contributing Code

We welcome code contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [Development Setup](#development-setup) below for details.

### 4. Improving Documentation

Documentation improvements are always welcome:

- Fix typos or clarify existing documentation
- Add examples to the README or API docs
- Improve docstrings in code
- Translate documentation to other languages
- Create tutorials or blog posts

---

## Development Setup

### Prerequisites

- Python 3.7 or higher
- Git
- pip or conda

### Setup Instructions

1. **Fork and clone the repository**

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/SimilarityText.git
cd SimilarityText
```

2. **Create a virtual environment** (recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n similaritytext python=3.9
conda activate similaritytext
```

3. **Install in development mode**

```bash
# Install with all dependencies
pip install -e .[transformers]

# Or minimal installation
pip install -e .
```

4. **Verify installation**

```bash
python -c "from similarity import Similarity, Classification; print('Success!')"
```

5. **Create a feature branch**

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### Project Structure

```
SimilarityText/
├── similarity/          # Main package
│   └── __init__.py     # Core classes and functions
├── example/            # Usage examples
│   ├── example.py
│   ├── exemplo2.py
│   └── example_advanced.py
├── tests/              # Test suite (to be added)
├── docs/               # Documentation (future)
├── README.md          # Main documentation
├── API.md             # API reference
├── CHANGELOG.md       # Version history
├── CONTRIBUTING.md    # This file
├── requirements.txt   # Dependencies
├── setup.py          # Package configuration
└── CLAUDE.md         # Claude Code guidance
```

---

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Single quotes for strings (except docstrings)
- **Naming**:
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_CASE`
  - Private methods: `_leading_underscore` or `__double_leading`

### Code Quality

```python
# Good example
def calculate_similarity(text_a, text_b, method='auto'):
    """
    Calculate similarity between two texts.

    Args:
        text_a (str): First text
        text_b (str): Second text
        method (str): Method to use ('auto', 'tfidf', 'transformer')

    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    if not text_a or not text_b:
        raise ValueError('Both texts must be non-empty')

    # Implementation
    return score
```

### Docstring Format

Use Google-style docstrings:

```python
def method_name(param1, param2):
    """
    Brief description of what the method does.

    Longer description if needed, explaining behavior,
    edge cases, and usage notes.

    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2

    Returns:
        type: Description of return value

    Raises:
        ExceptionType: When this exception is raised

    Example:
        >>> obj = ClassName()
        >>> result = obj.method_name('value1', 'value2')
        >>> print(result)
        expected output
    """
    pass
```

### Type Hints

Use type hints for function signatures:

```python
from typing import List, Dict, Optional, Union, Tuple

def process_texts(
    texts: List[str],
    language: str = 'english',
    options: Optional[Dict[str, any]] = None
) -> List[float]:
    """Process multiple texts and return scores."""
    pass
```

---

## Testing

### Running Examples

Test your changes with the existing examples:

```bash
# Basic examples
python example/example.py
python example/exemplo2.py

# Advanced examples
python example/example_advanced.py
```

### Manual Testing

Create test scripts to verify your changes:

```python
# test_my_feature.py
from similarity import Similarity, Classification

def test_new_feature():
    """Test the new feature."""
    sim = Similarity()
    result = sim.new_method()
    assert result is not None
    print("✓ Test passed")

if __name__ == '__main__':
    test_new_feature()
```

### Future: Unit Tests

We plan to add pytest-based unit tests. Contributions to the test suite are welcome!

```bash
# Future command
pytest tests/
```

---

## Pull Request Process

### Before Submitting

1. **Test your changes** thoroughly
2. **Update documentation** (README, API docs, docstrings)
3. **Check code style** (run linter if available)
4. **Update CHANGELOG.md** with your changes
5. **Ensure backwards compatibility** or document breaking changes

### PR Guidelines

1. **Create a clear title**: `Add transformer caching support` or `Fix #123: Language detection error`

2. **Write a comprehensive description**:
   ```markdown
   ## Description
   Brief description of changes

   ## Motivation
   Why this change is needed

   ## Changes Made
   - Added feature X
   - Fixed bug Y
   - Updated documentation

   ## Testing
   How you tested the changes

   ## Breaking Changes
   None / List breaking changes

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Examples tested
   - [ ] CHANGELOG.md updated
   ```

3. **Link related issues**: Use `Fixes #123` or `Relates to #456`

4. **Keep PRs focused**: One feature or fix per PR

5. **Respond to feedback**: Be open to suggestions and iterate

### Review Process

1. Maintainers will review your PR
2. Automated checks will run (if configured)
3. Reviewer may request changes
4. Once approved, PR will be merged
5. You'll be credited in CHANGELOG and as a contributor

---

## Reporting Bugs

### Security Issues

**Do not** report security vulnerabilities publicly. Email: fabiocax@gmail.com

### Bug Severity

- **Critical**: Crashes, data loss, security issues
- **High**: Major functionality broken
- **Medium**: Feature not working as expected
- **Low**: Minor issues, cosmetic problems

### Information to Include

- Minimal reproducible example
- Expected vs actual behavior
- System information
- Stack trace (if applicable)
- Screenshots (if relevant)

---

## Suggesting Enhancements

### Feature Requests

Good feature requests include:

1. **Clear use case**: Why is this needed?
2. **Proposed API**: How would users interact with it?
3. **Examples**: Show example usage
4. **Alternatives**: What other solutions exist?
5. **Impact**: Who would benefit?

### Ideas Welcome

We're open to ideas for:

- New similarity methods or algorithms
- Additional language support
- Performance optimizations
- API improvements
- Integration with other libraries
- Better documentation

---

## Documentation

### Types of Documentation

1. **Code documentation**: Docstrings and comments
2. **API documentation**: API.md
3. **User guide**: README.md
4. **Examples**: example/ directory
5. **Changelog**: CHANGELOG.md

### Documentation Guidelines

- **Be clear and concise**
- **Include examples** wherever possible
- **Use proper formatting** (Markdown, code blocks)
- **Keep it updated** with code changes
- **Consider beginners** when writing explanations

### Documentation Structure

```markdown
# Section Title

Brief introduction to the section.

## Subsection

Explanation of concept.

### Example

```python
code example
```

**Note**: Important information.
```

---

## Development Workflow

### Standard Workflow

```bash
# 1. Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes
# ... edit files ...

# 4. Test changes
python example/example.py

# 5. Commit changes
git add .
git commit -m "Add: description of changes"

# 6. Push to your fork
git push origin feature/my-feature

# 7. Create pull request on GitHub
```

### Commit Messages

Follow this format:

```
Type: Brief description (50 chars or less)

Longer explanation if needed (wrap at 72 chars).
Explain what and why, not how.

Fixes #123
```

**Types:**
- `Add:` New feature
- `Fix:` Bug fix
- `Docs:` Documentation only
- `Style:` Code style/formatting
- `Refactor:` Code restructuring
- `Test:` Adding tests
- `Chore:` Maintenance tasks

**Examples:**
```
Add: Transformer caching for better performance

Implements LRU cache for transformer embeddings,
reducing computation time for repeated texts.

Fixes #45
```

```
Fix: Language detection error on short texts

Added try-catch block and fallback to default language
when langdetect fails on texts < 20 characters.

Fixes #78
```

---

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open an Issue
- **Feature requests**: Open an Issue with [Feature Request] tag
- **Security issues**: Email fabiocax@gmail.com
- **Other**: Email fabiocax@gmail.com

---

## Recognition

Contributors will be:
- Listed in CHANGELOG.md
- Credited in release notes
- Added to a CONTRIBUTORS file (future)
- Mentioned in the README (for significant contributions)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Thank You!

Thank you for taking the time to contribute to SimilarityText! Your efforts help make this library better for everyone.

**Happy coding!** 🚀
