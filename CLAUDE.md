# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimilarityText is a Python package that uses AI/NLP techniques to identify semantic similarity between texts and classify text into categories. **Version 0.3.0** adds state-of-the-art transformer support and ML classification.

It provides two main components:

1. **Similarity**: Calculates semantic similarity between two texts
   - Classic: TF-IDF vectorization and cosine similarity
   - Modern: Transformer-based neural embeddings (sentence-transformers)

2. **Classification**: Text classification with multiple methods
   - Basic: Word frequency analysis
   - ML: SVM/Naive Bayes with TF-IDF
   - Advanced: Transformer-based neural classification

## Architecture

### Core Module: similarity/__init__.py

The package contains two main classes:

**Similarity Class**:
- Uses NLTK for tokenization and stemming (RSLPStemmer)
- Applies TF-IDF vectorization with stopword removal
- Calculates cosine similarity between texts
- Supports multiple languages (see LANGUAGES constant for full list)
- Optional automatic language detection via langdetect library
- Constructor parameters:
  - `update`: Download NLTK data packages on initialization (default: True)
  - `language`: Target language for text processing (default: 'english')
  - `langdetect`: Enable automatic language detection (default: False)
  - `nltk_downloads`: Additional NLTK packages to download
  - `quiet`: Suppress NLTK download output (default: True)
- Main method: `similarity(text_a, text_b)` returns float 0.0-1.0 (1.0 = identical)

**Classification Class**:
- Simple word frequency-based classification system
- Uses NLTK tokenization, RSLPStemmer for stemming, and stopword removal
- Learning process builds word frequency corpus per class
- Constructor parameter: `language` (default: 'english')
- Methods:
  - `learning(training_data)`: Train on list of {"class": str, "word": str} dicts
  - `calculate_score(sentence)`: Returns tuple (class_name, score) for input text
  - `calculate_class_score(sentence, class_name)`: Calculate score for specific class

### Text Processing Pipeline

Both classes follow similar preprocessing:
1. Tokenization (nltk.word_tokenize)
2. Stemming (RSLPStemmer for Portuguese, language-dependent)
3. Stopword removal (language-specific)
4. Lowercase normalization
5. Punctuation removal (Similarity class only)

## Installation & Setup

**Install package**:
```bash
pip install SimilarityText
```

**Install from source**:
```bash
pip install -r requeriments.txt
pip install -e .
```

**Build package**:
```bash
python setup.py sdist bdist_wheel
```

## Dependencies

Core dependencies (from requeriments.txt):
- nltk: Natural language processing
- scikit-learn/sklearn: Machine learning algorithms
- scipy: Scientific computing
- langdetect: Automatic language detection
- joblib, regex, tqdm, threadpoolctl: Supporting libraries

## Testing

No formal test suite exists. The `example/` directory contains usage examples:
- `example.py`: Demonstrates Similarity class with Portuguese text variations
- `exemplo2.py`: Demonstrates Classification class with emotional text categories

**Run examples**:
```bash
python example/example.py
python example/exemplo2.py
```

## Language Support

The package includes a comprehensive LANGUAGES list (lines 13-217 in similarity/__init__.py) mapping language codes to full names. While many languages are listed, practical support depends on NLTK's availability of language-specific resources (stopwords, tokenizers).

**Stemming Support**: The package now uses intelligent stemmer selection:
- Portuguese: RSLPStemmer (specialized)
- 16 other languages: SnowballStemmer (Arabic, Danish, Dutch, English, Finnish, French, German, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Turkish)
- Other languages: Simple lowercasing fallback

## Version 0.3.0 Features

### Advanced AI Methods

**Transformer Support** (optional):
- Install with: `pip install sentence-transformers torch transformers`
- Enable: `Similarity(use_transformers=True)` or `Classification(use_transformers=True)`
- Uses sentence-transformers models for semantic embeddings
- Default model: `paraphrase-multilingual-MiniLM-L12-v2`
- Significantly more accurate than TF-IDF, especially for semantic similarity
- Works across languages automatically

**ML Classification**:
- Uses LinearSVC (SVM) by default, falls back to Naive Bayes
- TF-IDF vectorization with preprocessing pipeline
- Enable: `Classification(use_ml=True)`
- Returns confidence scores
- More robust than word frequency method

### Key Bug Fixes

1. Fixed requirements.txt typo (was requeriments.txt)
2. Fixed RSLPStemmer being used for all languages (now language-aware)
3. Fixed crashes when stopwords unavailable for a language
4. Fixed language detection failures on short texts
5. Added punkt_tab to NLTK downloads for newer NLTK versions

### Code Architecture Changes

- `Similarity.similarity()` now accepts `method` parameter ('auto', 'transformer', 'tfidf')
- Split into `__similarity_tfidf()` and `__similarity_transformer()` methods
- Added `__get_stemmer()` helper for language-aware stemming
- `Classification` now has three modes: word frequency, ML, and transformers
- Added `predict()` and `predict_proba()` for sklearn-like API
- Better error handling throughout
