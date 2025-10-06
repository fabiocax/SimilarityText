# API Reference

Complete API documentation for SimilarityText v0.3.0

---

## Table of Contents

- [Similarity Class](#similarity-class)
  - [Constructor](#similarity-constructor)
  - [Methods](#similarity-methods)
- [Classification Class](#classification-class)
  - [Constructor](#classification-constructor)
  - [Methods](#classification-methods)
- [Data Structures](#data-structures)
- [Exceptions](#exceptions)
- [Constants](#constants)

---

## Similarity Class

The `Similarity` class provides text similarity calculation using TF-IDF or transformer-based methods.

### Similarity Constructor

```python
Similarity(
    update=True,
    language='english',
    langdetect=False,
    nltk_downloads=[],
    quiet=True,
    use_transformers=False,
    model_name='paraphrase-multilingual-MiniLM-L12-v2'
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `update` | `bool` | `True` | Download required NLTK data packages on initialization. Set to `False` to skip downloads if data is already available. |
| `language` | `str` | `'english'` | Default language for text processing. Affects tokenization, stopwords, and stemming. See [supported languages](#supported-languages). |
| `langdetect` | `bool` | `False` | Enable automatic language detection. If `True`, will detect and use the language of input texts. May override the `language` parameter. |
| `nltk_downloads` | `list[str]` | `[]` | Additional NLTK data packages to download. Useful for custom language resources. |
| `quiet` | `bool` | `True` | Suppress informational messages and warnings. Set to `False` for debugging. |
| `use_transformers` | `bool` | `False` | Enable transformer-based similarity calculation. Requires `sentence-transformers` package. Provides higher accuracy but slower performance. |
| `model_name` | `str` | `'paraphrase-multilingual-MiniLM-L12-v2'` | Name of the sentence-transformer model to use. Only relevant when `use_transformers=True`. See [available models](#available-models). |

#### Returns

Returns a `Similarity` instance.

#### Raises

- `ImportError`: If `use_transformers=True` but `sentence-transformers` is not installed. Falls back to TF-IDF method with a warning.

#### Example

```python
from similarity import Similarity

# Basic usage
sim = Similarity()

# With language detection
sim = Similarity(langdetect=True)

# With transformers
sim = Similarity(use_transformers=True)

# Custom configuration
sim = Similarity(
    language='portuguese',
    quiet=False,
    use_transformers=True,
    model_name='all-mpnet-base-v2'
)
```

---

### Similarity Methods

#### similarity()

Calculate semantic similarity between two texts.

```python
similarity(text_a, text_b, method='auto')
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_a` | `str` | Required | First text to compare |
| `text_b` | `str` | Required | Second text to compare |
| `method` | `str` | `'auto'` | Method selection: `'auto'`, `'tfidf'`, or `'transformer'` |

##### Method Options

- **`'auto'`**: Automatically selects the best available method. Uses transformers if initialized with `use_transformers=True`, otherwise falls back to TF-IDF.
- **`'tfidf'`**: Forces TF-IDF method regardless of initialization.
- **`'transformer'`**: Forces transformer method. Requires `use_transformers=True` during initialization.

##### Returns

`float`: Similarity score between 0.0 and 1.0, where:
- `0.0` = completely different texts
- `1.0` = identical texts
- Higher values = more similar

##### Raises

- `Exception`: If `langdetect=True` and detected languages differ between texts
- `LangDetectException`: If language detection fails (caught internally, falls back to default language)

##### Example

```python
from similarity import Similarity

sim = Similarity()

# Basic similarity
score = sim.similarity('Hello world', 'Hi world')
print(f"Score: {score:.4f}")  # Output: Score: 0.7071

# With method selection
sim_transformer = Similarity(use_transformers=True)
score_tfidf = sim_transformer.similarity(text1, text2, method='tfidf')
score_neural = sim_transformer.similarity(text1, text2, method='transformer')

# Cross-lingual (requires transformers)
sim = Similarity(use_transformers=True)
score = sim.similarity(
    'I love programming',
    'Eu amo programação'  # Portuguese
)
```

##### Performance

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| TF-IDF | ~50μs per pair | Good | Batch processing, lexical similarity |
| Transformer | ~5ms per pair | Excellent | Semantic understanding, cross-lingual |

---

#### detectlang()

Detect the language of a given text.

```python
detectlang(text)
```

##### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to analyze |

##### Returns

`str`: Detected language name in lowercase (e.g., `'english'`, `'portuguese'`, `'spanish'`)

##### Raises

- `LangDetectException`: If language detection fails (caught internally, returns default language)

##### Example

```python
from similarity import Similarity

sim = Similarity()

lang = sim.detectlang("Hello, how are you?")
print(lang)  # Output: 'english'

lang = sim.detectlang("Bonjour, comment allez-vous?")
print(lang)  # Output: 'french'

# Short text (may fail, returns default)
lang = sim.detectlang("Hi")
print(lang)  # Output: 'english' (default)
```

##### Notes

- Requires text with at least ~20 characters for reliable detection
- Uses the `langdetect` library internally
- Returns the default language (from constructor) if detection fails

---

## Classification Class

The `Classification` class provides text classification using word frequency, machine learning, or transformer-based methods.

### Classification Constructor

```python
Classification(
    language='english',
    use_ml=True,
    use_transformers=False,
    model_name='paraphrase-multilingual-MiniLM-L12-v2'
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | `str` | `'english'` | Language for text preprocessing (tokenization, stemming, stopwords). See [supported languages](#supported-languages). |
| `use_ml` | `bool` | `True` | Enable machine learning classifiers (LinearSVC or MultinomialNB). Provides good accuracy with reasonable performance. |
| `use_transformers` | `bool` | `False` | Enable transformer-based classification. Requires `sentence-transformers` package. Highest accuracy but slower. |
| `model_name` | `str` | `'paraphrase-multilingual-MiniLM-L12-v2'` | Sentence-transformer model name. Only relevant when `use_transformers=True`. |

#### Method Priority

When multiple methods are enabled:
1. **Transformers** (if `use_transformers=True`)
2. **Machine Learning** (if `use_ml=True`)
3. **Word Frequency** (fallback method)

#### Returns

Returns a `Classification` instance.

#### Raises

- `ImportError`: If `use_transformers=True` but `sentence-transformers` is not installed. Falls back to ML or word frequency method.

#### Example

```python
from similarity import Classification

# Word frequency method (simple)
clf = Classification(use_ml=False)

# Machine learning method (recommended)
clf = Classification(use_ml=True)

# Transformer method (highest accuracy)
clf = Classification(use_transformers=True)

# Custom configuration
clf = Classification(
    language='spanish',
    use_ml=True,
    model_name='distiluse-base-multilingual-cased-v2'
)
```

---

### Classification Methods

#### learning()

Train the classifier with labeled training data.

```python
learning(training_data)
```

##### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_data` | `list[dict]` | List of training examples. Each dict must contain `'class'` and `'word'` keys. |

##### Training Data Format

```python
training_data = [
    {"class": "category_name", "word": "text content"},
    {"class": "category_name", "word": "text content"},
    # ... more examples
]
```

##### Requirements

- Minimum: 2 examples per class (more is better)
- Recommended: 10+ examples per class for ML methods
- Keys: `'class'` (category label, string) and `'word'` (text content, string)

##### Returns

`None`: Trains the model in-place.

##### Example

```python
from similarity import Classification

training_data = [
    {"class": "spam", "word": "Buy now! Limited offer! Click here!"},
    {"class": "spam", "word": "You won $1000000! Claim your prize!"},
    {"class": "ham", "word": "Meeting scheduled for tomorrow at 3pm"},
    {"class": "ham", "word": "Thanks for your email, I'll review the document"},
]

clf = Classification(use_ml=True)
clf.learning(training_data)
```

##### Notes

- For ML method: Uses TF-IDF vectorization and fits LinearSVC/MultinomialNB
- For transformer method: Computes and stores embeddings
- For word frequency: Builds word frequency dictionary per class
- Can be called multiple times (retrains from scratch)

---

#### calculate_score()

Classify a text and optionally return confidence score.

```python
calculate_score(sentence, return_confidence=False)
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentence` | `str` | Required | Text to classify |
| `return_confidence` | `bool` | `False` | If `True`, returns tuple `(class, confidence)`. If `False`, returns only class name. |

##### Returns

- If `return_confidence=False`: `str` (predicted class name)
- If `return_confidence=True`: `tuple[str, float]` (predicted class, confidence score)

##### Confidence Score Interpretation

| Method | Confidence Meaning | Range |
|--------|-------------------|-------|
| Word Frequency | Sum of word frequencies | 0 to unlimited (higher = more confident) |
| ML (SVM) | Decision function value | -∞ to +∞ (higher = more confident) |
| ML (Naive Bayes) | Probability | 0.0 to 1.0 |
| Transformer | Cosine similarity | 0.0 to 1.0 |

##### Example

```python
from similarity import Classification

# Setup
clf = Classification(use_ml=True)
clf.learning(training_data)

# Basic prediction
predicted_class = clf.calculate_score("This is a test message")
print(f"Predicted: {predicted_class}")

# With confidence
predicted_class, confidence = clf.calculate_score(
    "This is a test message",
    return_confidence=True
)
print(f"Predicted: {predicted_class} (confidence: {confidence:.2f})")
```

---

#### predict()

Predict class label (sklearn-compatible interface).

```python
predict(sentence)
```

##### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sentence` | `str` | Text to classify |

##### Returns

`str`: Predicted class name

##### Example

```python
from similarity import Classification

clf = Classification(use_ml=True)
clf.learning(training_data)

# sklearn-style prediction
predicted = clf.predict("Sample text to classify")
print(f"Class: {predicted}")
```

##### Notes

- Equivalent to `calculate_score(sentence, return_confidence=False)`
- Provided for sklearn API compatibility

---

#### predict_proba()

Get class probabilities or confidence scores.

```python
predict_proba(sentence)
```

##### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sentence` | `str` | Text to classify |

##### Returns

`dict[str, float]`: Dictionary mapping class names to confidence/probability scores

##### Example

```python
from similarity import Classification

clf = Classification(use_ml=True)
clf.learning(training_data)

# Get probabilities for all classes
probabilities = clf.predict_proba("Sample text to classify")
print(f"Probabilities: {probabilities}")
# Output: {'class1': 0.75, 'class2': 0.20, 'class3': 0.05}

# Get most likely class
best_class = max(probabilities, key=probabilities.get)
print(f"Most likely: {best_class}")
```

##### Notes

- For ML with `predict_proba` support: Returns true probabilities (sum to 1.0)
- For other methods: Returns confidence scores (may not sum to 1.0)
- Useful for multi-class probability distributions

---

#### calculate_class_score()

Calculate score for a specific class (word frequency method only).

```python
calculate_class_score(sentence, class_name)
```

##### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sentence` | `str` | Text to score |
| `class_name` | `str` | Target class name |

##### Returns

`int` or `float`: Score for the specified class

##### Example

```python
from similarity import Classification

clf = Classification(use_ml=False)
clf.learning(training_data)

# Check score for specific class
score_spam = clf.calculate_class_score("Buy now!", "spam")
score_ham = clf.calculate_class_score("Buy now!", "ham")

print(f"Spam score: {score_spam}")
print(f"Ham score: {score_ham}")
```

##### Notes

- Only works with word frequency method (`use_ml=False`, `use_transformers=False`)
- Returns 0 if class doesn't exist
- Useful for understanding classification decisions

---

## Data Structures

### Training Data

Training data for classification is a list of dictionaries:

```python
training_data = [
    {
        "class": str,  # Category label
        "word": str    # Text content
    },
    # ... more examples
]
```

#### Example

```python
training_data = [
    {"class": "positive", "word": "I love this product!"},
    {"class": "positive", "word": "Excellent quality and service."},
    {"class": "negative", "word": "Terrible experience, very disappointed."},
    {"class": "negative", "word": "Poor quality, waste of money."},
]
```

---

## Exceptions

### LangDetectException

Raised by `langdetect` library when language detection fails. Caught internally by the library.

**When it occurs**: Short texts (<20 characters), texts without linguistic features, mixed languages

**Handling**: The library catches this exception and falls back to the default language.

### ImportError

Raised when optional dependencies are missing.

**When it occurs**:
- `use_transformers=True` but `sentence-transformers` not installed
- `use_transformers=True` but `torch` not installed

**Handling**: The library catches this exception, prints a warning, and falls back to TF-IDF or ML methods.

---

## Constants

### LANGUAGES

Tuple of language codes and names available in the library:

```python
LANGUAGES = [
    ('aa', 'Afar'),
    ('ab', 'Abkhazian'),
    ('en', 'English'),
    ('pt', 'Portuguese'),
    ('es', 'Spanish'),
    # ... 200+ language codes
]
```

Access in code:

```python
from similarity import LANGUAGES

# Get all language codes
codes = [code for code, name in LANGUAGES]

# Get all language names
names = [name for code, name in LANGUAGES]

# Create language dictionary
lang_dict = dict(LANGUAGES)
print(lang_dict['en'])  # Output: 'English'
```

---

## Supported Languages

### Languages with Advanced Stemming

These languages have dedicated stemming support:

- Arabic (`arabic`)
- Danish (`danish`)
- Dutch (`dutch`)
- English (`english`)
- Finnish (`finnish`)
- French (`french`)
- German (`german`)
- Hungarian (`hungarian`)
- Italian (`italian`)
- Norwegian (`norwegian`)
- Portuguese (`portuguese`)
- Romanian (`romanian`)
- Russian (`russian`)
- Spanish (`spanish`)
- Swedish (`swedish`)
- Turkish (`turkish`)

### Other Supported Languages

All other languages in the `LANGUAGES` constant are supported with basic processing (lowercasing, tokenization, stopwords if available).

---

## Available Models

### Multilingual Models

| Model Name | Size | Languages | Speed | Best For |
|------------|------|-----------|-------|----------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 418MB | 50+ | Fast | General purpose (default) |
| `paraphrase-multilingual-mpnet-base-v2` | 970MB | 50+ | Medium | High accuracy |
| `distiluse-base-multilingual-cased-v2` | 480MB | 50+ | Fast | Semantic search |

### English Models

| Model Name | Size | Speed | Best For |
|------------|------|-------|----------|
| `all-MiniLM-L6-v2` | 80MB | Very Fast | Speed-critical applications |
| `all-mpnet-base-v2` | 420MB | Medium | Highest accuracy |
| `paraphrase-mpnet-base-v2` | 420MB | Medium | Paraphrase detection |
| `all-MiniLM-L12-v2` | 120MB | Fast | Balanced performance |

Browse all models: [https://www.sbert.net/docs/pretrained_models.html](https://www.sbert.net/docs/pretrained_models.html)

---

## Usage Patterns

### Pattern 1: Quick Similarity Check

```python
from similarity import Similarity

sim = Similarity()
score = sim.similarity(text1, text2)

if score > 0.8:
    print("Texts are very similar")
elif score > 0.5:
    print("Texts are somewhat similar")
else:
    print("Texts are different")
```

### Pattern 2: Batch Processing

```python
from similarity import Similarity

sim = Similarity()

pairs = [
    ("text1", "text2"),
    ("text3", "text4"),
    # ... many pairs
]

scores = [sim.similarity(t1, t2) for t1, t2 in pairs]
```

### Pattern 3: Classification Pipeline

```python
from similarity import Classification

# Train
clf = Classification(use_ml=True)
clf.learning(training_data)

# Predict with confidence threshold
def classify_with_threshold(text, threshold=0.7):
    predicted, confidence = clf.calculate_score(text, return_confidence=True)
    if confidence >= threshold:
        return predicted
    else:
        return "uncertain"

result = classify_with_threshold("Sample text")
```

### Pattern 4: Multi-Model Ensemble

```python
from similarity import Similarity

# Use multiple methods and average
sim_classic = Similarity()
sim_neural = Similarity(use_transformers=True)

def ensemble_similarity(text1, text2):
    score1 = sim_classic.similarity(text1, text2)
    score2 = sim_neural.similarity(text1, text2)
    return (score1 + score2) / 2

score = ensemble_similarity(text1, text2)
```

---

## Performance Tips

1. **Reuse instances**: Create `Similarity` and `Classification` objects once and reuse them
2. **Batch similar operations**: Process similar texts together for better caching
3. **Choose appropriate method**: Use TF-IDF for speed, transformers for accuracy
4. **Limit transformer usage**: Use transformers only when semantic understanding is critical
5. **Enable quiet mode**: Set `quiet=True` in production to reduce I/O overhead
6. **Pre-download NLTK data**: Set `update=False` after first run

---

## Thread Safety

**Not thread-safe**: The library is not designed for concurrent access. Create separate instances per thread:

```python
from concurrent.futures import ThreadPoolExecutor
from similarity import Similarity

def process_pair(pair):
    sim = Similarity()  # Create instance per thread
    return sim.similarity(pair[0], pair[1])

with ThreadPoolExecutor(max_workers=4) as executor:
    scores = list(executor.map(process_pair, pairs))
```

---

## Version Compatibility

- **Python**: 3.7+
- **NLTK**: 3.0+
- **scikit-learn**: 0.24+
- **sentence-transformers**: 2.0+ (optional)
- **torch**: 1.9+ (optional, for transformers)

---

For more examples and guides, see the [README.md](README.md) and [example/](example/) directory.
