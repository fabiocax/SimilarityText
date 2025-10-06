"""
Advanced AI Examples with Transformers

This example demonstrates the new AI-powered features including:
- Transformer-based semantic similarity (more accurate than TF-IDF)
- Machine learning classification with SVM
- Multilingual support

To use transformers, install with:
    pip install sentence-transformers torch transformers

Or install the package with:
    pip install SimilarityText[transformers]
"""

from similarity import Similarity, Classification

print("=" * 60)
print("ADVANCED SIMILARITY EXAMPLES")
print("=" * 60)

# Example 1: Classic TF-IDF Method
print("\n1. Classic TF-IDF Method:")
sim_classic = Similarity(update=False, quiet=True)
score = sim_classic.similarity(
    'A casa de papel é uma série incrível',
    'A Casa de Papel é um show fantástico'
)
print(f"   Similarity (TF-IDF): {score:.4f}")

# Example 2: Transformer-based Similarity (requires sentence-transformers)
print("\n2. Transformer-based Similarity (Neural Networks):")
try:
    sim_ai = Similarity(update=False, quiet=True, use_transformers=True)
    score = sim_ai.similarity(
        'A casa de papel é uma série incrível',
        'A Casa de Papel é um show fantástico'
    )
    print(f"   Similarity (Transformers): {score:.4f}")
    print("   Note: Transformers typically give higher, more accurate scores!")
except Exception as e:
    print(f"   Transformers not available: {e}")
    print("   Install with: pip install sentence-transformers torch")

# Example 3: Multilingual Similarity
print("\n3. Multilingual Similarity (same meaning, different languages):")
try:
    sim_multilingual = Similarity(update=False, quiet=True, use_transformers=True)
    score = sim_multilingual.similarity(
        'I love artificial intelligence',
        'Eu amo inteligência artificial',
        method='transformer'
    )
    print(f"   English <-> Portuguese: {score:.4f}")
except Exception as e:
    print(f"   Transformers not available: {e}")

print("\n" + "=" * 60)
print("ADVANCED CLASSIFICATION EXAMPLES")
print("=" * 60)

# Prepare training data
training_data = []
training_data.append({"class": "amor", "word": "Eu te amo"})
training_data.append({"class": "amor", "word": "Você é o amor da minha vida"})
training_data.append({"class": "amor", "word": "No amor, jamais nos deixamos completar"})
training_data.append({"class": "medo", "word": "estou com medo"})
training_data.append({"class": "medo", "word": "tenho medo de fantasma"})
training_data.append({"class": "medo", "word": "isso é assustador"})
training_data.append({"class": "esperança", "word": "Enquanto há vida, há esperança"})
training_data.append({"class": "esperança", "word": "Vá firme na direção das suas metas"})
training_data.append({"class": "esperança", "word": "Acredite em dias melhores"})

# Example 4: Word Frequency Classification (original method)
print("\n4. Word Frequency Classification (Original):")
classifier_simple = Classification(language='portuguese', use_ml=False)
classifier_simple.learning(training_data)
result = classifier_simple.calculate_score("Te amo muito, você é especial")
print(f"   Text: 'Te amo muito, você é especial'")
print(f"   Predicted class: {result}")

# Example 5: Machine Learning Classification (SVM)
print("\n5. Machine Learning Classification (SVM + TF-IDF):")
classifier_ml = Classification(language='portuguese', use_ml=True, use_transformers=False)
classifier_ml.learning(training_data)
result = classifier_ml.calculate_score("Te amo muito, você é especial", return_confidence=True)
print(f"   Text: 'Te amo muito, você é especial'")
print(f"   Predicted class: {result[0]}")
print(f"   Confidence: {result[1]:.4f}")

# Example 6: Transformer-based Classification (most accurate)
print("\n6. Transformer-based Classification (Neural Networks):")
try:
    classifier_ai = Classification(language='portuguese', use_transformers=True)
    classifier_ai.learning(training_data)
    result = classifier_ai.calculate_score("Te amo muito, você é especial", return_confidence=True)
    print(f"   Text: 'Te amo muito, você é especial'")
    print(f"   Predicted class: {result[0]}")
    print(f"   Confidence: {result[1]:.4f}")
    print("   Note: Transformers understand semantic meaning better!")
except Exception as e:
    print(f"   Transformers not available: {e}")
    print("   Install with: pip install sentence-transformers torch")

# Example 7: Testing multiple sentences
print("\n7. Testing Multiple Sentences:")
test_sentences = [
    "Estou apaixonado por você",
    "Tenho medo do escuro",
    "Nunca desista dos seus sonhos",
]

classifier_ml = Classification(language='portuguese', use_ml=True)
classifier_ml.learning(training_data)

for sentence in test_sentences:
    predicted_class, confidence = classifier_ml.calculate_score(sentence, return_confidence=True)
    print(f"   '{sentence}'")
    print(f"   -> {predicted_class} (confidence: {confidence:.4f})")

print("\n" + "=" * 60)
print("COMPARISON: TF-IDF vs Transformers")
print("=" * 60)

# Compare different similarity methods
pairs = [
    ("O gato está dormindo", "O felino está descansando"),
    ("Vou ao mercado comprar frutas", "Preciso ir na feira comprar verduras"),
    ("Python é uma linguagem de programação", "Java é uma linguagem de programação"),
]

for text1, text2 in pairs:
    print(f"\nComparing:")
    print(f"  A: '{text1}'")
    print(f"  B: '{text2}'")

    # TF-IDF
    sim_tfidf = Similarity(update=False, quiet=True)
    score_tfidf = sim_tfidf.similarity(text1, text2)
    print(f"  TF-IDF score: {score_tfidf:.4f}")

    # Transformers
    try:
        sim_transformer = Similarity(update=False, quiet=True, use_transformers=True)
        score_transformer = sim_transformer.similarity(text1, text2)
        print(f"  Transformer score: {score_transformer:.4f}")
        print(f"  Difference: {abs(score_transformer - score_tfidf):.4f}")
    except:
        print("  Transformer score: N/A (not installed)")

print("\n" + "=" * 60)
print("Done! The new AI features provide more accurate results.")
print("=" * 60)
