# SimilarityText

Similarity uses the most advanced AI resources to identify semantic equality between two texts.

## Install

pip install SimilarityText

## Parametres
```
Similarity(
      update=True,
      language='english',
      langdetect=False,
      nltk_downloads=[],
      quiet=True
  )
```

## Example

```
from similarity import Similarity,Classification
a=Similarity()
print(a.similarity('La Casa de Papel','La Casa de papel'))


training_data = []
training_data.append({"class":"amor", "word":"Eu te amo"})
training_data.append({"class":"amor", "word":"Você é o amor da minha vida"})
training_data.append({"class":"medo", "word":"estou com medo"})
training_data.append({"class":"medo", "word":"tenho medo de fantasma"})

a = Classification()
a.learning(training_data)
a.calculate_score("Nossas dúvidas são traidoras e nos fazem perder o que, com frequência, poderíamos ganhar, por simples de arriscar. .")

```
