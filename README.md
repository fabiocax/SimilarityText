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
from similarity import Similarity
a=Similarity()
print(a.similarity('La Casa de Papel','La Casa de papel'))
```

