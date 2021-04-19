from similarity import Similarity

a=Similarity(update=False)


print(a.similarity('A caza de papil é linda','A Casa de papel é linda '))



print(a.similarity('A caza de papel é linda','A Casa de papel é linda '))


print(a.similarity('A casa de papel eh linda','A Casa de papel é linda '))
