
from similarity import Similarity



training_data = []
training_data.append({"class":"amor", "word":"Eu te amo"})
training_data.append({"class":"amor", "word":"Você é o amor da minha vida"})
training_data.append({"class":"amor", "word":"No amor, jamais nos deixamos completar"})
training_data.append({"class":"amor", "word":"assim te amo porque não sei amar de outra maneira"})
training_data.append({"class":"medo", "word":"estou com medo"})
training_data.append({"class":"medo", "word":"tenho medo de fantasma"})
training_data.append({"class":"perda", "word":"A perda de um inimigo não compensa a de um amigo."})
training_data.append({"class":"perda", "word":"Há campeões de tudo, inclusive de perda de campeonatos."})
training_data.append({"class":"esperança", "word":"Enquanto há vida, há esperança."})
training_data.append({"class":"esperança", "word":"Vá firme na direção das suas metas. Porque o pensamento cria, o desejo atrai e a fé realiza."})
a = Classification()
a.learning(training_data)
a.calculate_score("Nossas dúvidas são traidoras e nos fazem perder o que, com frequência, poderíamos ganhar, por simples de arriscar.")
#a.calculate_score("Te amo sem saber como, nem quando, nem onde, te amo diretamente sem problemas nem orgulho: assim te amo porque não sei amar de outra maneira. Porque eu te amo, tu não precisas de mim. Porque tu me amas, eu não preciso de ti. No amor, jamais nos deixamos completar")
#a.calculate_score("Os homens esquecem mais rapidamente a morte do pai do que a perda do patrimônio.")
#a.calculate_score('Perdi um campeonato ontem')
#a.calculate_score('Pra quem tem fé a vida nunca tem fim.')
