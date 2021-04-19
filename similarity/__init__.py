import nltk
import warnings
from nltk.corpus import stopwords
warnings.filterwarnings("ignore")
import random
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class Similarity:
            
    def __init__(self,update=True,language='portuguese'):
        self.language=language
        self.parametes=['stopwords','rslp','punkt']
        self.remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        if update == True:
            self.nltk_download()

    def nltk_download(self):
        for item in self.parametes:
            nltk.download(item)

    def LemTokens(self,tokens):
        lemmer = nltk.stem.RSLPStemmer()
        return [lemmer.stem(token) for token in tokens]
        
    def LemNormalize(self,text):
        return self.LemTokens(nltk.word_tokenize(text.lower().translate(self.remove_punct_dict)))

    def similarity(self,text_a,text_b):        
        sent_tokens = nltk.sent_tokenize(text_b, language=self.language)
        sent_tokens.append(text_a.lower())
        TfidfVec = TfidfVectorizer(tokenizer=self.LemNormalize, stop_words=stopwords.words(self.language))
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx=vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        return req_tfidf