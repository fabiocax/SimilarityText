"Author: Fabio Alberti"
import nltk
import warnings
from nltk.corpus import stopwords
warnings.filterwarnings("ignore")
import random
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import RSLPStemmer, SnowballStemmer
from langdetect import detect, LangDetectException

LANGUAGES = [
    ('aa', 'Afar'),
    ('ab', 'Abkhazian'),
    ('af', 'Afrikaans'),
    ('ak', 'Akan'),
    ('sq', 'Albanian'),
    ('am', 'Amharic'),
    ('ar', 'Arabic'),
    ('an', 'Aragonese'),
    ('hy', 'Armenian'),
    ('as', 'Assamese'),
    ('av', 'Avaric'),
    ('ae', 'Avestan'),
    ('ay', 'Aymara'),
    ('az', 'Azerbaijani'),
    ('ba', 'Bashkir'),
    ('bm', 'Bambara'),
    ('eu', 'Basque'),
    ('be', 'Belarusian'),
    ('bn', 'Bengali'),
    ('bh', 'Bihari languages'),
    ('bi', 'Bislama'),
    ('bo', 'Tibetan'),
    ('bs', 'Bosnian'),
    ('br', 'Breton'),
    ('bg', 'Bulgarian'),
    ('my', 'Burmese'),
    ('ca', 'Catalan; Valencian'),
    ('cs', 'Czech'),
    ('ch', 'Chamorro'),
    ('ce', 'Chechen'),
    ('zh', 'Chinese'),
    ('cu', 'Church Slavic; Old Slavonic; Church Slavonic; Old Bulgarian; Old Church Slavonic'),
    ('cv', 'Chuvash'),
    ('kw', 'Cornish'),
    ('co', 'Corsican'),
    ('cr', 'Cree'),
    ('cy', 'Welsh'),
    ('cs', 'Czech'),
    ('da', 'Danish'),
    ('de', 'German'),
    ('dv', 'Divehi; Dhivehi; Maldivian'),
    ('nl', 'Dutch; Flemish'),
    ('dz', 'Dzongkha'),
    ('el', 'Greek, Modern (1453-)'),
    ('en', 'English'),
    ('eo', 'Esperanto'),
    ('et', 'Estonian'),
    ('eu', 'Basque'),
    ('ee', 'Ewe'),
    ('fo', 'Faroese'),
    ('fa', 'Persian'),
    ('fj', 'Fijian'),
    ('fi', 'Finnish'),
    ('fr', 'French'),
    ('fy', 'Western Frisian'),
    ('ff', 'Fulah'),
    ('Ga', 'Georgian'),
    ('de', 'German'),
    ('gd', 'Gaelic; Scottish Gaelic'),
    ('ga', 'Irish'),
    ('gl', 'Galician'),
    ('gv', 'Manx'),
    ('el', 'Greek, Modern (1453-)'),
    ('gn', 'Guarani'),
    ('gu', 'Gujarati'),
    ('ht', 'Haitian; Haitian Creole'),
    ('ha', 'Hausa'),
    ('he', 'Hebrew'),
    ('hz', 'Herero'),
    ('hi', 'Hindi'),
    ('ho', 'Hiri Motu'),
    ('hr', 'Croatian'),
    ('hu', 'Hungarian'),
    ('hy', 'Armenian'),
    ('ig', 'Igbo'),
    ('is', 'Icelandic'),
    ('io', 'Ido'),
    ('ii', 'Sichuan Yi; Nuosu'),
    ('iu', 'Inuktitut'),
    ('ie', 'Interlingue; Occidental'),
    ('ia', 'Interlingua (International Auxiliary Language Association)'),
    ('id', 'Indonesian'),
    ('ik', 'Inupiaq'),
    ('is', 'Icelandic'),
    ('it', 'Italian'),
    ('jv', 'Javanese'),
    ('ja', 'Japanese'),
    ('kl', 'Kalaallisut; Greenlandic'),
    ('kn', 'Kannada'),
    ('ks', 'Kashmiri'),
    ('ka', 'Georgian'),
    ('kr', 'Kanuri'),
    ('kk', 'Kazakh'),
    ('km', 'Central Khmer'),
    ('ki', 'Kikuyu; Gikuyu'),
    ('rw', 'Kinyarwanda'),
    ('ky', 'Kirghiz; Kyrgyz'),
    ('kv', 'Komi'),
    ('kg', 'Kongo'),
    ('ko', 'Korean'),
    ('kj', 'Kuanyama; Kwanyama'),
    ('ku', 'Kurdish'),
    ('lo', 'Lao'),
    ('la', 'Latin'),
    ('lv', 'Latvian'),
    ('li', 'Limburgan; Limburger; Limburgish'),
    ('ln', 'Lingala'),
    ('lt', 'Lithuanian'),
    ('lb', 'Luxembourgish; Letzeburgesch'),
    ('lu', 'Luba-Katanga'),
    ('lg', 'Ganda'),
    ('mk', 'Macedonian'),
    ('mh', 'Marshallese'),
    ('ml', 'Malayalam'),
    ('mi', 'Maori'),
    ('mr', 'Marathi'),
    ('ms', 'Malay'),
    ('Mi', 'Micmac'),
    ('mk', 'Macedonian'),
    ('mg', 'Malagasy'),
    ('mt', 'Maltese'),
    ('mn', 'Mongolian'),
    ('mi', 'Maori'),
    ('ms', 'Malay'),
    ('my', 'Burmese'),
    ('na', 'Nauru'),
    ('nv', 'Navajo; Navaho'),
    ('nr', 'Ndebele, South; South Ndebele'),
    ('nd', 'Ndebele, North; North Ndebele'),
    ('ng', 'Ndonga'),
    ('ne', 'Nepali'),
    ('nl', 'Dutch; Flemish'),
    ('nn', 'Norwegian Nynorsk; Nynorsk, Norwegian'),
    ('nb', 'Bokmål, Norwegian; Norwegian Bokmål'),
    ('no', 'Norwegian'),
    ('oc', 'Occitan (post 1500)'),
    ('oj', 'Ojibwa'),
    ('or', 'Oriya'),
    ('om', 'Oromo'),
    ('os', 'Ossetian; Ossetic'),
    ('pa', 'Panjabi; Punjabi'),
    ('fa', 'Persian'),
    ('pi', 'Pali'),
    ('pl', 'Polish'),
    ('pt', 'Portuguese'),
    ('ps', 'Pushto; Pashto'),
    ('qu', 'Quechua'),
    ('rm', 'Romansh'),
    ('ro', 'Romanian; Moldavian; Moldovan'),
    ('ro', 'Romanian; Moldavian; Moldovan'),
    ('rn', 'Rundi'),
    ('ru', 'Russian'),
    ('sg', 'Sango'),
    ('sa', 'Sanskrit'),
    ('si', 'Sinhala; Sinhalese'),
    ('sk', 'Slovak'),
    ('sk', 'Slovak'),
    ('sl', 'Slovenian'),
    ('se', 'Northern Sami'),
    ('sm', 'Samoan'),
    ('sn', 'Shona'),
    ('sd', 'Sindhi'),
    ('so', 'Somali'),
    ('st', 'Sotho, Southern'),
    ('es', 'Spanish; Castilian'),
    ('sq', 'Albanian'),
    ('sc', 'Sardinian'),
    ('sr', 'Serbian'),
    ('ss', 'Swati'),
    ('su', 'Sundanese'),
    ('sw', 'Swahili'),
    ('sv', 'Swedish'),
    ('ty', 'Tahitian'),
    ('ta', 'Tamil'),
    ('tt', 'Tatar'),
    ('te', 'Telugu'),
    ('tg', 'Tajik'),
    ('tl', 'Tagalog'),
    ('th', 'Thai'),
    ('bo', 'Tibetan'),
    ('ti', 'Tigrinya'),
    ('to', 'Tonga (Tonga Islands)'),
    ('tn', 'Tswana'),
    ('ts', 'Tsonga'),
    ('tk', 'Turkmen'),
    ('tr', 'Turkish'),
    ('tw', 'Twi'),
    ('ug', 'Uighur; Uyghur'),
    ('uk', 'Ukrainian'),
    ('ur', 'Urdu'),
    ('uz', 'Uzbek'),
    ('ve', 'Venda'),
    ('vi', 'Vietnamese'),
    ('vo', 'Volapük'),
    ('cy', 'Welsh'),
    ('wa', 'Walloon'),
    ('wo', 'Wolof'),
    ('xh', 'Xhosa'),
    ('yi', 'Yiddish'),
    ('yo', 'Yoruba'),
    ('za', 'Zhuang; Chuang'),
    ('zh', 'Chinese'),
    ('zu', 'Zulu')
]

class Similarity:

    def __init__(self,update=True,language='english',langdetect=False,nltk_downloads=[],quiet=True,use_transformers=False,model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.__language=language
        self.__langdetect=langdetect
        self.__quiet=quiet
        self.__use_transformers=use_transformers
        self.__model_name=model_name
        self.__transformer_model=None
        self.__remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

        # Map languages to Snowball stemmer supported languages
        self.__snowball_languages = {
            'arabic', 'danish', 'dutch', 'english', 'finnish', 'french', 'german',
            'hungarian', 'italian', 'norwegian', 'portuguese', 'romanian', 'russian',
            'spanish', 'swedish', 'turkish'
        }

        parametes=['stopwords','rslp','punkt','punkt_tab']+nltk_downloads
        if update == True:
            for item in parametes:
                nltk.download(item, quiet=quiet)

        # Initialize transformer model if requested
        if self.__use_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self.__transformer_model = SentenceTransformer(self.__model_name)
                if not self.__quiet:
                    print(f'Loaded transformer model: {self.__model_name}')
            except ImportError:
                if not self.__quiet:
                    print('Warning: sentence-transformers not installed. Install with: pip install sentence-transformers')
                    print('Falling back to TF-IDF method')
                self.__use_transformers = False
    def detectlang(self,text):
        try:
            lang=detect(text)
            dict_lang=dict(LANGUAGES)
            detect_lang=dict_lang.get(lang, 'english')
            return detect_lang.lower()
        except LangDetectException:
            if not self.__quiet:
                print(f'Warning: Could not detect language for text: "{text[:50]}...". Using default: {self.__language}')
            return self.__language

    def __get_stemmer(self, language):
        """Get appropriate stemmer for the language"""
        if language == 'portuguese':
            return RSLPStemmer()
        elif language in self.__snowball_languages:
            return SnowballStemmer(language)
        else:
            # Return a dummy stemmer that just lowercases
            class DummyStemmer:
                def stem(self, token):
                    return token.lower()
            return DummyStemmer()

    def __LemTokens(self,tokens, language=None):
        if language is None:
            language = self.__language
        lemmer = self.__get_stemmer(language)
        return [lemmer.stem(token) for token in tokens]

    def __LemNormalize(self,text, language=None):
        if language is None:
            language = self.__language
        return self.__LemTokens(nltk.word_tokenize(text.lower().translate(self.__remove_punct_dict)), language)

    def similarity(self,text_a,text_b, method='auto'):
        """
        Calculate similarity between two texts.

        Args:
            text_a: First text
            text_b: Second text
            method: 'auto' (choose best), 'transformer' (neural), 'tfidf' (classic)

        Returns:
            Float between 0.0 and 1.0 (1.0 = identical)
        """
        # Use transformers if available and requested
        if (method == 'transformer' or (method == 'auto' and self.__use_transformers)) and self.__transformer_model is not None:
            return self.__similarity_transformer(text_a, text_b)

        # Fall back to TF-IDF method
        return self.__similarity_tfidf(text_a, text_b)

    def __similarity_transformer(self, text_a, text_b):
        """Calculate similarity using sentence transformers"""
        embeddings = self.__transformer_model.encode([text_a, text_b])
        similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity_score)

    def __similarity_tfidf(self, text_a, text_b):
        """Calculate similarity using TF-IDF (classic method)"""
        lang=self.__language
        if self.__langdetect == True:
            a=self.detectlang(text_a)
            b=self.detectlang(text_b)
            if a == b:
                lang=a
                if self.__quiet==False:
                    print(f'Language detected: {lang}')
            else:
                if not self.__quiet:
                    print(f'Warning: Languages differ ({a} vs {b}). Using {self.__language}')
                lang=self.__language

        # Check if stopwords are available for this language
        try:
            stop_words = stopwords.words(lang)
        except OSError:
            if not self.__quiet:
                print(f'Warning: Stopwords not available for {lang}. Proceeding without stopwords.')
            stop_words = None

        sent_tokens = nltk.sent_tokenize(text_b, language=lang)
        sent_tokens.append(text_a.lower())

        TfidfVec = TfidfVectorizer(
            tokenizer=lambda x: self.__LemNormalize(x, lang),
            stop_words=stop_words
        )
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)

        # Get the second highest similarity (highest would be self-similarity)
        scores = vals[0]
        scores_sorted = np.sort(scores)
        req_tfidf = scores_sorted[-2] if len(scores_sorted) > 1 else scores_sorted[-1]

        return float(req_tfidf)

class Classification:
    def __init__(self, language='english', use_ml=True, use_transformers=False, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Text classification with multiple methods.

        Args:
            language: Language for text processing
            use_ml: Use scikit-learn ML classifiers (SVM, Naive Bayes)
            use_transformers: Use transformer-based classification (most accurate)
            model_name: Transformer model name if use_transformers=True
        """
        self.dados = []
        self.language = language
        self.use_ml = use_ml
        self.use_transformers = use_transformers
        self.__transformer_model = None
        self.__vectorizer = None
        self.__classifier = None
        self.__classes = []
        self.__training_texts = []
        self.__training_labels = []

        # Snowball stemmer languages
        self.__snowball_languages = {
            'arabic', 'danish', 'dutch', 'english', 'finnish', 'french', 'german',
            'hungarian', 'italian', 'norwegian', 'portuguese', 'romanian', 'russian',
            'spanish', 'swedish', 'turkish'
        }

        if self.use_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self.__transformer_model = SentenceTransformer(model_name)
            except ImportError:
                print('Warning: sentence-transformers not installed. Falling back to ML method.')
                self.use_transformers = False
                self.use_ml = True

    def __get_stemmer(self):
        """Get appropriate stemmer for the language"""
        if self.language == 'portuguese':
            return RSLPStemmer()
        elif self.language in self.__snowball_languages:
            return SnowballStemmer(self.language)
        else:
            class DummyStemmer:
                def stem(self, token):
                    return token.lower()
            return DummyStemmer()

    def __Tokenize(self,sentence):
        sentence = sentence.lower()
        sentence = nltk.word_tokenize(sentence)
        return sentence

    def __Stemming(self,sentence):
        stemmer = self.__get_stemmer()
        phrase = []
        for word in sentence:
            phrase.append(stemmer.stem(word.lower()))
        return phrase

    def __RemoveStopWords(self,sentence):
        try:
            stop_words = nltk.corpus.stopwords.words(self.language)
        except OSError:
            stop_words = []
        phrase = []
        for word in sentence:
            if word not in stop_words:
                phrase.append(word)
        return phrase

    def learning(self, training_data):
        """
        Train the classifier with training data.

        Args:
            training_data: List of dicts with 'class' and 'word' keys
        """
        # Store training data
        self.__training_texts = [data['word'] for data in training_data]
        self.__training_labels = [data['class'] for data in training_data]
        self.__classes = list(set(self.__training_labels))

        if self.use_transformers and self.__transformer_model:
            # For transformers, we just store the data and compute embeddings on demand
            self.__training_embeddings = self.__transformer_model.encode(self.__training_texts)
        elif self.use_ml:
            # Use TF-IDF + ML classifier
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.svm import LinearSVC
            from sklearn.pipeline import Pipeline

            # Preprocessing function
            def preprocess(text):
                tokens = self.__Tokenize(text)
                tokens = self.__Stemming(tokens)
                tokens = self.__RemoveStopWords(tokens)
                return ' '.join(tokens)

            # Create pipeline
            self.__vectorizer = TfidfVectorizer(preprocessor=preprocess, max_features=1000)

            # Use LinearSVC for better performance
            try:
                from sklearn.svm import LinearSVC
                self.__classifier = LinearSVC(random_state=42, max_iter=2000)
            except:
                self.__classifier = MultinomialNB()

            # Train
            X = self.__vectorizer.fit_transform(self.__training_texts)
            self.__classifier.fit(X, self.__training_labels)
        else:
            # Use original word frequency method
            corpus_words = {}
            for data in training_data:
                frase = data['word']
                frase = self.__Tokenize(frase)
                frase = self.__Stemming(frase)
                frase = self.__RemoveStopWords(frase)
                class_name = data['class']
                if class_name not in corpus_words:
                    corpus_words[class_name] = {}
                for word in frase:
                    if word not in corpus_words[class_name]:
                        corpus_words[class_name][word] = 1
                    else:
                        corpus_words[class_name][word] += 1
            self.dados = corpus_words

    def calculate_class_score(self,sentence,class_name):
        """Calculate score for a specific class (word frequency method)"""
        score = 0
        sentence = self.__Tokenize(sentence)
        sentence = self.__Stemming(sentence)
        for word in sentence:
            if word in self.dados.get(class_name, {}):
                score += self.dados[class_name][word]
        return score

    def calculate_score(self, sentence, return_confidence=False):
        """
        Classify a sentence.

        Args:
            sentence: Text to classify
            return_confidence: If True, return (class, confidence_score)

        Returns:
            class_name or (class_name, confidence_score)
        """
        if self.use_transformers and self.__transformer_model and hasattr(self, '_Classification__training_embeddings'):
            # Use transformer embeddings
            sentence_embedding = self.__transformer_model.encode([sentence])[0]
            similarities = cosine_similarity([sentence_embedding], self.__training_embeddings)[0]

            # Find the most similar training example
            best_idx = np.argmax(similarities)
            predicted_class = self.__training_labels[best_idx]
            confidence = float(similarities[best_idx])

            if return_confidence:
                return predicted_class, confidence
            return predicted_class

        elif self.use_ml and self.__classifier and self.__vectorizer:
            # Use ML classifier
            X = self.__vectorizer.transform([sentence])
            predicted_class = self.__classifier.predict(X)[0]

            # Get confidence if classifier supports it
            if hasattr(self.__classifier, 'decision_function'):
                confidence = float(np.max(self.__classifier.decision_function(X)))
            elif hasattr(self.__classifier, 'predict_proba'):
                confidence = float(np.max(self.__classifier.predict_proba(X)))
            else:
                confidence = 1.0

            if return_confidence:
                return predicted_class, confidence
            return predicted_class

        else:
            # Use word frequency method
            high_score = 0
            classname = 'default'
            for classe in self.dados.keys():
                pontos = self.calculate_class_score(sentence, classe)
                if pontos > high_score:
                    high_score = pontos
                    classname = classe

            if return_confidence:
                return classname, high_score
            return classname

    def predict(self, sentence):
        """Alias for calculate_score for sklearn-like interface"""
        result = self.calculate_score(sentence, return_confidence=False)
        if isinstance(result, tuple):
            return result[0]
        return result

    def predict_proba(self, sentence):
        """Get class probabilities (if using ML method)"""
        if self.use_ml and self.__classifier and self.__vectorizer and hasattr(self.__classifier, 'predict_proba'):
            X = self.__vectorizer.transform([sentence])
            proba = self.__classifier.predict_proba(X)[0]
            return dict(zip(self.__classes, proba))
        else:
            # For other methods, return confidence scores
            result = self.calculate_score(sentence, return_confidence=True)
            if isinstance(result, tuple):
                class_name, score = result
                return {class_name: score}
    