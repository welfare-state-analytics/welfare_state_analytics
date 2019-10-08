import nltk.tokenize
import string

class CorpusTextStream():

    def __init__(self, reader):
        self.reader = reader

    def get_metadata(self):

        return self.reader.metadata

    def texts(self):

        for meta, content in self.reader.get_iterator():
            yield meta, content

    def documents(self):

        for meta, content in self.texts():
            yield meta, content

class CorpusTokenStream(CorpusTextStream):

    def __init__(self, reader, tokenizer=None, isalnum=True):
        super().__init__(reader)
        self.tokenizer = tokenizer or (lambda text: nltk.tokenize.word_tokenize(text, language='swedish'))
        self.n_raw_tokens =  { }
        self.n_tokens =  { }
        self.isalnum = isalnum

    def documents(self):

        for meta, content in super().documents():
            tokens = self.tokenizer(content)
            if self.isalnum:
                tokens = [ t for t in tokens if any(c.isalnum() for c in t) ]
            filename = meta if isinstance(meta, str) else meta.filename
            self.n_raw_tokens[filename] = len(tokens)
            self.n_tokens[filename] = len(tokens)
            yield meta, tokens

class ProcessedCorpus(CorpusTokenStream):

    def __init__(self, reader, **kwargs):

        super().__init__(reader, tokenizer=kwargs.get('tokenizer', None), isalnum=kwargs.get('isalnum', True))
        
        self.to_lower = kwargs.get('to_lower', False)
        self.deacc = kwargs.get('deacc', False)
        self.min_len = kwargs.get('min_len', 2)
        self.max_len = kwargs.get('max_len', None)
        self.numerals = kwargs.get('numerals', True)
        self.stopwords = kwargs.get('stopwords', None)
        self.symbols = kwargs.get('symbols', True)

    def documents(self):

        for meta, tokens in super().documents():

            if self.to_lower:
                tokens = (x.lower() for x in tokens)

            if self.min_len > 1:
                tokens = (x for x in tokens if len(x) >= self.min_len)

            if self.max_len is not None:
                tokens = (x for x in tokens if len(x) <= self.max_len)

            if self.numerals is False:
                tokens = (x for x in tokens if not x.isnumeric())

            if self.stopwords is not None:
                tokens = (x for x in tokens if not x in self.stopwords)

            if self.symbols is False:
                tokens = (x for x in tokens if not all([ c in string.punctuation for c in x ]))

            tokens = list(tokens)
            filename = meta if isinstance(meta, str) else meta.filename
            self.n_tokens[filename] = len(tokens)

            yield meta, tokens

