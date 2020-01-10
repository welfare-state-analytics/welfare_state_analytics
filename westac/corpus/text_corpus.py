import nltk.tokenize
import string
from tqdm import tqdm

ALPHABETIC_LOWER_CHARS = string.ascii_lowercase + "åäöéàáâãäåæèéêëîïñôöùûÿ"

class CorpusTextStream():

    def __init__(self, reader, use_tqdm=True):
        self.reader = reader
        self.use_tqdm = use_tqdm

    def get_metadata(self):

        return self.reader.metadata

    def texts(self):

        for meta, content in self.reader.get_iterator():
            yield meta, content

    def documents(self):

        docs = tqdm(self.texts()) if self.use_tqdm else self.texts()
        for meta, content in docs:
            yield meta, content

class CorpusTokenStream(CorpusTextStream):

    def __init__(self, reader, tokenizer=None, isalnum=True, use_tqdm=True):
        super().__init__(reader, use_tqdm=use_tqdm)
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

        super().__init__(
            reader,
            tokenizer=kwargs.get('tokenizer', None),
            isalnum=kwargs.get('isalnum', True),
            use_tqdm=kwargs.get('use_tqdm', True)
        )

        self.only_alphabetic = kwargs.get('only_alphabetic', True)
        self.to_lower = kwargs.get('to_lower', False)
        self.deacc = kwargs.get('deacc', False)
        self.min_len = kwargs.get('min_len', 2)
        self.max_len = kwargs.get('max_len', None)
        self.numerals = kwargs.get('numerals', True)
        self.stopwords = kwargs.get('stopwords', None)
        self.symbols = kwargs.get('symbols', True)
        self.alphabetic_chars = set(ALPHABETIC_LOWER_CHARS + ALPHABETIC_LOWER_CHARS.upper())
        self.symbols_chars = set("'\\¢£¥§©®°±øæç•›€™")\
            .union(set('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
        self.symbols_translation = dict.fromkeys(map(ord, self.symbols_chars), None)

    def documents(self):

        for meta, tokens in super().documents():

            if self.to_lower:
                tokens = (x.lower() for x in tokens)

            if self.min_len > 1:
                tokens = (x for x in tokens if len(x) >= self.min_len)

            if self.max_len is not None:
                tokens = (x for x in tokens if len(x) <= self.max_len)

            if self.symbols is False:
                # tokens = (x for x in tokens if not all([ c in string.punctuation for c in x ]))
                tokens = (x.translate(self.symbols_translation) for x in tokens)
                tokens = (x for x in tokens if len(x) >= self.min_len)

            if self.only_alphabetic:
                tokens = (x for x in tokens if any(c in x for c in self.alphabetic_chars))

            if self.numerals is False:
                tokens = (x for x in tokens if not x.isnumeric())

            if self.stopwords is not None:
                tokens = (x for x in tokens if not x in self.stopwords)

            tokens = list(tokens)
            filename = meta if isinstance(meta, str) else meta.filename
            self.n_tokens[filename] = len(tokens)

            yield meta, tokens
