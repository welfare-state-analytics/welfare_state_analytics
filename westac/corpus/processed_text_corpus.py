from __future__ import annotations

from typing import Any, Callable

import nltk.tokenize

from .tokens_transformer import DEFAULT_PROCESS_OPTS, TokensTransformer


class BaseCorpus():

    def __init__(self, reader):
        self.reader = reader
        self.iterator = None

    def get_metadata(self):

        return self.reader.metadata

    def documents(self):

        for meta, content in self.reader.get_iterator():
            yield meta, content

    def __iter__(self):

        self.iterator = None
        return self

    def __next__(self):

        if self.iterator is None:
            self.iterator = self.documents()

        return next(self.iterator)

class TokenizedCorpus(BaseCorpus):

    def __init__(self, reader: Any, tokenizer: Callable=None, isalnum: bool=True):
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

class ProcessedTextCorpus(TokenizedCorpus):

    def __init__(self, reader, **kwargs):

        super().__init__(
            reader,
            tokenizer=kwargs.get('tokenizer', None),
            isalnum=kwargs.get('isalnum', True)
        )

        opts = DEFAULT_PROCESS_OPTS
        opts = { **opts, **{ k:v for k,v in kwargs.items() if k in opts }}

        self.transformer = TokensTransformer(**opts)

    def documents(self):

        for meta, tokens in super().documents():

            tokens = list(self.transformer.transform(tokens))

            filename = meta if isinstance(meta, str) else meta.filename
            self.n_tokens[filename] = len(tokens)

            yield meta, tokens
