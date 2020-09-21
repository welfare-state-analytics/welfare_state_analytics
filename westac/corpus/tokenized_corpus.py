from __future__ import annotations

from typing import Any

import pandas as pd

from .tokens_transformer import DEFAULT_PROCESS_OPTS, TokensTransformer


class TokenizedCorpus():

    def __init__(self, reader: Any, **kwargs):

        if not hasattr(reader, 'metadata'):
            raise TypeError(f"Corpus reader {type(reader)} has no `metadata` property")

        if not hasattr(reader, 'filenames'):
            raise TypeError(f"Corpus reader {type(reader)} has no `filenames` property")

        self.reader = reader
        self.documents = pd.DataFrame([d.__dict__ for d in reader.metadata])

        opts = DEFAULT_PROCESS_OPTS
        opts = { **opts, **{ k:v for k,v in kwargs.items() if k in opts }}

        self.transformer = TokensTransformer(**opts)
        self.iterator = None

    def tokens_stream(self):

        n_raw_tokens = []
        n_tokens = []
        for dokument_name, tokens in self.reader:

            n_raw_tokens.append(len(tokens))

            tokens = self.transformer.transform(tokens)

            n_tokens.append(len(tokens))

            yield dokument_name, tokens

        self.documents['n_raw_tokens'] = n_raw_tokens
        self.documents['n_tokens'] = n_tokens

    @property
    def metadata(self):
        return self.reader.metadata

    @property
    def filenames(self):
        return self.reader.filenames

    def __iter__(self):
        self.iterator = self.tokens_stream()
        return self

    def __next__(self):
        if self.iterator is None:
            raise StopIteration
        return next(self.iterator)
