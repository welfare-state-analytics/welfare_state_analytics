from __future__ import annotations

from typing import Any, List

import textacy.preprocessing.remove as textacy_remove

from westac.corpus import utility

# pylint: disable=too-many-arguments

DEFAULT_PROCESS_OPTS = dict(
    only_alphabetic = True,
    to_lower = False,
    remove_accents = False,
    min_len = 2,
    max_len = 100,
    keep_numerals = True,
    remove_stopwords = False,
    extra_stopwords = None,
    stopwords = None,
    language = "swedish",
    keep_symbols = True
)

class TokensTransformer():
    """Transforms applied on tokenized text"""
    def __init__(self,
        only_alphabetic: bool = False,
        to_lower: bool = False,
        to_upper: bool = False,
        min_len: int = None,
        max_len: int = None,
        remove_accents: bool = False,
        remove_stopwords: bool = False,
        stopwords: Any = None,
        extra_stopwords: List[str] = None,
        language: str = "swedish",
        keep_numerals: bool = True,
        keep_symbols: bool = True
    ):
        self.transforms = []

        self.min_chars_filter(1)

        if to_lower:
            self.to_lower()

        if to_upper:
            self.to_lower()

        if max_len is not None:
            self.max_chars_filter(max_len)

        if keep_symbols is False:
            self.remove_symbols()

        if remove_accents:
            self.remove_accents()

        if min_len > 1:
            self.min_chars_filter(min_len)

        if only_alphabetic:
            self.only_alphabetic()

        if keep_numerals is False:
            self.remove_numerals()

        if remove_stopwords or (stopwords is not None):
            self.remove_stopwords(language_or_stopwords=(stopwords or language), extra_stopwords=extra_stopwords)

    def add(self, transform) -> TokensTransformer:
        self.transforms.append(transform)
        return self

    def transform(self, tokens) -> TokensTransformer:

        for ft in self.transforms:
            tokens = ft(tokens)

        return tokens

    def min_chars_filter(self, n_chars) -> TokensTransformer:
        if (n_chars or 0) < 1:
            return self
        return self.add(utility.min_chars_filter(n_chars))

    def max_chars_filter(self, n_chars) -> TokensTransformer:
        if (n_chars or 0) < 1:
            return self
        return self.add(utility.max_chars_filter(n_chars))

    def to_lower(self) -> TokensTransformer:
        return self.add(utility.lower_transform())

    def to_upper(self) -> TokensTransformer:
        return self.add(utility.upper_transform())

    def remove_symbols(self) -> TokensTransformer:
        return self.add(utility.remove_symbols()).add(utility.min_chars_filter(1))

    def only_alphabetic(self) -> TokensTransformer:
        return self.add(utility.only_alphabetic_filter())

    def remove_numerals(self) -> TokensTransformer:
        return self.add(utility.remove_numerals())

    def remove_stopwords(self, language_or_stopwords=None, extra_stopwords=None) -> TokensTransformer:
        if language_or_stopwords is None:
            return self
        return self.add(utility.remove_stopwords(language_or_stopwords, extra_stopwords))

    def remove_accents(self) -> TokensTransformer:
        return self.add(textacy_remove.remove_accents)
