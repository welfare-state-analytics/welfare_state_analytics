from __future__ import annotations

import types

from typing import Callable, List

import textacy.preprocessing.normalize as normalize

TRANSFORMS = types.SimpleNamespace(
    fix_hyphenation=normalize.normalize_hyphenated_words,
    fix_unicode=normalize.normalize_unicode,
    fix_whitespaces=normalize.normalize_whitespace
)

class TextTransformer():
    """Transforms applied on non-tokenized text
    """
    def __init__(self,
        transforms: List[Callable] = None
    ):
        self.transforms: List[Callable] = transforms or []

    def add(self, transform, condition=True) -> TextTransformer:
        if condition:
            self.transforms.append(transform)
        return self

    def transform(self, text: str) -> str:

        for ft in self.transforms:
            text = ft(text)

        return text.strip()

    def fix_hyphenation(self) -> TextTransformer:
        return self.add(TRANSFORMS.fix_hyphenation)

    def fix_unicode(self) -> TextTransformer:
        return self.add(TRANSFORMS.fix_unicode)

    def fix_whitespaces(self) -> TextTransformer:
        return self.add(TRANSFORMS.fix_whitespace)
