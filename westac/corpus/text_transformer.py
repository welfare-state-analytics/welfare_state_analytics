from __future__ import annotations

from typing import List, Callable

import textacy.preprocessing.normalize as normalize

from westac.corpus import utility

class TextTransformer():
    """Transforms applied on non-tokenized text
    """
    def __init__(self,
        fix_hyphenation: bool = False,
        fix_unicode: bool = False
    ):
        self.transforms: List[Callable] = []

        if fix_hyphenation:
            self.fix_hyphenation()

        if fix_unicode:
            self.fix_unicode()

    def add(self, transform) -> TextTransformer:
        self.transforms.append(transform)
        return self

    def transform(self, text: str) -> str:

        for ft in self.transforms:
            text = ft(text)

        return text

    def fix_hyphenation(self) -> TextTransformer:
        return self.add(normalize.normalize_hyphenated_words)

    def fix_unicode(self) -> TextTransformer:
        return self.add(normalize.normalize_unicode)

    def fix_whitespaces(self) -> TextTransformer:
        return self.add(normalize.normalize_whitespace)

        
