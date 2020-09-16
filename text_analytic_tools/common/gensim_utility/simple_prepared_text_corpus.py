import logging

from text_analytic_tools.common.gensim_utility import GenericTextCorpus, CompressedFileReader

logger = logging.getLogger(__name__)

class SimplePreparedTextCorpus(GenericTextCorpus):
    """Reads content in stream and returns tokenized text. No other processing.
    """
    def __init__(self, source, lowercase=False, itemfilter=None):

        self.reader = CompressedFileReader(source, itemfilter=itemfilter)
        self.filenames = self.reader.filenames
        self.lowercase = lowercase
        source = self.reader
        super(SimplePreparedTextCorpus, self).__init__(source)

    def default_token_filters(self):

        token_filters = [
            (lambda tokens: [ x.strip('_') for x in tokens ]),
        ]

        if self.lowercase:
            token_filters = token_filters + [ (lambda tokens: [ x.lower() for x in tokens ]) ]

        return token_filters

    def preprocess_text(self, text):
        return self.tokenizer(text)
