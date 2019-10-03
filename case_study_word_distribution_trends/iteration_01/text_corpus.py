import re
import nltk.tokenize
import utility
               
class CorpusTextStream():

    def __init__(self, reader):
        self.reader = reader

    def get_index(self):

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
    
    def __init__(self, reader, tokenizer=None, isalnum=True, to_lower=False, deacc=False, min_len=2, max_len=None, numerals=True):
        super().__init__(reader, tokenizer=tokenizer, isalnum=isalnum)
        self.to_lower = to_lower
        self.deacc = deacc
        self.min_len = min_len
        self.max_len = max_len
        self.numerals = numerals
        
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
            
            tokens = list(tokens)
            filename = meta if isinstance(meta, str) else meta.filename
            self.n_tokens[filename] = len(tokens)
            
            yield meta, tokens
    
