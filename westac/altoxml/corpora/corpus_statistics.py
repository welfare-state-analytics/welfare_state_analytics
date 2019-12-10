import collections
import pandas as pd

from corpora.corpus_source_reader import SparvCorpusSourceReader
from corpora.zip_utility import ZipReader
from common.utility import getLogger, extend

logger = getLogger(__name__)

KindOfPoS = collections.namedtuple('KindOfPoS', 'tag description is_deliminator')

SUC_POS_TAGS = {
    'AB': KindOfPoS(tag='AB',   description={'en': 'Adverb', 'se': 'Adverb' }, is_deliminator=False),
    'DT': KindOfPoS(tag='DT',   description={'en': 'Determiner', 'se': 'Determinerare, bestämningsord' }, is_deliminator=False),
    'HA': KindOfPoS(tag='HA',   description={'en': 'Interrogative/Relative Adverb', 'se': 'Frågande/relativt adverb' }, is_deliminator=False),
    'HD': KindOfPoS(tag='HD',   description={'en': 'Interrogative/Relative Determiner', 'se': 'Frågande/relativ bestämning' }, is_deliminator=False),
    'HP': KindOfPoS(tag='HP',   description={'en': 'Interrogative/Relative Pronoun', 'se': 'Frågande/relativt pronomen' }, is_deliminator=False),
    'HS': KindOfPoS(tag='HS',   description={'en': 'Interrogative/Relative Possessive', 'se': 'Frågande/relativt possessivuttryck' }, is_deliminator=False),
    'IE': KindOfPoS(tag='IE',   description={'en': 'Infinitival marker', 'se': 'Infinitivmärke' }, is_deliminator=False),
    'IN': KindOfPoS(tag='IN',   description={'en': 'Interjection', 'se': 'Interjektion' }, is_deliminator=False),
    'JJ': KindOfPoS(tag='JJ',   description={'en': 'Adjective', 'se': 'Adjektiv' }, is_deliminator=False),
    'KN': KindOfPoS(tag='KN',   description={'en': 'Conjunction', 'se': 'Konjunktion' }, is_deliminator=False),
    'NN': KindOfPoS(tag='NN',   description={'en': 'Noun', 'se': 'Substantiv' }, is_deliminator=False),
    'PC': KindOfPoS(tag='PC',   description={'en': 'Participle', 'se': 'Particip' }, is_deliminator=False),
    'PL': KindOfPoS(tag='PL',   description={'en': 'Particle', 'se': 'Partikel' }, is_deliminator=False),
    'PM': KindOfPoS(tag='PM',   description={'en': 'Proper Noun', 'se': 'Egennamn' }, is_deliminator=False),
    'PN': KindOfPoS(tag='PN',   description={'en': 'Pronoun', 'se': 'Pronomen' }, is_deliminator=False),
    'PP': KindOfPoS(tag='PP',   description={'en': 'Preposition', 'se': 'Preposition' }, is_deliminator=False),
    'PS': KindOfPoS(tag='PS',   description={'en': 'Possessive pronoun', 'se': 'Possessivuttryck' }, is_deliminator=False),
    'RG': KindOfPoS(tag='RG',   description={'en': 'Cardinal number', 'se': 'Räkneord: grundtal' }, is_deliminator=False),
    'RO': KindOfPoS(tag='RO',   description={'en': 'Ordinal number', 'se': 'Räkneord: ordningstal' }, is_deliminator=False),
    'SN': KindOfPoS(tag='SN',   description={'en': 'Subjunction', 'se': 'Subjunktion' }, is_deliminator=False),
    'VB': KindOfPoS(tag='VB',   description={'en': 'Verb', 'se': 'Verb' }, is_deliminator=False),
    'UO': KindOfPoS(tag='UO',   description={'en': 'Foreign word', 'se': 'Utländskt ord' }, is_deliminator=False),
    'MAD': KindOfPoS(tag='MAD', description={'en': 'Major delimiter', 'se': 'Meningsskiljande interpunktion' }, is_deliminator=True),
    'MID': KindOfPoS(tag='MID', description={'en': 'Minor delimiter', 'se': 'Interpunktion' }, is_deliminator=True),
    'PAD': KindOfPoS(tag='PAD', description={'en': 'Pairwise delimiter', 'se': 'Interpunktion' }, is_deliminator=True),
    '???': KindOfPoS(tag='???', description={'en': 'Unknown', 'se': '' }, is_deliminator=False)
}

PENN_TREEBANK_POS_TAGS = {
    'CC': KindOfPoS(tag='CC', description={'en': 'Coordinating conjunction', 'se': 'Konjunktion'}, is_deliminator=False),
    'CD': KindOfPoS(tag='CD', description={'en': 'Cardinal number', 'se': 'Räkneord: ordningstal'}, is_deliminator=False),
    'DT': KindOfPoS(tag='DT', description={'en': 'Determiner', 'se': 'Determinerare'}, is_deliminator=False),
    'EX': KindOfPoS(tag='EX', description={'en': 'Existential there', 'se': ''}, is_deliminator=False),
    'UO': KindOfPoS(tag='UO',   description={'en': 'Foreign word', 'se': 'Utländskt ord' }, is_deliminator=False),
    'FW': KindOfPoS(tag='FW', description={'en': 'Foreign word', 'se': ''}, is_deliminator=False),
    'IN': KindOfPoS(tag='IN', description={'en': 'Preposition or subordinating conjunction', 'se': ''}, is_deliminator=False),
    'JJ': KindOfPoS(tag='JJ', description={'en': 'Adjective', 'se': 'Adjektiv'}, is_deliminator=False),
    'JJR': KindOfPoS(tag='JJR', description={'en': 'Adjective, comparative', 'se': ''}, is_deliminator=False),
    'JJS': KindOfPoS(tag='JJS', description={'en': 'Adjective, superlative', 'se': ''}, is_deliminator=False),
    'LS': KindOfPoS(tag='LS', description={'en': 'List item marker', 'se': ''}, is_deliminator=False),
    'MD': KindOfPoS(tag='MD', description={'en': 'Modal', 'se': ''}, is_deliminator=False),
    'NN': KindOfPoS(tag='NN', description={'en': 'Noun, singular or mass', 'se': ''}, is_deliminator=False),
    'NNS': KindOfPoS(tag='NNS', description={'en': 'Noun, plural', 'se': ''}, is_deliminator=False),
    'NNP': KindOfPoS(tag='NNP', description={'en': 'Proper noun, singular', 'se': ''}, is_deliminator=False),
    'NNPS': KindOfPoS(tag='NNPS', description={'en': 'Proper noun, plural', 'se': ''}, is_deliminator=False),
    'PDT': KindOfPoS(tag='PDT', description={'en': 'Predeterminer', 'se': ''}, is_deliminator=False),
    'POS': KindOfPoS(tag='POS', description={'en': 'Possessive ending', 'se': ''}, is_deliminator=False),
    'PRP': KindOfPoS(tag='PRP', description={'en': 'Personal pronoun', 'se': ''}, is_deliminator=False),
    'PRP$': KindOfPoS(tag='PRP$', description={'en': 'Possessive pronoun', 'se': ''}, is_deliminator=False),
    'RB': KindOfPoS(tag='RB', description={'en': 'Adverb', 'se': ''}, is_deliminator=False),
    'RBR': KindOfPoS(tag='RBR', description={'en': 'Adverb, comparative', 'se': ''}, is_deliminator=False),
    'RBS': KindOfPoS(tag='RBS', description={'en': 'Adverb, superlative', 'se': ''}, is_deliminator=False),
    'RP': KindOfPoS(tag='RP', description={'en': 'Particle', 'se': ''}, is_deliminator=False),
    'SYM': KindOfPoS(tag='SYM', description={'en': 'Symbol', 'se': ''}, is_deliminator=False),
    'TO': KindOfPoS(tag='TO', description={'en': 'to', 'se': ''}, is_deliminator=False),
    'UH': KindOfPoS(tag='UH', description={'en': 'Interjection', 'se': ''}, is_deliminator=False),
    'VB': KindOfPoS(tag='VB', description={'en': 'Verb, base form', 'se': ''}, is_deliminator=False),
    'VBD': KindOfPoS(tag='VBD', description={'en': 'Verb, past tense', 'se': ''}, is_deliminator=False),
    'VBG': KindOfPoS(tag='VBG', description={'en': 'Verb, gerund or present participle', 'se': ''}, is_deliminator=False),
    'VBN': KindOfPoS(tag='VBN', description={'en': 'Verb, past participle', 'se': ''}, is_deliminator=False),
    'VBP': KindOfPoS(tag='VBP', description={'en': 'Verb, non-3rd person singular present', 'se': ''}, is_deliminator=False),
    'VBZ': KindOfPoS(tag='VBZ', description={'en': 'Verb, 3rd person singular present', 'se': ''}, is_deliminator=False),
    'WDT': KindOfPoS(tag='WDT', description={'en': 'Wh-determiner', 'se': ''}, is_deliminator=False),
    'WP': KindOfPoS(tag='WP', description={'en': 'Wh-pronoun', 'se': ''}, is_deliminator=False),
    'WP$': KindOfPoS(tag='WP$', description={'en': 'Possessive wh-pronoun', 'se': ''}, is_deliminator=False),
    'WRB': KindOfPoS(tag='WRB', description={'en': 'Wh-adverb', 'se': ''}, is_deliminator=False),
}

def create_sparv_stream(filepath, postags="''", transforms=None, lemmatize=False, pos_delimiter="_"):
    reader = ZipReader(filepath, '*.xml')
    stream = SparvCorpusSourceReader(
        source=reader,
        transforms=(transforms or [ lambda tokens: [ x.lower()  for x in tokens] ]),
        postags=postags,
        chunk_size=None,
        lemmatize=lemmatize,
        append_pos=pos_delimiter,
        ignores=""
    )
    return stream

class CorpusPoSStatistics():
    """Computes PoS statistics from a corpus source stream

    Example usage:

    def create_sparv_stream(filepath, postags="", transforms=None, lemmatize=False, pos_delimiter="_"):
       reader = ZipReader(filepath, '*.xml')
       stream = SparvCorpusSourceReader(
            source=reader,
            transforms=(transforms or [ lambda tokens: [ x.lower()  for x in tokens] ]),
            postags=postags,
            chunk_size=None,
            lemmatize=lemmatize,
            append_pos=pos_delimiter,
            ignores=""
        )
        return stream

    filename = '.\\temp\\daedalus_articles_pos_xml_1931-2017.zip'
    filepath = join_test_data_path(filename)

    column_functions = [
        ('year', lambda df: df.filename.apply(lambda x: int(re.search(r'daedalus_volume_(\\d{4})', x).group(1)))),
        ('article', lambda df: df.filename.apply(lambda x: int(re.search(r'article_(\\d{2})', x).group(1)))),
        ('segment', lambda df: df.filename.apply(lambda x: int(re.search(r'(\\d{2})\\.txt', x).group(1))))
    ]

    stream = create_sparv_stream(filepath)
    df = CorpusPoSStatistics(SUC_POS_TAGS).generate(stream)
    print(df)
    """

    def __init__(self, tag_set):

        self.tag_set = tag_set

    def generate(self, stream, column_functions=None):

        pos_tags = { k: 0 for k in self.tag_set.keys() }
        pos_delimiter = stream.append_pos
        pos_statistics = []
        pos_total_counter = collections.Counter()
        for document, tokens in stream:

            counter = collections.Counter([ x.split(pos_delimiter)[-1].upper() if pos_delimiter in x else '???' for x in tokens ])
            pos_total_counter.update(counter)

            counter_dict = dict(counter)

            pos_counts = extend(pos_tags, { k: v for k, v in counter_dict.items() if k in pos_tags.keys() })
            other_counts = [ k for k in counter_dict.keys() if k not in pos_tags.keys() and k != '' ]

            if len(other_counts) > 0:
                logger.warning('Warning strange PoS tags: File %s, tags %s', document, other_counts)

            pos_statistics.append(extend(pos_counts, filename=document))

        df = pd.DataFrame(pos_statistics)
        for (column_name, column_function) in (column_functions or []):
            df[column_name] = column_function(df)

        return df

        #corpus = SparvTextCorpus(stream, prune_at=2000000)
        #for document, tokens in corpus:
        #    print("{}: {}".format(document, len(tokens)))

