import glob
from os.path import join as jj

import pandas as pd
from penelope import corpus
from tqdm import tqdm


def load():

    source_folder: str = '/data/westac/riksdagen_corpus_data/tagged-speech-corpus.numeric.feather'
    vocabulary = pd.read_feather(jj(source_folder, 'token2id.feather'))
    document_index: pd.DataFrame = corpus.DocumentIndexHelper.load(
        jj(source_folder, 'document_index.feather')
    ).document_index
    document_index['protocol_name'] = document_index.document_name.str.split('_').str[0]
    # protocol_speeches: dict = document_index.groupby('protocol_name').agg({'document_id': list})['document_id'].to_dict()

    id2token: dict = vocabulary.set_index('token_id').token.to_dict()
    assert document_index is not None

    filenames = sorted(glob.glob(jj(source_folder, '**/prot-*.feather'), recursive=True))

    # assert len(filenames) == len(document_index.document_name.apply(lambda x: x.split('_')[0]).unique())

    n_tokens = 0
    fg = id2token.get
    for filename in tqdm(filenames, total=len(filenames)):
        # document_name: str = utility.strip_path_and_extension(filename)
        group_frame: pd.DataFrame = pd.read_feather(filename)
        for document_id, speech_frame in group_frame.groupby('document_id'):
            n_tokens += len(speech_frame)
            speech_frame['token'] = speech_frame.token_id.apply(fg)


if __name__ == '__main__':
    load()
