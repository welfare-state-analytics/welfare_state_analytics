import os
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from gensim.matutils import Sparse2Corpus
from penelope.utility import flatten

PUBLICATION2ID = {'AFTONBLADET': 1, 'EXPRESSEN': 2, 'DAGENS NYHETER': 3, 'SVENSKA DAGBLADET': 4}
ID2PUBLICATION = {v: k for k, v in PUBLICATION2ID.items()}
ID2PUB = {1: 'AB', 2: 'EX', 3: 'DN', 4: 'SVD'}
PUB2ID = {v: k for k, v in ID2PUB.items()}

corpus_dataset_filename: str = "corpus_dataset.zip"
document_dataset_filename: str = "document_dataset.zip"
vocabulary_dataset_filename: str = "vocabulary_dataset.zip"
censored_corpus_filename: str = "text1_utantext.zip"
meta_textblocks_filename: str = "meta_textblocks.zip"
sparse_matrix_filename: str = "corpus_sparse_doc_term_matrx.npz"
document_processed_filename: str = "document_processed_dataset.zip"
reconstructed_text_corpus_file: str = "reconstructed_text_corpus.csv.zip"


def load_corpus_dtm_as_data_frame(corpus_folder) -> pd.DataFrame:
    """Load corpus DTM, source "dtm1.rds", arrays drm$i, drm$j, drm$v'"""

    filename: str = os.path.join(corpus_folder, corpus_dataset_filename)

    df_corpus: pd.DataFrame = pd.read_csv(
        filename, compression='zip', header=0, sep=',', quotechar='"', na_filter=False
    )
    df_corpus.columns = ["document_id", "token_id", "tf"]
    df_corpus.document_id -= 1
    df_corpus.token_id -= 1

    return df_corpus


def load_vocabulary_file_as_data_frame(corpus_folder: str) -> pd.DataFrame:
    """Load vocabulary"""

    filename: str = os.path.join(corpus_folder, vocabulary_dataset_filename)

    df_vocabulary: pd.DataFrame = pd.read_csv(
        filename, compression='zip', header=0, sep=',', quotechar='"', na_filter=False
    )
    df_vocabulary.columns = ["token"]

    return df_vocabulary


def load_censured_text_as_data_frame(corpus_folder) -> pd.DataFrame:
    """ Load censored corpus data """

    filename = os.path.join(corpus_folder, censored_corpus_filename)

    df_censured_text: pd.DataFrame = pd.read_csv(
        filename, compression='zip', header=0, sep=',', quotechar='"', na_filter=False
    )
    df_censured_text.columns = ['id', 'doc_id', 'publication', 'date']
    df_censured_text.id -= 1

    df_censured_text = df_censured_text[['doc_id', 'publication', 'date']].drop_duplicates()

    return df_censured_text


def load_meta_text_blocks_as_data_frame(corpus_folder: str) -> pd.DataFrame:
    """ Load censored corpus data """

    filename: str = os.path.join(corpus_folder, meta_textblocks_filename)
    df_meta: pd.DataFrame = pd.read_csv(filename, compression='zip', header=0, sep=',', quotechar='"', na_filter=False)
    df_meta = df_meta[['id', 'pred_bodytext']].drop_duplicates()
    df_meta.columns = ["doc_id", "pred_bodytext"]
    df_meta = df_meta.set_index("doc_id")
    return df_meta


def load(corpus_folder) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df_corpus: pd.DataFrame = load_corpus_dtm_as_data_frame(corpus_folder)
    df_vocabulary: pd.DataFrame = load_vocabulary_file_as_data_frame(corpus_folder)
    df_document: pd.DataFrame = load_document_index(corpus_folder)

    return df_corpus, df_document, df_vocabulary


def load_as_sparse_matrix(corpus_folder) -> sp.coo_matrix:

    filename: str = os.path.join(corpus_folder, sparse_matrix_filename)

    if not os.path.isfile(filename):
        df_corpus: pd.DataFrame = load_corpus_dtm_as_data_frame(corpus_folder)
        dtm: sp.coo_matrix = sp.coo_matrix((df_corpus.tf, (df_corpus.document_id, df_corpus.token_id)))  # type: ignore
        sp.save_npz(filename, dtm, compressed=True)  # type: ignore
    else:
        dtm: sp.coo_matrix = sp.load_npz(filename)  # type: ignore

    return dtm


def load_reconstructed_text_corpus(corpus_folder: str) -> pd.DataFrame:
    filename: str = os.path.join(corpus_folder, reconstructed_text_corpus_file)
    if not os.path.isfile(filename):
        df_corpus: pd.DataFrame = load_corpus_dtm_as_data_frame(corpus_folder)
        df_vocabulary: pd.DataFrame = load_vocabulary_file_as_data_frame(corpus_folder)
        id2token: dict = df_vocabulary['token'].to_dict()
        df_reconstructed_text_corpus: pd.DataFrame = (df_corpus.groupby('document_id')).apply(
            lambda x: ' '.join(flatten(x['tf'] * (x['token_id'].apply(lambda y: [id2token[y]]))))
        )
        df_reconstructed_text_corpus.to_csv(filename, compression='zip', header=0, sep=',', quotechar='"')  # type: ignore
    else:
        df_reconstructed_text_corpus: pd.DataFrame = pd.read_csv(filename, compression='zip', header=None, sep=',', quotechar='"')  # type: ignore
        df_reconstructed_text_corpus.columns = ['document_id', 'text']
        df_reconstructed_text_corpus = df_reconstructed_text_corpus.set_index('document_id')

    return df_reconstructed_text_corpus


def load_as_dtm(corpus_folder) -> Tuple[Any, pd.DataFrame, Mapping[int, Any]]:

    dtm = load_as_sparse_matrix(corpus_folder)
    id2token: Mapping[int, str] = load_vocabulary_file_as_data_frame(corpus_folder)['token'].to_dict()  # type: ignore
    document_index: pd.DataFrame = load_document_index(corpus_folder)

    return dtm, document_index, id2token


def load_as_dtm2(
    corpus_folder: str, publication_ids: List[str] = None
) -> Tuple[sp.csr_matrix, pd.DataFrame, Mapping[int, str]]:

    dtm, document_index, id2token = load_as_dtm(corpus_folder)

    if publication_ids is not None:

        document_index: pd.DataFrame = document_index[document_index.publication_id.isin(publication_ids)]
        dtm: sp.csr_matrix = dtm.tocsr()[document_index.index, :]
        document_index: pd.DataFrame = document_index.reset_index().drop('id', axis=1)
        token_ids = dtm.sum(axis=0).nonzero()[1]
        dtm = dtm[:, token_ids]
        id2token: Mapping[int, Any] = {i: id2token[k] for i, k in enumerate(token_ids)}

    return dtm, document_index, id2token


def load_as_gensim_sparse_corpus(corpus_folder: str) -> Tuple[Sparse2Corpus, pd.DataFrame, dict]:
    dtm, document_index, id2token = load_as_dtm(corpus_folder)
    corpus: Sparse2Corpus = Sparse2Corpus(dtm, documents_columns=False)
    document_index['n_terms'] = np.asarray(corpus.sparse.sum(axis=0)).reshape(-1).astype(np.uint32)  # type: ignore
    return corpus, document_index, id2token


def load_as_gensim_sparse_corpus2(
    corpus_folder: str, publication_ids: List[str] = None
) -> Tuple[Sparse2Corpus, pd.DataFrame, dict]:

    dtm, document_index, id2token = load_as_dtm(corpus_folder)

    if publication_ids is not None:
        document_index = document_index[document_index.publication_id.isin(publication_ids)]
        dtm: sp.csc_matrix = dtm.tocsr()[document_index.index, :]
        document_index = document_index.reset_index().drop('id', axis=1)
        token_ids = dtm.sum(axis=0).nonzero()[1]
        dtm = dtm[:, token_ids]
        id2token = {i: id2token[k] for i, k in enumerate(token_ids)}

    corpus: Sparse2Corpus = Sparse2Corpus(dtm, documents_columns=False)
    document_index['n_terms'] = np.asarray(corpus.sparse.sum(axis=0)).reshape(-1).astype(np.uint32)  # type: ignore
    return corpus, document_index, id2token


def load_dates_subset_as_dtm(corpus_folder: str, dates) -> Tuple[sp.csr_matrix, pd.DataFrame, Mapping[int, str]]:
    dtm, document_index, id2token = load_as_dtm(corpus_folder)
    document_index = document_index[document_index.date.isin(dates)]
    dtm = dtm.tocsr()[document_index.index, :]
    document_index = document_index.reset_index().drop('id', axis=1)
    token_ids = dtm.sum(axis=0).nonzero()[1]
    dtm = dtm[:, token_ids]
    id2token = {i: id2token[k] for i, k in enumerate(token_ids)}
    return dtm, document_index, id2token


def slim_documents(document_index):

    # df = document_index[['publication', 'year']].copy()
    # df['publication_id'] = df.publication.apply(lambda x: PUBLICATION2ID[x]).astype(np.uint16)
    # df = df[['publication_id', 'year']]
    return document_index[['publication_id', 'year']].copy()


# pylint: skip-file
def load_document_index(corpus_folder, force=False):
    """ Load document_index data, source "dtm1.rds", arrays drm$dimnames[1] """

    processed_filename = os.path.join(corpus_folder, document_processed_filename)

    if not os.path.isfile(processed_filename) or force:

        filename = os.path.join(corpus_folder, document_dataset_filename)

        document_index = pd.read_csv(filename, compression='zip', header=0, sep=',', quotechar='"', na_filter=False)
        document_index.columns = ["doc_id"]
        document_index.index.name = 'id'

        # Add publication and date
        df_censured_text = load_censured_text_as_data_frame(corpus_folder)
        document_index = pd.merge(
            document_index,
            df_censured_text[['doc_id', 'publication', 'date']],
            how='inner',
            left_on='doc_id',
            right_on='doc_id',
        )

        # Add pred_bodytext
        df_meta = load_meta_text_blocks_as_data_frame(corpus_folder)
        document_index = pd.merge(document_index, df_meta, how='inner', left_on='doc_id', right_index=True)

        # Add year
        document_index['date'] = pd.to_datetime(document_index.date)
        document_index['year'] = document_index.date.dt.year
        document_index['publication_id'] = document_index.publication.apply(lambda x: PUBLICATION2ID[x]).astype(
            np.uint16
        )

        document_index.to_csv(
            processed_filename, compression='zip', header=True, sep=',', quotechar='"', index=True, index_label="id"
        )

    else:
        # loading cached...
        document_index = pd.read_csv(
            processed_filename, compression='zip', header=0, sep=',', quotechar='"', na_filter=False, index_col="id"
        )
        if 'publication_id' not in document_index.columns:
            document_index['publication_id'] = document_index.publication.apply(lambda x: PUBLICATION2ID[x]).astype(
                np.uint16
            )

    return document_index
