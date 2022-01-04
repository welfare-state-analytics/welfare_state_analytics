from __future__ import annotations

import os
import zipfile
from typing import Any, List, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from gensim.matutils import Sparse2Corpus
from penelope.utility import deprecated, flatten

PUBLICATION2ID = {'AFTONBLADET': 1, 'EXPRESSEN': 2, 'DAGENS NYHETER': 3, 'SVENSKA DAGBLADET': 4}
ID2PUBLICATION = {v: k for k, v in PUBLICATION2ID.items()}
ID2PUB = {1: 'AB', 2: 'EX', 3: 'DN', 4: 'SVD'}
PUB2ID = {v: k for k, v in ID2PUB.items()}

CORPUS_DATASET_FILENAME: str = "corpus_dataset.zip"
DOCUMENT_DATASET_FILENAME: str = "document_dataset.zip"
VOCABULARY_DATASET_FILENAME: str = "vocabulary_dataset.zip"
CENSORED_CORPUS_FILENAME: str = "text1_utantext.zip"
META_TEXTBLOCKS_FILENAME: str = "meta_textblocks.zip"
RECONSTRUCTED_TEXT_CORPUS_FILE: str = "reconstructed_text_corpus.csv.zip"

"""Cache/processed data"""
SPARSE_MATRIX_FILENAME: str = "corpus_sparse_doc_term_matrx.npz"
DOCUMENT_PROCESSED_FILENAME: str = "document_processed_dataset.zip"

# pylint: disable=no-member, unsupported-assignment-operation, unsubscriptable-object


class SourceCorpus:
    def __init__(
        self,
        folder: str,
        corpus: pd.DataFrame | sp.spmatrix | Sparse2Corpus,
        vocabulary: pd.DataFrame,
        document_index: pd.DataFrame,
    ):

        self.folder: str = folder
        self.corpus: pd.DataFrame | sp.spmatrix | Sparse2Corpus = corpus
        self.vocabulary: pd.DataFrame = vocabulary
        self.document_index: pd.DataFrame = document_index
        self.id2token: dict = self.vocabulary['token'].to_dict()

    def clone(
        self,
        *,
        corpus: pd.DataFrame | sp.spmatrix | Sparse2Corpus = None,
        vocabulary: pd.DataFrame = None,
        document_index: pd.DataFrame = None,
    ):
        return SourceCorpus(
            folder=self.folder,
            corpus=corpus if corpus is not None else self.corpus,
            vocabulary=vocabulary if vocabulary is not None else self.vocabulary,
            document_index=document_index if document_index is not None else self.document_index,
        )

    def _to_coo_matrix(self) -> sp.coo_matrix:
        """Create a sparse matrix"""
        filename: str = os.path.join(self.folder, SPARSE_MATRIX_FILENAME)
        if not os.path.isfile(filename):
            dtm: sp.coo_matrix = sp.coo_matrix((self.corpus.tf, (self.corpus.document_id, self.corpus.token_id)))  # type: ignore
            sp.save_npz(filename, dtm, compressed=True)  # type: ignore
        else:
            dtm: sp.coo_matrix = sp.load_npz(filename)  # type: ignore
        return dtm

    def to_coo_corpus(self, inplace=True) -> "SourceCorpus":
        """Transforms dataframecorpus to a COO matrix DTM"""
        if not isinstance(self.corpus, pd.DataFrame):
            raise ValueError("to_coo_corpus only allowed for dataframe corpus")
        if not inplace:
            return self.clone(corpus=self._to_coo_matrix())
        self.corpus = self._to_coo_matrix()
        return self

    def slice_by_document_ids(self, document_index: pd.DataFrame, inplace=True) -> "SourceCorpus":

        if not sp.issparse(self.corpus):
            raise ValueError("slice_by_document_ids only allowed for sparse corpus")

        dtm: sp.csr_matrix = self.corpus.tocsr()[document_index.index, :]

        token_ids = dtm.sum(axis=0).nonzero()[1]
        id2token: Mapping[int, Any] = {i: self.id2token[k] for i, k in enumerate(token_ids)}
        vocabulary: pd.DataFrame = pd.DataFrame(
            data={'token_id': id2token.keys(), 'token': id2token.values()}
        ).set_index('token_id')

        dtm = dtm[:, token_ids]

        document_index: pd.DataFrame = document_index.reset_index().drop('id', axis=1)

        if not inplace:
            return self.clone(corpus=dtm, vocabulary=vocabulary, document_index=document_index)

        self.corpus = dtm
        self.vocabulary = vocabulary
        self.document_index = document_index
        self.id2token = id2token

        return self

    def slice_by_publications(self, publication_ids: List[str] = None, inplace=True) -> "SourceCorpus":
        """slice corpus by set of publications"""

        if not sp.issparse(self.corpus):
            raise ValueError("slice_by_publications only allowed for sparse corpus")

        if publication_ids:

            document_index: pd.DataFrame = self.document_index[self.document_index.publication_id.isin(publication_ids)]

            return self.slice_by_document_ids(document_index, inplace)

        return self

    @deprecated
    def to_gensim_sparse_corpus(self, inplace=True) -> "SourceCorpus":
        """load_as_gensim_sparse_corpus"""

        corpus: Sparse2Corpus = Sparse2Corpus(self.corpus, documents_columns=False)
        document_index: pd.DataFrame = self.document_index
        document_index['n_tokens'] = np.asarray(corpus.sparse.sum(axis=0)).reshape(-1).astype(np.uint32)  # type: ignore

        if inplace:
            self.corpus = corpus
            self.document_index = document_index
            return self

        return self.clone(corpus=corpus)

    @deprecated
    def to_gensim_sparse_corpus2(self, publication_ids: List[str] = None, inplace=True) -> "SourceCorpus":

        return (
            self.to_coo_corpus(inplace).slice_by_publications(publication_ids, inplace).to_gensim_sparse_corpus(inplace)
        )

    @deprecated
    def slice_by_dates(self, dates: Sequence[Any], inplace=True) -> "SourceCorpus":
        """load_dates_subset_as_dtm"""
        document_index: pd.DataFrame = self.document_index[self.document_index.date.isin(dates)]
        return self.slice_by_document_ids(document_index, inplace=inplace)

    @deprecated
    def slim_documents(self) -> pd.DataFrame:
        # df = document_index[['publication', 'year']].copy()
        # df['publication_id'] = df.publication.apply(lambda x: PUBLICATION2ID[x]).astype(np.uint16)
        # df = df[['publication_id', 'year']]
        return self.document_index[['publication_id', 'year']].copy()

    def info(self) -> None:

        print('Corpus metrics, source "dtm1.rds", arrays drm$i, drm$j, drm$v')
        print("  {} max document ID".format(self.corpus.document_id.max()))
        print("  {} unique document ID".format(self.corpus.document_id.unique().shape[0]))
        print("  {} max token ID".format(self.corpus.token_id.max()))
        print("  {} unique token ID".format(self.corpus.token_id.unique().shape[0]))

        print('Document metrics, source "dtm1.rds", arrays drm$dimnames[1]')
        print("  {} max ID".format(self.document_index.index.max()))
        print("  {} unique ID".format(self.document_index.index.unique().shape[0]))
        print("  {} unique names".format(self.document_index.doc_id.unique().shape[0]))

        print('Vocabulary metrics, source "dtm1.rds", arrays drm$dimnames[2]')
        print("  {} max ID".format(self.vocabulary.index.max()))
        print("  {} unique ID".format(self.vocabulary.index.unique().shape[0]))
        print("  {} unique token".format(self.vocabulary.token.unique().shape[0]))


class SourceRepository:
    @staticmethod
    def load(folder: str) -> "SourceCorpus":

        corpus: pd.DataFrame = SourceRepository._load_source_dtm(folder)
        vocabulary: pd.DataFrame = SourceRepository._load_source_vocabulary(folder)
        document_index: pd.DataFrame = SourceRepository.load_document_index(folder)
        document_index['tf'] = corpus.groupby(["document_id"])['tf'].sum()

        return SourceCorpus(folder, corpus, vocabulary, document_index)

    @staticmethod
    def _load_source_dtm(folder: str) -> pd.DataFrame:
        """Load corpus DTM, source "dtm1.rds", arrays drm$i, drm$j, drm$v'"""

        filename: str = os.path.join(folder, CORPUS_DATASET_FILENAME)

        dtm_data: pd.DataFrame = pd.read_csv(
            filename, compression='zip', header=0, sep=',', quotechar='"', na_filter=False
        )
        dtm_data.columns = ["document_id", "token_id", "tf"]
        dtm_data.document_id -= 1  # pylint: disable=no-member
        dtm_data.token_id -= 1  # pylint: disable=no-member

        return dtm_data

    @staticmethod
    def _load_source_vocabulary(folder: str) -> pd.DataFrame:
        """Load vocabulary"""

        filename: str = os.path.join(folder, VOCABULARY_DATASET_FILENAME)

        vocabulary: pd.DataFrame = pd.read_csv(
            filename, compression='zip', header=0, sep=',', quotechar='"', na_filter=False
        )
        vocabulary.columns = ["token"]

        return vocabulary

    @staticmethod
    def load_document_index(folder, force=False):
        """ Load document_index data, source "dtm1.rds", arrays drm$dimnames[1] """

        processed_filename: str = os.path.join(folder, DOCUMENT_PROCESSED_FILENAME)

        if not os.path.isfile(processed_filename) or force:

            filename = os.path.join(folder, DOCUMENT_DATASET_FILENAME)

            document_index: pd.DataFrame = pd.read_csv(
                filename, compression='zip', header=0, sep=',', quotechar='"', na_filter=False
            )
            document_index.columns = ["filename"]
            document_index.index.name = 'id'

            # Add publication and date
            censured_text: pd.DataFrame = SourceRepository._load_censured_text(folder)
            document_index: pd.DataFrame = pd.merge(
                document_index,
                censured_text[['filename', 'publication', 'date']],
                how='inner',
                left_on='filename',
                right_on='filename',
            )

            # Add pred_bodytext
            meta_data: pd.DataFrame = SourceRepository.load_meta_text_blocks(folder)
            document_index: pd.DataFrame = pd.merge(
                document_index, meta_data, how='inner', left_on='filename', right_index=True
            )

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
            document_index: pd.DataFrame = pd.read_csv(
                processed_filename, compression='zip', header=0, sep=',', quotechar='"', na_filter=False, index_col="id"
            )
            if 'publication_id' not in document_index.columns:
                document_index['publication_id'] = document_index.publication.apply(lambda x: PUBLICATION2ID[x]).astype(
                    np.uint16
                )
            if 'doc_id' in document_index.document_index.columns:
                document_index = document_index.document_index.rename(columns={'doc_id': 'filename'})

        return document_index

    @staticmethod
    def _load_censured_text(folder: str) -> pd.DataFrame:
        """ Load censored corpus data """

        filename: str = os.path.join(folder, CENSORED_CORPUS_FILENAME)

        censured_text: pd.DataFrame = pd.read_csv(
            filename, compression='zip', header=0, sep=',', quotechar='"', na_filter=False
        )
        censured_text.columns = ['id', 'filename', 'publication', 'date']
        censured_text.id -= 1

        censured_text = censured_text[['filename', 'publication', 'date']].drop_duplicates()

        return censured_text

    @staticmethod
    def load_meta_text_blocks(folder: str) -> pd.DataFrame:
        """ Load censored corpus data """

        filename: str = os.path.join(folder, META_TEXTBLOCKS_FILENAME)
        meta_data: pd.DataFrame = pd.read_csv(
            filename, compression='zip', header=0, sep=',', quotechar='"', na_filter=False
        )
        meta_data = meta_data[['id', 'pred_bodytext']].drop_duplicates()
        meta_data.columns = ['filename', 'pred_bodytext']
        meta_data['document_name'] = meta_data.filename
        meta_data = meta_data.set_index('document_name', drop=False).rename_axis('')
        return meta_data

    @staticmethod
    def load_reconstructed_text(folder: str) -> pd.DataFrame:
        filename: str = os.path.join(folder, RECONSTRUCTED_TEXT_CORPUS_FILE)
        if not os.path.isfile(filename):
            corpus: pd.DataFrame = SourceRepository._load_source_dtm(folder)
            vocabulary: pd.DataFrame = SourceRepository._load_source_vocabulary(folder)
            id2token: dict = vocabulary['token'].to_dict()
            reconstructed_text: pd.DataFrame = (corpus.groupby('document_id')).apply(
                lambda x: ' '.join(flatten(x['tf'] * (x['token_id'].apply(lambda y: [id2token[y]]))))
            )
            reconstructed_text.to_csv(filename, compression='zip', header=0, sep=',', quotechar='"')  # type: ignore
        else:
            reconstructed_text: pd.DataFrame = pd.read_csv(filename, compression='zip', header=None, sep=',', quotechar='"')  # type: ignore
            reconstructed_text.columns = ['document_id', 'text']
            reconstructed_text = reconstructed_text.set_index('document_id')

        return reconstructed_text


def plot_document_size_distribution(document_index: pd.DataFrame) -> pd.DataFrame:

    tf: pd.DataFrame = document_index.groupby("tf").size()
    dx = pd.DataFrame({"tf": list(range(0, tf.index.max() + 1))}).set_index("tf")
    tf: pd.DataFrame = dx.join(tf.rename("x"), how="left").fillna(0).astype(np.int)

    ax = tf.plot.bar(figsize=(20, 10), rot=45)

    ticks = ax.xaxis.get_ticklocs()
    ticklabels = [lst.get_text() for lst in ax.xaxis.get_ticklabels()]
    ax.xaxis.set_ticks(ticks[::100])
    ax.xaxis.set_ticklabels(ticklabels[::100])

    return tf


def unique_documents_per_year_and_publication(document_index: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = (
        document_index.groupby(["year", "publication"])
        .agg(document_count=("doc_id", "nunique"))
        .reset_index()
        .set_index(["year", "publication"])
    )
    return df


def mean_tokens_per_year(document_index: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = (
        document_index.groupby(["year", "publication"])
        .agg(term_count=("term_count", "mean"))
        .reset_index()
        .set_index(["year", "publication"])
        .unstack("publication")
    )
    return df


class ExtractDN68:
    @staticmethod
    def extract_to_excel(folder: str, document_index: pd.DataFrame):
        """ Load DN 68 and write reconstructed text to Excel file and zip file"""
        dn68: pd.DataFrame = document_index[
            (document_index.publication == 'DAGENS NYHETER') & (document_index.year == 1968)
        ]
        rt: pd.DataFrame = SourceRepository.load_reconstructed_text(folder)

        dn68_text = rt.merge(dn68, how='inner', left_index=True, right_on='document_id')[
            ['document_id', 'year', 'date', 'term_count', 'text']
        ]
        dn68_text.columns = ['document_id', 'year', 'date', 'term_count', 'text']
        dn68_text.to_excel('dn68_text.xlsx')
        # dn68_text.to_csv('dn68_text.csv', sep='\t')

        with zipfile.ZipFile('dn68.zip', 'w', zipfile.ZIP_DEFLATED) as out:
            i = 0
            for index, row in dn68_text.iterrows():
                i += 1
                filename = 'dn_{}_{}_{}.txt'.format(row['date'], index, 1)
                text = str(row['text'])
                out.writestr(filename, text, zipfile.ZIP_DEFLATED)


def migrate_document_index(folder: str = '/data/westac/textblock_politisk'):

    model_name: str = 'gensim_mallet-lda.topics.100.AB.DN'
    model_path: str = os.path.join(folder, model_name)
    source_name: str = os.path.join(model_path, 'documents.zip')
    target_name: str = os.path.join(model_path, 'documents.NEW.zip')

    source_corpus: SourceCorpus = (
        SourceRepository.load(folder).to_coo_corpus(inplace=True).slice_by_publications([1, 3], inplace=True)
    )

    if 'doc_id' in source_corpus.document_index.columns:
        source_corpus.document_index = source_corpus.document_index.rename(columns={'doc_id': 'filename'})

    document_index: pd.DataFrame = pd.read_csv(source_name, sep='\t', index_col=0)

    document_index.merge(source_corpus.document_index[['filename', 'tf']], on='filename', how='inner').rename(
        columns={'tf': 'n_tokens'}
    ).to_csv(target_name, compression=dict(method='zip', archive_name="document_index.csv"), sep='\t', header=True)
