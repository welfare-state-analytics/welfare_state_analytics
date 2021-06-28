# %%
import pandas as pd
from penelope.co_occurrence import Bundle

bundle = Bundle.load(
    folder='/data/westac/shared/v2_information_w1_VB_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS',
    tag='v2_information_w1_VB_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS'
)

# %%

dir(bundle)
# df = pd.read_feather('/data/westac/shared/v2_information_w1_VB_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS/v2_information_w1_VB_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_co-occurrence.csv.feather')
# %%

# 839K  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_concept_corpus_windows_counts.pickle
# 608K  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_concept_document_windows_counts.npz
# 2.2M  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_concept_vector_data.npz
# 1.9K  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_concept_vectorizer_data.json
# 2.0G  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_concept_vectorizer_data.pickle
# 2.2G  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_co-occurrence.csv.feather
# 1.9K  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_co-occurrence.csv.json
# 2.1G  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_co-occurrence.csv.zip
# 664K  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_co-occurrence.dictionary_tf.pbz2
# 1.6M  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_co-occurrence.dictionary.zip
# 464K  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_co-occurrence.document_index.zip
# 6.5M  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_corpus_windows_counts.pickle
#  49M  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_document_windows_counts.npz
# 594M  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_vector_data.npz
# 1.9K  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_vectorizer_data.json
# 2.0G  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_vectorizer_data.pickle
# 472M  v2_information_w5_NNPM_PASSTHROUGH_TF10_LEMMA_KEEPSTOPS_vocabs_mapping.pickle
# %%
bundle.concept_corpus.data.shape

# %%
# len(bundle.concept_corpus.term_frequency.nonzero()[0].ravel())
bundle.corpus.data.data.nbytes
# %%

def nbytes(corpus):
    try:
        #return corpus.data.data.nbytes + corpus.data.indptr.nbytes + corpus.data.indices.nbytes
        return (corpus.data.nbytes + corpus.indptr.nbytes + corpus.indices.nbytes) / 1024
    except:
        return None

def mask_nonzero_other(self, other) -> None:

    mask = other > 0
    B = bundle.corpus.data
    B = B.multiply(mask)
    B.eliminate_zeros()
    return B


print(f"Corpus: {bundle.corpus.data.shape} size {nbytes(bundle.corpus.data)}")
print(f"Corpus: {bundle.concept_corpus.data.shape} size {nbytes(bundle.concept_corpus.data)}")

mask = bundle.concept_corpus.data > 0

B = bundle.corpus.data
B = B.multiply(mask)
B.eliminate_zeros()
print(nbytes(B))

# %%

B[:,bundle.concept_corpus.data.nonzero()[0]] = 0
print(B.shape)
https://stackoverflow.com/questions/41505416/efficient-way-to-set-elements-to-zero-where-mask-is-true-on-scipy-sparse-matrix
# %%
mask = bundle.concept_corpus.data > 0
B = bundle.corpus.data - bundle.corpus.data.multiply(mask)

# %%
nbytes(mask)
# %%
import numpy as np
A = np.array([1, 2, 9, 3, 5, 1, 6, 4])

def nlargest(a, n_top: int) -> np.ndarray:
    return np.argpartition(a, -n_top)[-n_top:]

print(nlargest(A,3))

indices = np.argpartition(A, -3)[-3:]

print(indices)
print([A[i] for i in indices])
# indices = np.argpartition(-self.term_frequency, n_top-1)[-n_top:]
# np.argpartition(arr, len(arr) - 1)[len(arr) - 1]

# %%

from collections import defaultdict
dd = defaultdict()
dd.default_factory = dd.__len__

pairs = ((i,i+1) for i in range(0,5))
for p in pairs:
    _ = dd[p]
dd
# %%
