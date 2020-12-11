import os

import penelope.corpus.vectorized_corpus as vectorized_corpus
from penelope.common.most_discriminating_terms import compute_most_discriminating_terms

root_folder = "/Users/frno0044/Documents/Kod/welfare_state_analytics"
corpus_folder = os.path.join(root_folder, "output")

v_corpus = (
    vectorized_corpus.VectorizedCorpus.load(tag='SOU_1945-1989_NN+VB+JJ_lemma_L0_+N_+S', folder=corpus_folder)
    .slice_by_n_count(10)
    .slice_by_n_top(500000)
)

df = compute_most_discriminating_terms(
    v_corpus, v_corpus.document_index, top_n_terms=250, max_n_terms=2000, period1=(1945, 1967), period2=(1968, 1989)
)

df.to_excel('sou_mdw_45-67_vs_68-89.xlsx')

df = compute_most_discriminating_terms(
    v_corpus, v_corpus.document_index, top_n_terms=250, max_n_terms=2000, period1=(1945, 1967), period2=(1968, 1989)
)

df.to_excel('sou_mdw_45-67_vs_68-89.xlsx')
