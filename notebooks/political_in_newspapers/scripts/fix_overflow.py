import os

import gensim
import numpy as np
import pandas as pd


def compile_mallet_document_topics(model, minimum_probability=0.001):
    def document_topics_iter(model, minimum_probability=0.0):
        data_iter = enumerate(model.load_document_topics())
        for document_id, topic_weights in data_iter:
            for (topic_id, weight) in (
                (topic_id, weight) for (topic_id, weight) in topic_weights if weight >= minimum_probability
            ):
                yield (document_id, topic_id, weight)

    data = document_topics_iter(model, minimum_probability)
    df_doc_topics = pd.DataFrame(data, columns=['document_id', 'topic_id', 'weight'])
    df_doc_topics['document_id'] = df_doc_topics.document_id.astype(np.uint32)
    df_doc_topics['topic_id'] = df_doc_topics.topic_id.astype(np.uint16)

    return df_doc_topics


def fix_int32_overflow(data_folder, model_name):

    target_folder = os.path.join(data_folder, model_name)

    model_filename = os.path.join(target_folder, 'model', 'gensim.model')

    ldaMallet = gensim.models.wrappers.LdaMallet.load(model_filename)
    ldaMallet.prefix = '{}/{}/'.format(data_folder, model_name)

    document_index: pd.DataFrame = pd.read_csv(
        os.path.join(target_folder, 'documents.zip'), '\t', header=0, index_col=0, na_filter=False
    )
    document_topic_weights: pd.DataFrame = compile_mallet_document_topics(ldaMallet, minimum_probability=0.001)

    document_topic_weights: pd.DataFrame = document_topic_weights.merge(
        document_index[['publication_id', 'year']], how='inner', left_on='document_id', right_on='document_id'
    )

    target_file: str = os.path.join(target_folder, 'document_topic_weights.zip')
    document_topic_weights.to_csv(target_file, '\t')
    print(' Stored new version of {}.'.format(target_file))


def run():
    corpus_folder = '/home/roger/source/welfare_state_analytics/data/textblock_politisk'
    for n_topics in [50, 100, 200, 400]:
        model_name = 'gensim_mallet-lda.topics.{}.AB.DN'.format(n_topics)
        print('Fixing {}...'.format(model_name))
        fix_int32_overflow(corpus_folder, model_name)

    print('Done!')


if __name__ == "__main__":
    run()
