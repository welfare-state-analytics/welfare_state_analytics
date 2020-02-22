import os
import textacy

import text_analytic_tools.utility as utility
import text_analytic_tools.common.textacy_utility as textacy_utility
import text_analytic_tools.common.text_corpus as text_corpus

logger = utility.getLogger('corpus_text_analysis')

def load_or_create(
    source_path,
    language,
    container=None,
    document_index=None,  # data_frame or lambda corpus: corpus_index
    merge_entities=False,
    overwrite=False,
    binary_format=True,
    use_compression=True,
    disabled_pipes=None,
    domain=None,
    tick=utility.noop
):
    assert domain is not None, "Is now mandatory"
    tick = tick or utility.noop
    container = container or textacy_utility.CorpusContainer.container()
    for key in container.__dict__:
        container.__dict__[key] = None

    nlp_args = { 'disable': disabled_pipes or [] }

    store_extension = 'bin' if binary_format else 'pkl'
    store_compression = 'bz2' if use_compression else ''

    container.source_path = source_path
    container.language = language
    container.textacy_corpus = None
    container.prepped_source_path = utility.path_add_suffix(source_path, '_preprocessed')

    if not os.path.isfile(container.prepped_source_path):
        textacy_utility.preprocess_text(container.source_path, container.prepped_source_path, tick=tick)

    container.textacy_corpus_path = textacy_utility.generate_corpus_filename(
        container.prepped_source_path,
        container.language,
        nlp_args=nlp_args,
        extension=store_extension,
        compression=store_compression
    )

    container.nlp = textacy_utility.setup_nlp_language_model(container.language, **nlp_args)

    if overwrite or not os.path.isfile(container.textacy_corpus_path):

        logger.info('Computing new corpus ' + container.textacy_corpus_path + '...')

        reader = text_corpus.CompressedFileReader(container.prepped_source_path)
        stream = domain.get_document_stream(reader, container.language, document_index=document_index)

        logger.info('Stream created...')

        tick(0, len(reader.filenames))

        logger.info('Creating corpus (this might take some time)...')
        container.textacy_corpus = textacy_utility.create_textacy_corpus(stream, container.nlp, tick)

        logger.info('Storing corpus (this might take some time)...')
        textacy_utility.save_corpus(container.textacy_corpus, container.textacy_corpus_path)

        tick(0)

    else:
        tick(1, 2)
        logger.info('...reading corpus (this might take several minutes)...')
        container.textacy_corpus = textacy_utility.load_corpus(container.textacy_corpus_path, container.nlp)

    if merge_entities:
        merge_named_entities(container)

    tick(0)
    logger.info('Done!')

    return container

def merge_named_entities(container):
    logger.info('Working: Merging named entities...')
    try:
        for doc in container.textacy_corpus:
            named_entities = textacy.extract.entities(doc)
            textacy.spacier.utils.merge_spans(named_entities, doc)
    except Exception as ex:
        logger.error(ex)
        logger.info('NER merge failed')

