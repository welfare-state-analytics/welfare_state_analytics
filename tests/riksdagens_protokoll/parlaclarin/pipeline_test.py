import pytest
from penelope.pipeline import CorpusPipeline
from penelope.pipeline.config import CorpusConfig
from tqdm.auto import tqdm
from westac.riksdagens_protokoll.parlaclarin import members, pipelines

CONFIG_FILENAME = './tests/test_data/parlaclarin/riksdagens-protokoll.yml'


@pytest.mark.skip("long running")
@pytest.mark.long_running
@pytest.mark.parametrize(
    'corpus_path',
    [
        '/data/riksdagen_corpus_data/annotated',
    ],
)
def test_run_through_entire_corpus(corpus_path: str):

    corpus_config: CorpusConfig = CorpusConfig.load(CONFIG_FILENAME)

    _ = members.ParliamentaryData.load(members.GITHUB_DATA_URL)

    pipeline: CorpusPipeline = pipelines.to_tagged_frame_pipeline(
        source_folder=corpus_path,
        corpus_config=corpus_config,
        checkpoint_filter=None,
        filename_filter=None,
        filename_pattern=None,
    )

    for _ in tqdm(pipeline.resolve(), total=11100):
        pass
