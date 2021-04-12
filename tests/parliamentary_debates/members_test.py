from penelope.pipeline import CorpusPipeline
from penelope.pipeline.config import CorpusConfig
from westac.parliamentary_debates.members import ParliamentaryMembers, GITHUB_DATA_URL
from westac.parliamentary_debates.pipelines import to_tagged_frame_pipeline

import tqdm

def test_load_members():

    parliament_data = ParliamentaryMembers.load(GITHUB_DATA_URL)

    assert parliament_data is not None


def test_run_through_entire_corpus():

    corpus_config: CorpusConfig = CorpusConfig.load('./tests/test_data/parliamentary-debates.yml')

    parliament_data = ParliamentaryMembers.load(GITHUB_DATA_URL)

    corpus_path = '/data/riksdagen_corpus_data/annotated'

    pipeline: CorpusPipeline = to_tagged_frame_pipeline(
        source_folder=corpus_path,
        corpus_config=corpus_config,
        checkpoint_filter=None,
        filename_filter=None,
        filename_pattern=None,
    )

    for payload in tqdm.tqdm(pipeline.resolve(), total=11100):
        pass
