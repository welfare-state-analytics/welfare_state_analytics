import penelope.notebook.word_trends as word_trends
from bokeh.plotting import output_notebook
from IPython.core.display import display
from penelope.notebook.word_trends import main_gui

# %%
from penelope.pipeline import CorpusConfig

# %%
from penelope.pipeline.sparv import pipelines

import __paths__

output_notebook()

gui = main_gui.create_to_dtm_gui(
    corpus_folder=__paths__.data_folder,
    corpus_config="riksdagens-protokoll",
    resources_folder=__paths__.resources_folder,
)
display(gui)

config = CorpusConfig.load(f"{__paths__.resources_folder}/riksdagens-protokoll.yml").folder(__paths__.data_folder)
main_gui.compute_callback(main_gui.LAST_ARGS, config)


config = CorpusConfig.load(f"{__paths__.resources_folder}/riksdagens-protokoll.yml").folder(__paths__.data_folder)
p = pipelines.to_tagged_frame_pipeline(config)

p.exhaust()

assert p.payload.document_index is not None
assert 'year' in p.payload.document_index.columns


corpus = main_gui.LAST_CORPUS

corpus_tag = "MARS"
corpus_folder = "./data/MARS"

trends_data: word_trends.TrendsData = word_trends.TrendsData(
    corpus=corpus,
    corpus_folder=corpus_folder,
    corpus_tag=corpus_tag,
    n_count=25000,
).update()

gui = word_trends.GofTrendsGUI(
    gofs_gui=word_trends.GoFsGUI().setup(),
    trends_gui=word_trends.TrendsGUI().setup(),
)

# display(gui.layout())
# gui.display(trends_data=trends_data)
# %%

