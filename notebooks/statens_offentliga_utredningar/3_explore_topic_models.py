# # %%

# import os
# import sys
# import importlib

# if os.environ.get('JUPYTER_IMAGE_SPEC', '') == 'westac_lab':
#     root_folder = '/home/jovyan/work/welfare_state_analytics'
# else:
#     root_folder = (lambda x: os.path.join(os.getcwd().split(x)[0], x))('welfare_state_analytics')

# corpus_folder = '/data/westac/sou_kb_labb'

# sys.path = list(set(sys.path + [ root_folder ]))

# from IPython.display import display
# from IPython.core.interactiveshell import InteractiveShell

# InteractiveShell.ast_node_interactivity = "all"

# import notebooks.political_in_newspapers.corpus_data as corpus_data
# import bokeh.plotting

# from notebooks.common import setup_pandas

# %matplotlib inline

# bokeh.plotting.output_notebook()
# setup_pandas()

# # %%
# import notebooks.common.load_topic_model_gui as load_gui
# import text_analytic_tools.text_analysis.topic_model_container as topic_model_container
# _ = importlib.reload(load_gui)

# current_state = lambda: topic_model_container.TopicModelContainer.singleton()

# load_gui.display_gui(corpus_folder, current_state())
# #load_gui.load_model(corpus_folder, current_state(), 'test.4days')
# # %%

# import notebooks.common.topic_wordcloud_gui as wordcloud_gui
# try:
#     wordcloud_gui.display_gui(current_state())
# except Exception as ex:
#     print(ex)

# # %%
# import notebooks.political_in_newspapers.notebook_gui.topic_word_distribution_gui as topic_word_distribution_gui

#     topic_word_distribution_gui.display_gui(current_state())
#     #topic_word_distribution_gui.display_topic_tokens(current_state(), topic_id=0, n_words=100, output_format='Chart')
# except Exception as ex:
#     print(ex)
# %%
# import notebooks.political_in_newspapers.notebook_gui.topic_trends_gui as trends_gui
# from penelope.topic_modelling import topic_weight_over_time

# _ = importlib.reload(topic_weight_over_time)
# _ = importlib.reload(trends_gui)

# trends_gui.display_gui(current_state())
# trends_gui.display_topic_trend(current_state().compiled_data.document_topic_weights, topic_id=0, year=None, year_aggregate='mean', output_format='Table')
# %%
