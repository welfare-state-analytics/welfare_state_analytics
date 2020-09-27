SHELL = /bin/bash

CORPUS_FOLDER:="./data"

.PHONY: gensim_lda_200

gensim_lda_200 : d=x
	corpus_filename=/data/westac/sou_kb_labb/SOU-KB-labb-corpus-1945-1989.sparv.xml_text_20200923195551.zip
	engine=lda_gensim
	n_topics=200
    max_iters=2000
	model_name=$(engine).topics.$(n_topics).SOU-KB-labb-1945-1989

gensim_lda_200 :
	echo pipenv run python compute_model.py $(model_name) \
		--corpus-filename $(CORPUS_FOLDER)/$(model_name) \
		--n-topics $(n_topics) \
		--engine $(engine) \
		--max-iter $(max_iters) \
		--prefix $(CORPUS_FOLDER)/$(model_name)
