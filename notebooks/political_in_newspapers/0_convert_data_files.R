# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown]
# ### Install R

# %% [markdown]
#
# Install R
#
# ```bash
# % ...
# ```
#
# Install IRKernel
#
# ```bash
# % sudo apt-get install libxml2-dev
# % sudo -i R 
# > install.packages("xml2")
# > install.packages("roxygen2")
# > install.packages("rversions")
# > devtools::install_github("IRkernel/IRkernel")
# > Ctrl-D
#
# ```
#
# ```bash
# $ pipenv shell
# $ R
# > IRkernel::installspec()
# $
# ```

# %% [markdown]
# ### Load corpus file

# %%
drm = readRDS("dtm1.rds")
str(drm)

# %%
document_dataset   = drm$dimnames[1]
vocabulary_dataset = drm$dimnames[2]

coo_dataset        = list()
coo_dataset[[1]]   = drm$i
coo_dataset[[2]]   = drm$j
coo_dataset[[3]]   = drm$v

write.csv(document_dataset, "document_dataset.csv", row.names = FALSE)
write.csv(vocabulary_dataset, "vocabulary_dataset.csv", row.names = FALSE)
write.csv(coo_dataset, "corpus_dataset.csv", row.names = FALSE)


# %%
source_dataframe = readRDS("../../../data/textblock_politisk/text1_utantext.rds")
write.csv(source_dataframe,'../../../data/textblock_politisk/text1_utantext.csv')

# %% [markdown]
# ### Compress files
#
# ```bash
# % zip corpus_dataset.zip corpus_dataset.csv
# % zip document_dataset.zip document_dataset.csv
# % zip vocabulary_dataset.zip vocabulary_dataset.csv
# % zip text1_utantext.zip text1_utantext.csv
# ```
