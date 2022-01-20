# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from dataclasses import dataclass

import matplotlib.pyplot as plt

# %%
import matplotlib.ticker as ticker
import pandas as pd
from IPython.display import display
from ipywidgets import Dropdown, HBox, Output, ToggleButton, VBox

pd.options.mode.chained_assignment = None

# # %matplotlib inline


def load_speech_index(index_path: str, members_path: str) -> pd.DataFrame:
    """Load speech index. Merge with person index (parla. members, ministers, speakers)"""
    speech_index: pd.DataFrame = pd.read_feather(index_path)
    #     members: pd.DataFrame = pd.read_csv(members_path)
    members: pd.DataFrame = pd.read_csv(members_path, delimiter='\t').set_index('id')
    speech_index['protocol_name'] = speech_index.filename.str.split('_').str[0]
    speech_index = speech_index.merge(members, left_on='who', right_index=True, how='inner').fillna('')
    speech_index.loc[speech_index['gender'] == '', 'gender'] = 'unknown'
    return speech_index, members


def plot_pivot(data: pd.DataFrame, kind: str):
    if kind == 'table':
        display(data.round(2))
    elif kind == 'excel':
        display(data.round(2))
        data.to_excel('output.xlsx')
        print('Saved as output.xlsx')
    else:
        data.plot(kind=kind, figsize=(20, 10))
        plt.show()


def compute_statistics(
    *, temporal_key: str, pivot_key: str, pivot_sub_key: str, pivot_value: str, normalize: bool, mode: str
):

    data: pd.DataFrame = SPEECH_INDEX.copy()

    if pivot_value:
        data = data[data[pivot_key] == pivot_value]
        pivot_key = pivot_sub_key

    if temporal_key == 'decade':
        data[temporal_key] = data.year - data.year % 10

    pivot: pd.DataFrame = None

    if mode == 'token':
        pivot = data.groupby([temporal_key, pivot_key]).agg({'n_tokens': sum}).unstack(level=1)
    elif mode == 'speech':
        pivot = pd.DataFrame(data.groupby([temporal_key, pivot_key]).size()).unstack(level=1)
    elif mode == 'speaker':
        pivot = data.groupby([temporal_key, pivot_key]).agg({'who': lambda x: len(set(x))}).unstack(level=1)
    pivot = pivot.fillna(0)

    if normalize:
        pivot = pivot.div(pivot.sum(axis=1), axis=0)
    if hasattr(pivot.columns, 'levels'):
        pivot.columns = pivot.columns.levels[1].tolist()

    return pivot


@dataclass
class CaseOneGUI:

    pivot_key: str = None
    pivot_sub_key: str = None
    pivot_values = Dropdown(description='Pivot', options=[], layout={'width': '160px'})
    mode = Dropdown(
        description='Mode', options=['token', 'speech', 'speaker'], value='token', layout={'width': '160px'}
    )
    period = Dropdown(description='Period', options=['year', 'decade'], value='decade', layout={'width': '160px'})
    kind = Dropdown(
        description='Kind', options=['area', 'line', 'bar', 'table', 'excel'], value='table', layout={'width': '160px'}
    )
    normalize = ToggleButton(description='Normalize', value=True, layout={'width': '160px'})
    output = Output()

    def layout(self):
        return VBox(
            [HBox([VBox([self.pivot_values, self.period]), VBox([self.kind, self.mode]), self.normalize]), self.output]
        )

    def setup(self, pivot_values) -> "CaseOneGUI":
        self.pivot_values.options = pivot_values
        self.pivot_values.observe(self.handler, 'value')
        self.mode.observe(self.handler, 'value')
        self.period.observe(self.handler, 'value')
        self.kind.observe(self.handler, 'value')
        self.normalize.observe(self.handler, 'value')
        return self

    def update(self):

        opts: dict = dict(
            temporal_key=self.period.value,
            pivot_key=self.pivot_key,
            pivot_sub_key=self.pivot_sub_key,
            pivot_value=self.pivot_values.value,
            normalize=self.normalize.value,
            mode=self.mode.value,
        )

        self.output.clear_output()
        with self.output:
            # print(opts)
            data: pd.DataFrame = compute_statistics(**opts)
            plot_pivot(data, kind=self.kind.value)

    def handler(self, *_):
        self.update()


SPEECH_INDEX, MEMBERS = load_speech_index(
    '/data/westac/riksdagen_corpus_data/tagged-speech-corpus.v0.3.0.id.lemma.no-stopwords.lowercase.feather/document_index.feather',
    '/data/westac/riksdagen_corpus_data/tagged-speech-corpus.v0.3.0.id.lemma.no-stopwords.lowercase.feather/person_index.csv',
)

PARTYS = SPEECH_INDEX.party_abbrev.unique().tolist()
GENDERS = SPEECH_INDEX.gender.unique().tolist()

# %%
# MEMBERS =  pd.read_csv('person_index.csv',delimiter='\t').set_index('id')
SPEECH_INDEX.party_abbrev.value_counts()

# %% [markdown]
# ## Words/speeches per party over time
#  - Ministers are encoded as party `gov` and gender `unknown`
#  - Mode: `token` number of tokens, `speech` number of speeches, `speaker`number of unique speakers

# %%
guip = CaseOneGUI(pivot_key='party_abbrev', pivot_sub_key='gender').setup(pivot_values=[''] + PARTYS)
display(guip.layout())
guip.update()

# %% [markdown]
# ## Words/speeches per gender over time

# %%
guig = CaseOneGUI(pivot_key='gender').setup(pivot_values=[''] + GENDERS)
display(guig.layout())
guig.update()

# %% [markdown]
# ## Words per speech over time
#

# %%
pd.set_option('display.max_rows', 1000)
print('NOTE 1995 REMOVED')
SPEECH_INDEX[SPEECH_INDEX.year != 1995].groupby(['year', 'gender']).agg(
    {'n_tokens': lambda x: sum(x) / len(x)}
).unstack(level=1).plot(figsize=(20, 10))

# %% [markdown]
# # Additional quality metrics

# %%
SPEECH_INDEX.groupby(['protocol_name'])['n_tokens'].sum().hist(bins=1000, figsize=(20, 10))

# %%

print(MEMBERS.columns)
# SPEECH_INDEX[SPEECH_INDEX.born != ''].groupby(['born']).size().plot() # TODO FIXA GENOMSNITTSÅLDER
# SPEECH_INDEX.groupby(['year','party_abbrev']).agg({'born': lambda x: next((z for z in x if z != ''),-1)})


def count_empty(x):
    return (x.isna() | x.isnull() | x.eq('') | x.eq('unknown')).sum()


empty_percentages_stats = []

for col in MEMBERS.columns:
    #     print(f"Members without {col} specified: {round((MEMBERS[col].isna().sum()) / len(MEMBERS['born']),4)*100}%")
    empty_percentage = round(count_empty(MEMBERS[col]) / len(MEMBERS['born']) * 100, 3)
    if empty_percentage > 0:
        empty_percentages_stats.append((col, empty_percentage))
#         print(f"Members without {col} specified: {empty_percentage}%")
pd.DataFrame(empty_percentages_stats, columns=['Value', 'Missing (%)']).set_index('Value').plot.bar()

# %%
MEMBERS.groupby(['occupation']).size().sort_values(ascending=False).head(30)
# MEMBERS

# %%
SPEECH_INDEX.groupby(['protocol_name']).size().hist(bins=100, figsize=(20, 10))

# %%

# %%
print(SPEECH_INDEX.columns)
SPEECH_INDEX.head(10)

# %%
SPEECH_INDEX.groupby(['protocol_name']).size().hist(bins=100, figsize=(20, 10))

# %% [markdown]
# # Statistik
#
# 1. Procentuell bar eller area. Normalisera efter antal ledamöter?
# 2. Män och kvinnor (samma bild som ovan)
# 3. Snitt längd av tal pedatar år, kön?
#

# %%
year_group = SPEECH_INDEX[SPEECH_INDEX.year != 1995].drop(columns='document_id').groupby('year')

ax = year_group.size().hist(bins=20, figsize=(20, 10))
ax.set_title('Histogram of number of speeches per year')

# %%
year_group.size().plot(figsize=(20, 10), title='Number of speeches per year')

# %%
year_group['n_tokens'].sum().div(year_group.size()).plot.bar(
    figsize=(20, 10), title='Speech length on average per year', rot=90
)


axes = year_group['n_tokens'].describe().plot(subplots=True, figsize=(20, 20), rot=90)

for ax in axes:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

# %%
year_party = SPEECH_INDEX[SPEECH_INDEX.year != 1995].drop(columns=['document_id']).groupby(['year', 'party_abbrev'])
year_party.describe()
