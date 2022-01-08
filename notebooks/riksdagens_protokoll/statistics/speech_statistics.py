from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from ipywidgets import Dropdown, HBox, Output, ToggleButton, VBox

pd.options.mode.chained_assignment = None

# pylint: disable=no-member, unused-argument
# %matplotlib inline


def load_speech_index(index_path: str, members_path) -> pd.DataFrame:
    """Load speech index. Merge with person index (parla. members, ministers, speakers)"""
    speech_index: pd.DataFrame = pd.read_csv('/data/westac/shared/document_index.csv.xz', sep='\t')
    members: pd.DataFrame = pd.read_json('/data/westac/shared/members.json')
    speech_index['protocol_name'] = speech_index.filename.str.split('_').str[0]
    speech_index = speech_index.merge(members, left_on='who', right_index=True, how='inner').fillna('')
    speech_index.loc[speech_index['gender'] == '', 'gender'] = 'unknown'
    return speech_index


SPEECH_INDEX: pd.DataFrame = load_speech_index(
    '/data/westac/shared/document_index.csv.xz', '/data/westac/shared/members.json'
)
PARTYS = SPEECH_INDEX.party_abbrev.unique().tolist()
GENDERS = SPEECH_INDEX.gender.unique().tolist()


def plot_pivot(data: pd.DataFrame, kind: str):
    if kind == 'table':
        display(data)
    else:
        data.plot(kind=kind, figsize=(20, 10))
        plt.show()


# %% [markdown]
# ## Words per party and gender
#
# Note:
#  - Ministers are encoded as party `gov` and gender `unknown`

# %%


def compute_statistics(temporal_key: str, party: str, normalize: bool):
    data: pd.DataFrame = SPEECH_INDEX.copy()

    if party:
        data = data[data.party_abbrev == party]

    if temporal_key == 'decade':
        data[temporal_key] = data.year - data.year % 10

    pivot = data.groupby([temporal_key, 'gender']).agg({'n_tokens': sum}).unstack(level=1).fillna(0)

    pivot.columns = pivot.columns.levels[1].tolist()
    if normalize:
        category_sums = data.groupby(temporal_key)['n_tokens'].sum()
        pivot = pivot.div(category_sums, axis=0)
    return pivot


@dataclass
class CaseOneGUI:

    partys = Dropdown(description='Party', options=[''] + PARTYS, layout={'width': '150px'})
    period = Dropdown(description='Period', options=['year', 'decade'], value='decade', layout={'width': '180px'})
    kind = Dropdown(
        description='Kind', options=['area', 'line', 'bar', 'table'], value='area', layout={'width': '150px'}
    )
    normalize = ToggleButton(description='Normalize', value=True, layout={'width': '150px'})
    output = Output()

    def layout(self):
        return VBox([HBox([self.partys, self.period, self.normalize, self.kind]), self.output])

    def setup(self) -> "CaseOneGUI":
        self.partys.observe(self.handler, 'value')
        self.period.observe(self.handler, 'value')
        self.kind.observe(self.handler, 'value')
        self.normalize.observe(self.handler, 'value')
        return self

    def update(self):

        data: pd.DataFrame = compute_statistics(self.period.value, self.partys.value, self.normalize.value)

        self.output.clear_output()
        with self.output:
            plot_pivot(data, kind=self.kind.value)

    def handler(self, *_):
        self.update()


display(CaseOneGUI().setup().layout())


# # %% [markdown]
# # ## Words/speeches per gender over time

# # %%
# dg = du.groupby(['year', 'party_abbrev']).size().unstack(level=1)
# dg = dg.div(dg.sum(axis=1), axis=0)
# dg.plot(kind='area', figsize=(15,15))

# # %% [markdown]
# # ## Words per party and gender
# #
# # Note:
# #  - Ministers are encoded as party `gov` and gender `unknown`

# # %% [markdown]
# # ## Words per party and gender
# #
# # Note:
# #  - Ministers are encoded as party `gov` and gender `unknown`

# # %% [markdown]
# # # Statistik
# #
# # 1. Procentuell bar eller area. Normalisera efter antal ledamöter?
# # 2. Män och kvinnor (samma bild som ovan)
# # 3. Snitt längd av tal pedatar år, kön?
# #

# # %%
# di['protocol_name'] = di.filename.str.split('_').str[0]

# # %%
# du = di.merge(members, left_on='who', right_index=True, how='inner').fillna('')
# du.loc[du['gender']=='','gender'] = 'unknown'
# #dy = di.groupby('year').size()
# #du['C'] / dy

# # %%
# dg = du.groupby(['year', 'party_abbrev']).size().unstack(level=1)
# dg = dg.div(dg.sum(axis=1), axis=0)
# dg.plot(kind='area', figsize=(15,15))

# # %%


# # %%
# members

# # %%
# dg = du.groupby(['year', 'gender']).agg({'name': lambda x: len(set(x)) }).unstack(level=1)
# dg = dg.div(dg.sum(axis=1), axis=0)
# dg.plot(kind='area', figsize=(20,20))

# dg = du.groupby(['year', 'gender']).size().unstack(level=1)
# dg = dg.div(dg.sum(axis=1), axis=0)
# dg.plot(kind='area', figsize=(20,20))

# # %%
# dg = du.groupby(['year', 'gender']).agg({'n_tokens': lambda x: len(set(x)) }).unstack(level=1)
# dy = du.groupby(['year']).n_tokens.sum()
# dg = dg.div(dy, axis=0)
# dg.plot(kind='area', figsize=(20,20))

# #warnings.suppress(SettingWithCopyWarning)
# PARTY_OPTIONS = du.party_abbrev.unique().tolist()
# %matplotlib inline

# partys = ipywidgets.Dropdown(description='Party', options=PARTY_OPTIONS)
# output = ipywidgets.Output()

# def display_gender_stats(*_):
#     global du
#     dd = du[du.party_abbrev==partys.value]
#     dd['decade'] = dd.year - dd.year % 10
#     dg = dd.groupby(['decade', 'gender']).agg({'n_tokens': sum}).unstack(level=1).fillna(0)
#     dy = dd.groupby('decade')['n_tokens'].sum()
#     #print(dg)
#     dg.columns = dg.columns.levels[1].tolist()
#     dg = dg.div(dy, axis=0)
#     output.clear_output()
#     with output:
#         #print(dg)
#         dg.plot(kind='bar', figsize=(20,10))
#         plt.show()

# partys.observe(display_gender_stats, 'value')

# display(ipywidgets.VBox([partys, output]))


# # %%
# dg.columns = ['tom', 'ma', 'unknown', 'woman']
# dy = du.groupby(['year']).n_tokens.sum()
# dg['apa'] = dy
# print(dg)
# #.reset_index().set_index('year')
# #dg['n_tokens'] = du.groupby(['year']).n_tokens.sum()
# #dg['n_tokens'] = dy
# #dg = dg.div(dy, axis=0)
# #print(dg)
# #dg.plot(kind='area', figsize=(20,20))

# # %%
# du.head()

# # %%
# 897924.0 +  574630.0 + 10053.0 +   5174.0

# # %%
# du.groupby(['year'])['n_tokens'].sum()

# # %%
