import types

class DataFrameTextReader:

    def __init__(self, df, **column_filters):

        assert 'txt' in df.columns

        self.df = df

        if not 'filename' in self.df.columns:
            self.df['filename'] = self.df.index.astype(str)

        for column, value in column_filters.items():
            assert column in self.df.columns, column + ' is missing'
            if isinstance(value, tuple):
                assert len(value) == 2
                self.df = self.df[self.df[column].between(*value)]
            elif isinstance(value, list):
                self.df = self.df[self.df[column].isin(value)]
            else:
                self.df = self.df[self.df[column] == value]

        if len(self.df[self.df.txt.isna()]) > 0:
            print('Warn: {} n/a rows encountered'.format(len(self.df[self.df.txt.isna()])))
            self.df = self.df.dropna()

        self.iterator = None
        self.metadata = self.compile_metadata()
        self.metadict = { x.filename: x for x in (self.metadata or [])}
        self.filenames = [ x.filename for x in self.metadata ]

    def __iter__(self):

        self.iterator = None
        return self

    def __next__(self):

        if self.iterator is None:
            self.iterator = self.get_iterator()

        return next(self.iterator)

    def get_iterator(self):
        return ((row['filename'], row['txt']) for _, row in self.df.iterrows())

    def compile_metadata(self):
        assert 'filename' in self.df.columns
        df_m = self.df[[ x for x in list(self.df.columns) if x != 'txt' ]]
        metadata = [
            types.SimpleNamespace(**meta) for meta in df_m.to_dict(orient='records')
        ]
        return metadata