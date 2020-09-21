import types

class DataFrameTextReader:
    """Text iterator that returns row-wise text documents from a Pandas DataFrame
    """
    def __init__(self, df, **column_filters):
        """
        Parameters
        ----------
        df : DataFrame
            Data frame having one document per row. Text must be in column 'txt' and filename/id in `filename`
        """
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
        self.metadata = self._compile_metadata()
        self.metadict = { x.filename: x for x in (self.metadata or [])}
        self.filenames = [ x.filename for x in self.metadata ]

    def __iter__(self):
        self.iterator = self._create_iterator()
        return self

    def __next__(self):
        return next(self.iterator)

    def _create_iterator(self):
        return ((row['filename'], row['txt']) for _, row in self.df.iterrows())

    def _compile_metadata(self):
        """Returns document metadata as a list of dicts

        Returns
        -------
        List[types.SimpleNamespace]
            File meta data extracted from dataframe
        """
        assert 'filename' in self.df.columns

        df_m = self.df[[ x for x in list(self.df.columns) if x != 'txt' ]]
        metadata = [
            types.SimpleNamespace(**meta) for meta in df_m.to_dict(orient='records')
        ]

        return metadata
