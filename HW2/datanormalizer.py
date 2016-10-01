import pandas

class Normalizer(object):
    def __init__(self):
        pass

    def get_mean(self):
        raise NotImplementedError()

    def get_std(self):
        raise NotImplementedError()

    def get_normalized_data(self):
        raise NotImplementedError()

class DataFrameStdNormalizer(Normalizer):
    def __init__(self, df):
        assert type(df) is pandas.core.frame.DataFrame
        self.mean_cols = df.mean(0)
        self.std_cols = df.std(0) # default ddof = 1 (normalized with N-1, unbiased estimator of std)

    def get_mean(self):
        return self.mean_cols

    def get_std(self):
        return self.std_cols

    def get_normalized_data(self, df):
        if type(df) is not pandas.core.frame.DataFrame:
            df_new = []
            for i in range(0,len(df)):
                df_new.append((df[i]-self.means_col[i])/self.std_devs_col[i])
        else:
            return (df - self.mean_cols) / (self.std_cols)
            #df_new = df.copy()
            #for i in range(0, len(df.iloc[0])):
            #    df_new.iloc[:, i] = df.iloc[:, i].apply(lambda x: (x-self.means_col[i])/self.std_devs_col[i])
        return df_new
