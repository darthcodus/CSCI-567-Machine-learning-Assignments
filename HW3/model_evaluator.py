class ModelEvaluator(object):
    def __init__(self, model):
        self.model = model

    def mean_squared_error(self, test_df_features_normalized, test_df_targets):
        meansquarederror = 0
        for i, row in enumerate(test_df_features_normalized.values):
            predicted = self.model.predict(row)
            actual = test_df_targets.iloc[i]
            meansquarederror += (predicted-actual)**2

        # avgerror /= len(test_df_targets)
        meansquarederror /= len(test_df_targets)
        # print("Mean error: %f" % avgerror)
        return meansquarederror
