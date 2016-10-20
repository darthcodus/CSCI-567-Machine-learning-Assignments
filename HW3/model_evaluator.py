class ModelEvaluator(object):
    def __init__(self, model):
        self.model = model

    def mean_squared_error(self, test_features, test_targets):
        meansquarederror = 0
        for i, row in enumerate(test_features):
            predicted = self.model.predict(row)
            actual = test_targets[i]
            meansquarederror += (predicted-actual)**2

        # avgerror /= len(test_df_targets)
        meansquarederror /= len(test_targets)
        # print("Mean error: %f" % avgerror)
        return meansquarederror
