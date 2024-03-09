import numpy as np
import pandas as pd
import joblib


class MyModel:
    def __init__(self, cat=None, xgb=None, lgb=None, RFC=None):
        self.cat = cat
        self.xgb = xgb
        self.lgb = lgb
        self.RFC = RFC

    def predict(self, test_x, threshold=0.95):
        test_output_df = pd.DataFrame(columns=['lgb', 'xgb', 'cat'], index=range(test_x.shape[0]))
        test_output_df = test_output_df.fillna(0)
        for i in range(10):
            test_output_df['cat'] += self.cat[i].predict_proba(test_x)[:, 1] / 10
            test_output_df['xgb'] += self.xgb[i].predict_proba(test_x)[:, 1] / 10
            test_output_df['lgb'] += self.lgb[i].predict_proba(test_x)[:, 1] / 10
        pred = self.RFC.predict_proba(test_output_df)[:, 1]
        final_pred = np.array([1 if x >= threshold else 0 for x in pred])
        return final_pred

    def load(self, path):
        model = joblib.load(path)
        return model
