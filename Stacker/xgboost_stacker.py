import xgboost as xgb
import numpy as np

class XGBOOST_STACKER():
    def __init__(self, _subsample, _colsample_bytree, _max_depth, _min_child_weight,
                 _eta, _lambda, _num_round,_nthread, _objective, _booster):
        self.subsample = _subsample
        self.objective = _objective
        self.booster = _booster
        self.colsample_bytree = _colsample_bytree
        self.max_depth = _max_depth
        self.min_child_weight = _min_child_weight
        self.eta = _eta
        self.reg_lambda = _lambda
        self.num_round = _num_round
        self.nthread = _nthread


    def fit(self,X,y):
        dtrain = xgb.DMatrix(X,y)
        params_xgb = {
            "eta": self.eta,
            "objective": self.objective,
            "booster": self.booster,
            "max_depth": int(self.max_depth),
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "lambda": self.reg_lambda,
            "eval_metric": "mae",
            "nthread": self.nthread,
            "silent": 0,
        }

        def logregobj(preds, dtrain):
            labels = dtrain.get_label()
            con = 2
            x = preds - labels
            grad = con * x / (np.abs(x) + con)
            hess = con ** 2 / (np.abs(x) + con) ** 2
            return grad, hess

        def evalerror(preds, dtrain):
            labels = dtrain.get_label()
            return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))

        self.model = xgb.train(params_xgb,
                               dtrain, num_boost_round=self.num_round,
                               obj=logregobj,
                               feval=evalerror)

    def predict(self,X):
        dtest = xgb.DMatrix(X)
        if self.booster == 'gblinear':
            return self.model.predict(dtest)
        return self.model.predict(dtest, ntree_limit = self.num_round)