
import numpy as np
import pandas as pd
from collections import Generator
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.metrics import accuracy_score, auc, make_scorer, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from data_processing_tools import load_obj, save_obj


class XGBModel(object):
    # Gradient boosting model with grid search hyper parameter configurations implementations
    def __init__(self,
                 x: np.array,
                 y: np.array,
                 learning_rate: float = 0.1,
                 n_estimators: int = 1000,
                 max_depth: int = 5,
                 min_child_weight: float = 1.0,
                 gamma: float = 0.0,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 objective: str = 'binary:logistic',
                 scale_pos_weight: float = 1.0,
                 seed: int = 27,
                 cv: int = 5,
                 early_stopping_iters: int = 50,
                 reg_alpha: float = 1e-5
                 ):
        """
        :param x: training data x
        :param y: training data y
        :param learning_rate: learning rate for xgb model
        :param n_estimators: number of estimators
        :param max_depth: maximum depth of xgb model
        :param min_child_weight: minimum weight in each child leaf
        :param gamma: gamma value of xgb model
        :param subsample: ...
        :param colsample_bytree: ...
        :param objective: ...
        :param scale_pos_weight: proportion of negative samples (the less class)
        :param seed: random seed
        :param cv: k-fold cross-validation
        :param early_stopping_iters: whether applying early stopping mechanism
        :param reg_alpha: ...
        """
        print('Initializing parameters...')
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.scale_pos_weight = scale_pos_weight
        self.seed = seed
        self.cv = cv
        self.esi = early_stopping_iters
        self.reg_alpha = reg_alpha
        print('Parameters initialization finished...')
        print('Launching parameters tuning progress...')
        self.optimal_n_estimators = self.adjust_ne()
        self.optimal_max_depth = self.adjust_md()
        self.optimal_min_child_weight = self.adjust_mcw()
        self.optimal_gamma = self.adjust_g()
        self.optimal_subsample, self.optimal_colsample_bytree = self.adjust_s_cb()
        self.optimal_reg_alpha = self.adjust_ra()
        print('Parameter tuning progress finished, re-fit the model using model_main method...')

    def adjust_ne(self):
        print('Adjusting parameter "n_estimators" via cross-validation...')
        m = XGBClassifier(learning_rate=self.learning_rate,
                          n_estimators=self.n_estimators,
                          max_depth=self.max_depth,
                          min_child_weight=self.min_child_weight,
                          gamma=self.gamma,
                          subsample=self.subsample,
                          colsample_bytree=self.colsample_bytree,
                          reg_alpha=self.reg_alpha,
                          objective=self.objective,
                          scale_pos_weight=self.scale_pos_weight,
                          seed=self.seed)
        params = m.get_xgb_params()
        dtrain = xgb.DMatrix(self.x, self.y)
        cv_result = xgb.cv(params, dtrain,
                           num_boost_round=m.get_params()['n_estimators'],
                           nfold=self.cv,
                           metrics='auc',
                           early_stopping_rounds=self.esi)
        m.set_params(n_estimators=cv_result.shape[0])
        m.fit(self.x, self.y, eval_metric='auc')

        # Predict training set:
        dtrain_predictions = m.predict(self.x)
        dtrain_predprob = m.predict_proba(self.x)[:, 1]

        # Print model report:
        print("\nModel Report")
        print("Accuracy : %.4g" % accuracy_score(self.y, dtrain_predictions))
        print("AUC Score (Train): %f" % roc_auc_score(self.y, dtrain_predprob))

        print('n_estimators=', cv_result.shape[0])
        print('\nDone!')
        return cv_result.shape[0]

    def adjust_md(self):
        print('Adjusting parameter "max_depth" via cross-validation...')
        params = {'max_depth': range(3, 11, 1)}
        xgb_model = XGBClassifier(learning_rate=self.learning_rate,
                                  n_estimators=self.optimal_n_estimators,
                                  max_depth=self.max_depth,
                                  min_child_weight=self.min_child_weight,
                                  gamma=self.gamma,
                                  subsample=self.subsample,
                                  colsample_bytree=self.colsample_bytree,
                                  reg_alpha=self.reg_alpha,
                                  objective=self.objective,
                                  scale_pos_weight=self.scale_pos_weight,
                                  seed=self.seed)
        search = GridSearchCV(estimator=xgb_model,
                              param_grid=params,
                              scoring='roc_auc',
                              cv=self.cv)
        search.fit(self.x, self.y)
        print('Done! Optimal searching score: {}'.format(search.best_score_))
        return search.best_params_['max_depth']

    def adjust_mcw(self):
        print('Adjusting parameter "min_child_weight" via cross-validation...')
        params = {'min_child_weight': range(1, 6, 1)}
        xgb_model = XGBClassifier(learning_rate=self.learning_rate,
                                  n_estimators=self.optimal_n_estimators,
                                  max_depth=self.optimal_max_depth,
                                  min_child_weight=self.min_child_weight,
                                  gamma=self.gamma,
                                  subsample=self.subsample,
                                  colsample_bytree=self.colsample_bytree,
                                  reg_alpha=self.reg_alpha,
                                  objective=self.objective,
                                  scale_pos_weight=self.scale_pos_weight,
                                  seed=self.seed)
        search = GridSearchCV(estimator=xgb_model,
                              param_grid=params,
                              scoring='roc_auc',
                              cv=self.cv)
        search.fit(self.x, self.y)
        print('Done! Optimal searching score: {}'.format(search.best_score_))
        return search.best_params_['min_child_weight']

    def adjust_g(self):
        print('Adjusting parameter "gamma" via cross-validation...')
        params = {'gamma': [x / 10 for x in range(5)]}
        xgb_model = XGBClassifier(learning_rate=self.learning_rate,
                                  n_estimators=self.optimal_n_estimators,
                                  max_depth=self.optimal_max_depth,
                                  min_child_weight=self.optimal_min_child_weight,
                                  gamma=self.gamma,
                                  subsample=self.subsample,
                                  colsample_bytree=self.colsample_bytree,
                                  reg_alpha=self.reg_alpha,
                                  objective=self.objective,
                                  scale_pos_weight=self.scale_pos_weight,
                                  seed=self.seed)
        search = GridSearchCV(estimator=xgb_model,
                              param_grid=params,
                              scoring='roc_auc',
                              iid=False,
                              cv=self.cv)
        search.fit(self.x, self.y)
        print('Optimal searching score: {}'.format(search.best_score_))
        return search.best_params_['gamma']

    def adjust_s_cb(self):
        print('Adjusting parameters "subsample" and "colsample_bytree" via cross-validation...')
        params = {'subsample': [x / 10 for x in range(6, 10)],
                  'colsample_bytree': [x / 10 for x in range(6, 10)]}
        xgb_model = XGBClassifier(learning_rate=self.learning_rate,
                                  n_estimators=self.optimal_n_estimators,
                                  max_depth=self.optimal_max_depth,
                                  min_child_weight=self.optimal_min_child_weight,
                                  gamma=self.optimal_gamma,
                                  subsample=self.subsample,
                                  colsample_bytree=self.colsample_bytree,
                                  reg_alpha=self.reg_alpha,
                                  objective=self.objective,
                                  scale_pos_weight=self.scale_pos_weight,
                                  seed=self.seed)
        search = GridSearchCV(estimator=xgb_model,
                              param_grid=params,
                              scoring='roc_auc',
                              iid=False,
                              cv=self.cv)
        search.fit(self.x, self.y)
        print('Done! Optimal searching score: {}'.format(search.best_score_))
        return search.best_params_['subsample'], search.best_params_['colsample_bytree']

    def adjust_ra(self):
        print('Adjusting parameter "reg_alpha" via cross-validation...')
        params = {'reg_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 100]}
        xgb_model = XGBClassifier(learning_rate=self.learning_rate,
                                  n_estimators=self.optimal_n_estimators,
                                  max_depth=self.optimal_max_depth,
                                  min_child_weight=self.optimal_min_child_weight,
                                  gamma=self.optimal_gamma,
                                  subsample=self.optimal_subsample,
                                  colsample_bytree=self.optimal_colsample_bytree,
                                  objective=self.objective,
                                  reg_alpha=self.reg_alpha,
                                  scale_pos_weight=self.scale_pos_weight,
                                  seed=self.seed)
        search = GridSearchCV(estimator=xgb_model,
                              param_grid=params,
                              scoring='roc_auc',
                              iid=False,
                              cv=self.cv)
        search.fit(self.x, self.y)
        print('Done! Optimal searching score: {}'.format(search.best_score_))
        return search.best_params_['reg_alpha']

    def model_main(self):
        main_model = XGBClassifier(learning_rate=self.learning_rate,
                                   n_estimators=self.optimal_n_estimators,
                                   max_depth=self.optimal_max_depth,
                                   min_child_weight=self.optimal_min_child_weight,
                                   gamma=self.optimal_gamma,
                                   subsample=self.optimal_subsample,
                                   colsample_bytree=self.optimal_colsample_bytree,
                                   objective=self.objective,
                                   reg_alpha=self.optimal_reg_alpha,
                                   scale_pos_weight=self.scale_pos_weight,
                                   seed=self.seed)
        main_model.fit(self.x, self.y)
        return main_model


def train_multiple_model(n: int,
                         dg: Generator,
                         x: np.array,
                         u: bool = True,
                         score_file_name: str = 'model_evaluation_scores') -> None:
    """
    training multiple xgboost models and save them to local
    :param u: whether to apply previous wrongly trained samples to next iteration
    :param score_file_name: model score saving file name
    :param n: number of models to train
    :param dg: data generator (iterator)
    :param x: training data [x, y] with label fetched by "y"
    :return: None

    The training process takes the highly unbalanced data distribution into consideration, let "negative samples" denote
    the "isFraud==1" category and the "positive samples" denote the "isFraud==0" class.
    Split the training data into two parts: negative sample part (n samples) and positive sample part (m samples), known
    that n << m, the value of m // n is approximately 60.

    We intend to train 60 models using data sets:
    (m[x:y] means to use the data from row x to row y, n[:] means to use the whole set)
    [(n[:], m[0:n]), (n[:], m[n:2n]), ..., (n[:], m[59n:60n])]

    Moreover, we keep record of the wrong predicted training data from training iteration i and concat them with the
    training data in the i+1 training iteration. The retrain mechanism can improve the auc score of the model.
    """
    scores = {}
    balanced_data = pd.DataFrame()
    for i in range(10, 10+n):
        print('Executing model {}'.format(i))
        if i == 10 or u:
            balanced_data = pd.concat([next(dg), x[x['y'] == 1]], axis=0)
        else:
            balanced_data = pd.concat([balanced_data, next(dg), x[x['y'] == 1]], axis=0)

        balanced_data = shuffle(balanced_data)

        yt = np.array(balanced_data['y'])
        xt = balanced_data.drop(columns=['y'])

        model_core = XGBModel(xt, yt, scale_pos_weight=sum(yt)/len(yt))
        m = model_core.model_main()

        y_pred = m.predict(xt)

        scores['xgb_model_{}'.format(i)] = roc_auc_score(yt, m.predict_proba(xt)[:, 1])

        if not u:
            wrong_predicted_idx = [i for i, ys in enumerate(zip(yt, y_pred)) if ys[0] != ys[1]]

            balanced_data = balanced_data.iloc[wrong_predicted_idx]
            n_to_sample = min(balanced_data['y'].sum(), len(wrong_predicted_idx) - balanced_data['y'].sum())

            # From wrongly predicted training data sampling 1:1 data to join the next model's training
            balanced_data = pd.concat([balanced_data[balanced_data['y'] == 1].sample(n_to_sample),
                                       balanced_data[balanced_data['y'] == 0].sample(n_to_sample)])

        m.save_model('xgb_model_{}'.format(i))
    save_obj(scores, score_file_name)


class XGBVote(object):
    def __init__(self,
                 model_files: list,
                 score_file: str,
                 amp: bool = False,
                 amp_threshold: float = 0.79,
                 amp_coefficient: float = 0.5,
                 decay_coefficient: float = -0.5):
        """
        A weighted voting model (ensemble model) based on Extreme Gradient Boosting (XGB) algorithm.
        :param model_files: model files
        :param score_file: model score file
        :param amp: whether amplify the weights of each model
        :param amp_threshold: if model's auc score is greater than the threshold, then amplify the weight, else decay
        :param amp_coefficient: the value to amplify
        :param decay_coefficient: the value to decay

        Let auc_i be the roc_auc_score of model i, prob_ij be the predicted probability of sample j from model i,
        the finalized prediction (in probability) of sample j given by the voting model is (assume there are in all
        n testing samples):

        weighted_prob_j = (auc_1 * prob_1j + auc_2 * prob_2j + ... + auc_n * prob_nj) / (auc_1 + auc_2 + ... + auc_n)

        Specifically, when amplification is applied, if the model i's roc_auc score is originally o_auc_i, then the
        adjusted score is calculated as follow:

        if o_auc_i >= amp_threshold:
            auc_i = o_auc_i + amp_coefficient
        else:
            auc_i = o_auc_i + decay_coefficient

        Because the original model scores are so close that one can hardly distinguish between better and worse
        among the models.
        """
        self.files = model_files
        self.amp = amp
        self.scores = load_obj(score_file)
        self.amp_threshold = amp_threshold
        self.amp_coefficient = amp_coefficient
        self.decay_coefficient = decay_coefficient
        print('Evaluating models...')
        self.model_scores = self.model_evaluation()
        self.total_score = sum(self.model_scores.values())
        print('Model evaluated. Models: {} | Total_score: {} | weight_amp: {} | amp_threshold: {} | amp_coefficient: '
              '{} | deay_coefficient: {}.'.format(len(self.files), self.total_score,
                                                  self.amp, self.amp_threshold,
                                                  self.amp_coefficient, self.decay_coefficient))

    def model_evaluation(self):
        """
        Implements the amplification / decay of model score
        :return: the adjusted scores
        """
        for file in self.scores:
            if self.amp:
                self.scores[file] += \
                    self.amp_coefficient if self.scores[file] >= self.amp_threshold else self.decay_coefficient
        return self.scores

    def predict(self, x_test, prob=True, threshold=0.5, front='xgb_models_3/'):
        """
        Predict according to the voting model
        :param front: directory of xgb models
        :param x_test: test set
        :param prob: return probability or predicted labels
        :param threshold: threshold to justify the predicted labels. 1 if prob > 0.5 else 0
        :return: either probabilities or labels
        """
        weighted_prediction = np.zeros(x_test.shape[0])
        for file in self.files:
            x = xgb.Booster()
            x.load_model(front+file)
            w_prob = x.predict(xgb.DMatrix(x_test)) * self.model_scores[file]
            weighted_prediction += w_prob
        if prob:
            return weighted_prediction / self.total_score
        else:
            return np.array([1 if x >= threshold else 0 for x in weighted_prediction/self.total_score])


def grid_search_amp(x_train,
                    y_train,
                    model_files,
                    params_save_file='xgb_model_3/optimal_amp_params',
                    score_file='xgb_models_3/model_evaluation_scores',
                    front='xgb_models_3/',
                    thresholds=None,
                    amp_coe=None,
                    decay_coe=None):
    if thresholds is None:
        thresholds = [0.78, 0.79, 0.80, 0.81, 0.82, 0.83]
    if amp_coe is None:
        amp_coe = [0.3, 0.4, 0.5, 0.6, 0.7]
    if decay_coe is None:
        decay_coe = [-0.2, -0.3, -0.4, -0.5, -0.6]
    scores_params = []
    for threshold in thresholds:
        for ampc in amp_coe:
            for decayc in decay_coe:
                v = XGBVote(model_files=model_files,
                            score_file=score_file,
                            amp=True,
                            amp_threshold=threshold,
                            amp_coefficient=ampc,
                            decay_coefficient=decayc)
                p = v.predict(x_train, prob=True, front=front)
                scores_params.append((threshold, ampc, decayc,
                                      roc_auc_score(y_train, p)))
                print(threshold)
    a, b, c, _ = max(scores_params, key=lambda x: x[-1])
    save = {'optimal_amp_threshold': a, 'optimal_amp_coe': b, 'optimal_decay_coe': c}
    save_obj(save, params_save_file)
    return a, b, c
