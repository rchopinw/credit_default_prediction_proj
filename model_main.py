import pandas as pd
from XGB_model import train_multiple_model, XGBVote, grid_search_amp
from data_processing_tools import *
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

model_files = ['xgb_model_{}'.format(i) for i in range(10, 70)]

# Train from scratch:
# Loading the cleaned data
data = pd.read_csv('finalized_data.csv', encoding='utf_8_sig')

# defining label and feature set
labels = np.array(data['isFraud'])
x_data = data.drop(columns=['isFraud'])

# Train test splitting
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3, shuffle=True)
X_train['y'] = y_train

# Defining a data generator for multiple training tasks
dg = data_gen(X_train[X_train['y'] == 0], sum(y_train))

# Approach one: ensemble model - training multiple xgboost model and make the inference by voting
train_multiple_model(60, dg, X_train)

# Searching for the optimal amplification coefficient and decay coefficient
optimal_amp_threshold, optimal_amp_coe, optimal_decay_coe, _ = grid_search_amp(X_train.drop(columns=['y']),
                                                                               y_train,
                                                                               model_files)
# End of execution if you "Train from scratch"

# Start from local
# Loading the model from local:
X_test = pd.read_csv('YOUR_OWN_TEST_FILE.csv', encoding='utf_8_sig')
optima = load_obj('xgb_models_3/optimal_amp_params')
optimal_amp_threshold, optimal_amp_coe, optimal_decay_coe = [optima.get(x) for x in
                                                             ['optimal_amp_threshold', 'optimal_amp_coe', 'optimal_decay_coe']]
# End of execution if you "Start from local"

# Either you "Train from scratch" or "Start from local", execute the following:
# Establishing a voting model
vote = XGBVote(model_files=model_files,
               score_file='xgb_models_3/model_evaluation_scores',
               amp=True,
               amp_threshold=optimal_amp_threshold,
               amp_coefficient=optimal_amp_coe,
               decay_coefficient=optimal_decay_coe)

# Making inferences on testing set
y_pred_prob = vote.predict(x_test=X_test, prob=True, front='xgb_models_3/')

# Setting a threshold to make the final 0/1 predictions
t = 0.45
y_pred = [1 if x >= t else 0 for x in y_pred_prob]

# Making some tests on test set
print("Confusion Matrix", confusion_matrix(y_test, y_pred))
cm = pd.DataFrame(confusion_matrix(y_test, y_pred),
                  columns=['isFraud=0 (Predicted)', 'isFraud=1 (Predicted)'],
                  index=['isFraud=0 (True)', 'isFraud=1 (True)'])
print(cm)
cm.to_csv('Model_performance_confusion_matrix.csv', encoding='utf_8_sig', index=True)

print("Overall Accuracy", accuracy_score(y_test, y_pred))

print("Precision Recall F1-Score Support",
      precision_recall_fscore_support(y_test, y_pred))
prfs = pd.DataFrame(precision_recall_fscore_support(y_test, y_pred),
                    index=['precision', 'recall', 'f-score', 'Num_samples'],
                    columns=['isFraud=0', 'isFraud=1'])
print(prfs)
prfs.to_csv('Model_performance_precision_recall_fscore.csv', encoding='utf_8_sig', index=True)

print('Roc_auc score: {}'.format(roc_auc_score(y_test, y_pred_prob)))

draw_roc_curve(y_test, y_pred_prob)

# Approach two: deep model
