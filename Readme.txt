# Author: Bangxi Xiao
# Contact: bangxi_xiao@brown.edu, bangxi@sas.upenn.edu, rchopin@outlook.com
# Date: 2021/02/16

This file keeps record of the package requirements and the basic info of each file.

1. Package Requirements (Environment Specifications):
	1) tensorflow-gpu==2.0.0
	2) numpy==1.19.1
	3) xgboost==1.1.1
	4) seaborn==0.10.1
	5) scikit-learn==0.23.2
	6) pandas==1.0.5
	7) matplotlib==3.2.2

2. File Specifications:
	1) DNN_model.py: this file implements a deep neural network based on tensorflow, with grid search cross-validation method to find the optimal number of hidden units in each hidden layer, the model was regularized by L1_L2 regularization approach and between the layers we added dropout to prevent overfitting. The model training part involves early stopping mechanism, adam algorithm with manually-set learning rate decay and the final training result can be directly observed via TensorBoard. Also, it uses weight adjustments to deal with the highly-unbalanced data problem.
	2) XGB_model.py: this file implements a gradient boosting model via xgboost module, with step-wise grid search approach to find the optimal hyper parameters. Also, we trained multiple models to form an ensemble model: we eventually developed a voting method to justify the final predictions because the original data was so unbalanced that we have to make full use of the negative as well as the positive samples. In case of the great complexity of large-scaled grid search, we used step-wise grid search, that is, to deal with the hyper parameters one by one instead of considering all the possible combinations - this could be very time-consuming and computational unefficient.
	3) model_main.py: this file includes basic questions raised in the E-mail and the latter codes implements both the DNN model and the XGB model.
	4) data_processing_tools.py: this file includes some basic but useful data cleaning tools
	5) data_cleaning.py: feature selection, feature generation, feature format transformation and empty value processing, finally stored the cleaned dataset into local.
	6) EDA.ipynb: EDA of the data, questions raised in the E-mail

3. Outlook: 
	I tried to build a deep model with tensorflow to do the task, but due to the time and my personal computational resources limit, the grid search of deep model's parameters was so slow that I had to give up on this. However, the model file was kept in the project file with name "DNN_model.py". The main problem we tackled in this challenge was straight-forward - the unbalanced sample problem, there are more that I can do if having more time: generating more features from original ones, consider the time series of user transactions (the pattern of normal transactions), and so on.

4. Guide to run the code:
	1) If you want to start from scratch, put "transactions.txt" file with "data_cleaning.py" in the same file and run "data_cleaning.py" thoroughly, then the cleaned data will be generated locally with name "finalized_data.csv". In file "model_main.py", comment all the codes from "# Start from local" to "# End of execution if you 'Start from local' ". Then, run file "model_main.py" and train the ensembled xgb model and you can see all the performance output regarding the model.
	2) If you want to start from loading the local trained model, prepare the test data and make it run in the "data_cleaning.py" file to guarantee that it will have the same shape with training data I used. Then, in file "model_main.py", comment all the codes from "# Train from scratch" to "# End of execution if you 'Train from scratch'". Make sure you replace "YOUR_OWN_TEST_FILE.csv" with the one generated from "data_cleaning.py" script. After that, execute all the codes in the file and you will have the test results saved to local.