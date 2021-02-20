
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt


class DNNModel(object):
    def __init__(self,
                 x,
                 y,
                 lr,
                 cv=5,
                 batch_size=1000,
                 dropout=0.2,
                 hidden_unit_search_range=None,
                 num_layers=4,
                 class_weight=1 / 10,
                 activation='relu',
                 l1=0.001,
                 l2=0.001,
                 weighted_train=True):
        """
        :param x:
        :param y:
        :param lr:
        :param cv:
        :param batch_size:
        :param dropout:
        :param hidden_unit_search_range:
        :param num_layers:
        :param class_weight:
        :param activation:
        :param l1:
        :param l2:
        :param weighted_train:
        """
        self.x = x
        self.y = y
        self.lr = lr
        self.batch_size = batch_size
        self.n, self.d = self.x.shape
        self.y_binary = np.array([[1, 0] if y else [0, 1] for y in self.y])
        self.num_classes = len(np.unique(self.y))
        self.cv = cv
        self.class_weight = class_weight
        self.dropout = dropout
        if not hidden_unit_search_range:
            self.hidden_unit_search_range = [int(x * self.d) for x in [0.5, 0.8, 1.5, 2.0, 3.0, 4.0]]
        else:
            self.hidden_unit_search_range = hidden_unit_search_range
        self.num_layers = num_layers
        self.activation = activation
        self.l1 = l1
        self.l2 = l2
        self.weighted_train = weighted_train
        self.optimal_layers = [300, 250, 160, 140] # self.grid_search_model()
        self.optimal_model = self.build_model()

    def grid_search_via_keras(self):

        def get_model(num_node_1, num_node_2, num_node_3, num_node_4):
            model = Sequential()
            model.add(Dense(num_node_1,
                            input_shape=self.x[1].shape,
                            activation=self.activation,
                            kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2)))
            model.add(Dropout(self.dropout))
            model.add(Dense(num_node_2,
                            activation=self.activation,
                            kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2)))
            model.add(Dropout(self.dropout))
            model.add(Dense(num_node_3,
                            activation=self.activation,
                            kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2)))
            model.add(Dropout(self.dropout))
            model.add(Dense(num_node_4,
                            activation=self.activation,
                            kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2)))
            model.add(Dense(self.num_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=[AUC()])
            return model
        m = KerasClassifier(build_fn=get_model, epochs=20)
        params = dict(num_node_1=self.hidden_unit_search_range,
                      num_node_2=self.hidden_unit_search_range,
                      num_node_3=self.hidden_unit_search_range,
                      num_node_4=self.hidden_unit_search_range)
        grid = GridSearchCV(estimator=m,
                            param_grid=params,
                            cv=5,
                            scoring='roc_auc',
                            return_train_score=True)
        fitted = grid.fit(self.x, self.y_binary)
        return fitted.best_params_.values()

    def build_model(self):
        """
        :return: optimal model
        """
        print('Constructing model with optimal parameters...')
        model = Sequential()
        model.add(Dense(self.optimal_layers[0],
                        input_shape=self.x[1].shape,
                        activation=self.activation,
                        kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2)))
        for i in self.optimal_layers[1:]:
            model.add(Dense(i,
                            activation=self.activation,
                            kernel_regularizer=l1_l2(l1=self.l1, l2=self.l2)))
            model.add(Dropout(self.dropout))
        model.add(Dense(self.num_classes,
                        activation='softmax'))
        schedule = ExponentialDecay(self.lr, decay_steps=10_0000, decay_rate=0.96)
        optimizer = Adam(learning_rate=schedule)
        model.compile(optimizer=optimizer,
                      metrics=[AUC()],
                      loss='categorical_crossentropy')
        print(model.summary())
        return model

    def train_model(self):
        print('Training model with unbalanced data...')
        weight_adjustment = {0: 1 - self.class_weight, 1: self.class_weight}
        callbacks = [EarlyStopping(monitor='val_auc', patience=10)]  # , self.tensorboard_callback('optimal_model_hist')
        num_epochs = self.n // self.batch_size + 10
        train_history = self.optimal_model.fit(self.x, self.y_binary,
                                               shuffle=True,
                                               validation_split=1 / self.cv,
                                               callbacks=callbacks,
                                               batch_size=self.batch_size,
                                               epochs=num_epochs,
                                               class_weight=weight_adjustment)
        history = train_history.history
        n_epochs = len(history['val_auc'])
        plt.plot(range(1, n_epochs + 1), history['val_auc'], 'x-b')
        plt.xlabel('Training Epochs')
        plt.ylabel('Validation AUC')
        plt.show()

        plt.plot(range(1, n_epochs + 1), history['val_loss'], 'x-b', label='val_loss')
        plt.plot(range(1, n_epochs + 1), history['loss'], 'o-r', label='train_loss')
        plt.legend()
        plt.xlabel('Training Epochs')
        plt.ylabel('LOSS')
        plt.show()
        return self.optimal_model, history

    # def tensorboard_callback(self, exp_name):
    #     return TensorBoard(log_dir=exp_name, profile_batch=0, histogram_freq=1)
