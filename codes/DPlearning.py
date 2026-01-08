# 深度学习
import pickle

import keras_tuner
import kerastuner
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras import layers, models
import pandas as pd
from tensorflow import keras
from keras import regularizers
from tensorflow.keras import callbacks
from keras_tuner import RandomSearch

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pickle
from typing import Union, Tuple, Optional
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib import pyplot as plt

class DPlearning:
    def __init__(self, num_layers: int = 5, batch_size: int = 15, unit: int = 64, learning_rate: float = 0.0001,
                 classification: bool = True):
        """
        @param num_layers: number of layer,>0
        @param classification: True for classification task；False for regression task
        """
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.unit = unit
        self.classfication = classification
        self.model = None

    def Train(self, x_train, y_train, x_test, y_test, best_model_save_path: str, epoch: int = 50) -> None:
        """
        @param best_model_save_path: the path of the best models, the file must be ended with .keras
        @param x_train:
        @param y_train:
        @param x_test:
        @param y_test:
        """

        self.__num_features = x_train.shape[1]
        if self.model is None:
            self.__CreateModel()

        # region ModelCheckpoint
        checkpoint = callbacks.ModelCheckpoint(
            filepath=best_model_save_path,
            monitor='val_accuracy' if self.classfication else "val_loss",
            save_best_only=True,  # only save the best model
            mode='max' if self.classfication else "min",
            verbose=1,  # output log
            save_weights_only=True  # if not true the model will not be automatically saved
        )
        # endregion

        # History
        history = self.model.fit(x_train, y_train,
                                 epochs=epoch,
                                 batch_size=self.batch_size,
                                 validation_split=0.1,
                                 verbose=1, callbacks=[checkpoint])

        if self.classfication:
            test_accuracy = history.history['val_accuracy']
            train_accuracy = history.history['accuracy']
            test_loss = history.history['val_loss']
            train_loss = history.history['loss']
            DPlearning.__Draw_Val_Accuracy(train_accuracy, test_accuracy, train_loss, test_loss)
        else:
            test_loss = history.history['val_loss']
            train_loss = history.history['loss']
            test_mae = history.history['val_mae']
            train_mae = history.history['mae']
            DPlearning.__Draw_Val_loss(train_loss, test_loss, train_mae, test_mae)

        self.model = self.Load_Kears_model(best_model_save_path)
        loss, accuracy = self.model.evaluate(x_test, y_test)
        if self.classfication:
            print(f'accuracy for testing set: {accuracy:.4f}')
        else:
            print(f'mae for testing set: {accuracy:.4f}')
        self.Evaluate(x_test, y_test)

    def Predict(self, x, data: pd.DataFrame = None) -> list[float]:
        prediction = self.model.predict(x)
        y_p = [p[0] for p in prediction]
        return y_p

    def Parma_search(self, x_train, y_train, x_test, y_test, directory: str,
                     max_trials: int = 5, executions_per_trial: int = 3, epochs: int = 30) -> None:

        def build_model(hp):
            model = keras.Sequential()
            model.add(layers.Dense(units=hp.Int('units', min_value=8, max_value=128, step=8),
                                   activation='relu', input_shape=(x_train.shape[1],)))
            model.add(layers.Dense(8, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid' if self.classfication else 'linear'))

            if self.classfication:
                model.compile(
                    optimizer=keras.optimizers.Adam(
                        hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='LOG')),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
            else:
                model.compile(
                    optimizer=keras.optimizers.Adam(
                        hp.Float('learning_rate', min_value=1e-4, max_value=1e-1)),
                    loss='mean_squared_error',
                    metrics=['mae']
                )
            return model

        obj = "val_accuracy" if self.classfication else "val_mae"
        tuner = RandomSearch(
            build_model,
            objective=obj,
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=directory,
            project_name='model_optimization'
        )
        tuner.search(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        self.model = tuner.get_best_models(num_models=1)[0]
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        try:
            print(f"""
                best number of hidden layers: {best_hps.get('num_layers')}
                units for a layer: {best_hps.get('units_0')}, {best_hps.get('units_1')}, {best_hps.get('units_2')}        
                """)
        except:
            pass
        print(f"best learning rate: {best_hps.get('learning_rate')}")

    def Save_Keras_model(self, filepath: str = None) -> None:
        if filepath is not None:
            tf.keras.models.save_model(self.model, filepath)

    def Load_Kears_model(self, filepath: str) -> None:
        self.model = keras.models.load_model(filepath)
        return self.model

    def Evaluate(self, x_test, y_test) -> None:
        if self.classfication:
            y_p = [1 if p > 0.5 else 0 for p in self.Predict(x_test)]
            print(confusion_matrix(y_test, y_p))
            print(classification_report(y_test, y_p))
        else:
            y_predict = self.model.predict(x_test)
            y_p = [p[0] for p in y_predict]
            print("R2:", r2_score(y_test, y_p))
            import seaborn as sns
            from matplotlib import pyplot as plt
            plt.figure(figsize=(5, 5), dpi=300)
            df = pd.DataFrame({'Experimental Values': y_test, 'Predicted Values': y_p})
            sns.lmplot(x="Experimental Values", y="Predicted Values", data=df, fit_reg=True, scatter_kws={"s": 10},
                       line_kws={"color": "#FAAFBA"})
            plt.grid(False)
            plt.show()

    @staticmethod
    def __Draw_Val_Accuracy(accuracy, val_accuracy, loss, val_loss):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5), dpi=300)
        plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, marker=None, color="#db65f0", label='val_accuracy',
                 linewidth=1)
        plt.plot(range(1, len(accuracy) + 1), accuracy, marker=None, color="#77acee", label='accuracy', linewidth=1)
        plt.plot(range(1, len(val_loss) + 1), val_loss, marker=None, color="#e1b2e9", label='val_loss', linewidth=1)
        plt.plot(range(1, len(loss) + 1), loss, marker=None, color="#9db2cf", label='loss', linewidth=1)
        plt.title('Accuracy/loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy/loss')
        plt.xticks(range(1, len(val_accuracy) + 1, 10))  # 设置 x 轴刻度
        plt.legend()
        # plt.grid()
        plt.show()

    @staticmethod
    def __Draw_Val_loss(loss, val_loss, mae, val_mae):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5), dpi=300)
        plt.plot(range(1, len(val_mae) + 1), val_mae, marker=None, color="#db65f0", label='val_mae',
                 linewidth=1)
        plt.plot(range(1, len(mae) + 1), mae, marker=None, color="#77acee", label='mae', linewidth=1)
        plt.plot(range(1, len(val_loss) + 1), val_loss, marker=None, color="#e1b2e9", label='val_loss', linewidth=1)
        plt.plot(range(1, len(loss) + 1), loss, marker=None, color="#9db2cf", label='loss', linewidth=1)
        plt.title('loss/mae vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('loss/mae')
        plt.xticks(range(1, len(val_loss) + 1, 10))
        plt.legend()
        # plt.grid()
        plt.show()

    def __CreateModel(self):
        self.model = models.Sequential([layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                                                     input_shape=(self.__num_features,)), ])
        for _ in range(self.num_layers):
            self.model.add(layers.Dense(self.unit, activation='relu', kernel_regularizer=regularizers.l2(0.01)), )
        self.model.add(layers.Dense(int(self.unit / 2), activation='relu', kernel_regularizer=regularizers.l2(0.01)), )
        if self.classfication:
            self.model.add(layers.Dense(1, activation='sigmoid'), )
        else:
            self.model.add(layers.Dense(1, activation='linear'), )
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = 'binary_crossentropy' if self.classfication else 'mean_squared_error'
        metric = "accuracy" if self.classfication else "mae"
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])