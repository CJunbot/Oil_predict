import keras
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import _WeightedLoss
from copy import deepcopy as dp
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random
import warnings
import keras.backend as K


def custom_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


CFG = {
    'EPOCHS': 50,
    'LEARNING_RATE':1e-2,
    'BATCH_SIZE':256,
    'SEED':42,
    'EARLY_STOPPING_STEPS':10,
    'EARLY_STOP':False,
    'num_features':204,
    'num_preds':1
}

train = pd.read_csv('data/train_after.csv')
test = pd.read_csv('data/test.csv')

categorical_features = ['COMPONENT_ARBITRARY', 'YEAR']
# Inference(실제 진단 환경)에 사용하는 컬럼
test_stage_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR' , 'ANONYMOUS_2', 'AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V', 'V40', 'ZN']

train = train.fillna(0)
test = test.fillna(0)

all_X = train.drop(['ID', 'Y_LABEL'], axis = 1)
all_y = train['Y_LABEL']

test = test.drop(['ID'], axis = 1)

train_X, val_X, train_y, val_y = train_test_split(all_X, all_y, test_size=0.2, random_state=CFG['SEED'], stratify=all_y)


def get_values(value):
    return value.values.reshape(-1, 1)


for col in train_X.columns:
    if col not in categorical_features:
        scaler = StandardScaler()
        train_X[col] = scaler.fit_transform(get_values(train_X[col]))
        val_X[col] = scaler.transform(get_values(val_X[col]))
        if col in test.columns:
            test[col] = scaler.transform(get_values(test[col]))

le = LabelEncoder()
for col in categorical_features:
    train_X[col] = le.fit_transform(train_X[col])
    val_X[col] = le.transform(val_X[col])
    if col in test.columns:
        test[col] = le.transform(test[col])

def get_model_3():
    inp = keras.layers.Input((CFG['num_preds']*CFG['num_features'],))
    x = keras.layers.Reshape((CFG['num_preds']*CFG['num_features'],1))(inp)
    x = keras.layers.Conv1D(32,CFG['num_preds'],strides=CFG['num_preds'], activation='elu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(24,1, activation='elu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(16,1, activation='elu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(4,1, activation='elu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Reshape((CFG['num_features']*4,1))(x)
    x = keras.layers.AveragePooling1D(2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.BatchNormalization()(x)
    out = keras.layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=inp, outputs=out)


def lr_scheduler(epoch):
    if epoch <= CFG['EPOCHS'] * 0.8:
        return CFG['LEARNING_RATE']
    else:
        return CFG['LEARNING_RATE'] * 0.1


def train_NN():
        optimizer = keras.optimizers.Adam(lr=CFG['LEARNING_RATE'], decay=0.00001)
        model = get_model_3()
        callbacks = []
        callbacks.append(keras.callbacks.LearningRateScheduler(lr_scheduler))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[custom_f1])

        model.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=CFG['EPOCHS'],
                  verbose=2, batch_size=CFG['BATCH_SIZE'], callbacks=callbacks)

        pred = model.predict(val_X, batch_size=256)
        return pred

train_NN()