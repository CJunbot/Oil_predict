import keras
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import _WeightedLoss
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import keras.backend as K
import warnings
from copy import deepcopy
warnings.filterwarnings(action='ignore')


def get_values(value):
    return value.values.reshape(-1, 1)


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

y = train['Y_LABEL']
x = train.drop(['ID', 'Y_LABEL'], axis=1)
test = test.drop(['ID'], axis=1)


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
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = np.zeros(len(train))
    for tr_idx, val_idx in folds.split(x, y):
        x_d = deepcopy(x)
        y_d = deepcopy(y)
        test_d = deepcopy(test)
        x_train, x_val = x_d.iloc[tr_idx], x_d.iloc[val_idx]
        y_train, y_val = y_d.iloc[tr_idx], y_d.iloc[val_idx]
        for col in x_train.columns:
            if col not in categorical_features:
                scaler = StandardScaler()
                x_train[col] = scaler.fit_transform(get_values(x_train[col]))
                x_val[col] = scaler.transform(get_values(x_val[col]))
                if col in test_d.columns:
                    test_d[col] = scaler.transform(get_values(test_d[col]))

        le = LabelEncoder()
        for col in categorical_features:
            x_train[col] = le.fit_transform(x_train[col])
            x_val[col] = le.transform(x_val[col])
            if col in test_d.columns:
                test_d[col] = le.transform(test_d[col])

        optimizer = keras.optimizers.Adam(lr=CFG['LEARNING_RATE'], decay=0.00001)
        model = get_model_3()
        callbacks = []
        callbacks.append(keras.callbacks.LearningRateScheduler(lr_scheduler))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[custom_f1])

        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=CFG['EPOCHS'],
                  verbose=2, batch_size=CFG['BATCH_SIZE'], callbacks=callbacks)

        cash = model.predict(x_val)
        print(cash.shape)
        pred[val_idx] = cash.reshape(-1)
    return pred


pred = train_NN()
pred = pd.DataFrame(pred).to_csv('data/teacher.csv', index=False)
