import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
import numpy as np
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
import keras
from keras import regularizers
import tensorflow as tf
from keras.losses import binary_crossentropy
import gc
import scipy.special
from tqdm import *
from scipy.stats import norm, rankdata
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings(action='ignore')

CFG = {
    'EPOCHS': 10000,
    'LEARNING_RATE':1e-2,
    'BATCH_SIZE':256,
    'SEED':42,
    'EARLY_STOPPING_STEPS':10,
    'EARLY_STOP':False,
    'num_features':204,
    'num_preds':1
}

gamma = 2.0
alpha = .25
epsilon = K.epsilon()

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


def focal_loss(y_true, y_pred):
    pt_1 = y_pred * y_true
    pt_1 = K.clip(pt_1, epsilon, 1 - epsilon)
    CE_1 = -K.log(pt_1)
    FL_1 = alpha * K.pow(1 - pt_1, gamma) * CE_1

    pt_0 = (1 - y_pred) * (1 - y_true)
    pt_0 = K.clip(pt_0, epsilon, 1 - epsilon)
    CE_0 = -K.log(pt_0)
    FL_0 = (1 - alpha) * K.pow(1 - pt_0, gamma) * CE_0

    loss = K.sum(FL_1, axis=1) + K.sum(FL_0, axis=1)
    return loss


def knowledge_distillation_loss_withFL(y_true, y_pred, beta=0.1):
    # Extract the groundtruth from dataset and the prediction from teacher model
    y_true, y_pred_teacher = y_true[:, :1], y_true[:, 1:]

    # Extract the prediction from student model
    y_pred, y_pred_stu = y_pred[:, :1], y_pred[:, 1:]

    loss = beta * focal_loss(y_true, y_pred) + (1 - beta) * binary_crossentropy(y_pred_teacher, y_pred_stu)

    return loss


def knowledge_distillation_loss_withBE(y_true, y_pred, beta=0.1):
    # Extract the groundtruth from dataset and the prediction from teacher model
    y_true, y_pred_teacher = y_true[:, :1], y_true[:, 1:]

    # Extract the prediction from student model
    y_pred, y_pred_stu = y_pred[:, :1], y_pred[:, 1:]

    loss = beta * binary_crossentropy(y_true, y_pred) + (1 - beta) * binary_crossentropy(y_pred_teacher, y_pred_stu)

    return loss


def get_model():
    inp = keras.layers.Input(18)
    x = keras.layers.Dense(10, activation='relu')(inp)
    x = keras.layers.Flatten()(x)
    out = keras.layers.Dense(2, activation='sigmoid')(x)
    return keras.Model(inputs=inp, outputs=out)


def get_model_3():
    inp = keras.layers.Input(18)
    x = keras.layers.Reshape((18,1))(inp)
    x = keras.layers.Conv1D(14,CFG['num_preds'],strides=CFG['num_preds'], activation='elu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(10,1, activation='elu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(6,1, activation='elu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(4,1, activation='elu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Reshape((18*4,1))(x)
    x = keras.layers.AveragePooling1D(2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.BatchNormalization()(x)
    out = keras.layers.Dense(2, activation='sigmoid')(x)
    return keras.Model(inputs=inp, outputs=out)


train = pd.read_csv('data/train_after.csv')
test = pd.read_csv('data/test.csv')
train_knowledge = pd.read_csv('data/teacher.csv')

categorical_features = ['COMPONENT_ARBITRARY', 'YEAR']
# Inference(실제 진단 환경)에 사용하는 컬럼
test_stage_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR' , 'ANONYMOUS_2', 'AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V', 'V40', 'ZN']

train = train.fillna(0)
test = test.fillna(0)

y = train['Y_LABEL']
x = train.drop(['ID', 'Y_LABEL'], axis=1)
x = x[test_stage_features]
test = test.drop(['ID'], axis=1)

y_knowledge = train_knowledge['target']

train_X, val_X, train_y, val_y, y_knowledge_train, y_knowledge_valid = train_test_split(x, y, y_knowledge, test_size=0.2, random_state=42, stratify=y)
#train_y = np.vstack((train_y, y_knowledge_train)).T
#val_y = np.vstack((val_y, y_knowledge_valid)).T


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


model = get_model_3()
model.compile(loss=knowledge_distillation_loss_withBE, optimizer='adam', metrics=[custom_f1])

checkpoint = ModelCheckpoint('student_model_BE.h5', monitor='val_auc_2', verbose=1,
                             save_best_only=True, mode='max', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_auc_2', factor=0.5, patience=4,
                                   verbose=1, mode='max', epsilon=0.0001)
early = EarlyStopping(monitor="val_auc_2",
                      mode="max",
                      patience=9)
callbacks_list = [checkpoint, reduceLROnPlat, early]

history = model.fit(train_X,train_y,
                    epochs=CFG['EPOCHS'],
                    batch_size = CFG['BATCH_SIZE'],
                    validation_data=(val_X, val_y))


pred = train_NN()
submit = pd.read_csv('data/sample_submission.csv')
submit['Y_LABEL'] = pred
submit.head()
submit.to_csv('data/submit.csv', index=False)
