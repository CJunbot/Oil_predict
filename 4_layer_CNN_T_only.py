import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
from copy import deepcopy
warnings.filterwarnings(action='ignore')


def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using: {device}')

# Hyper parameters
CFG = {
    'EPOCHS': 50,
    'LEARNING_RATE':1e-2,
    'BATCH_SIZE':256,
    'SEED':42,
    'EARLY_STOPPING_STEPS':10,
    'EARLY_STOP':False,
    'num_features':204,
    'num_features_test':18,
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

# k fold 학습 시 각각의 best validation score 출력을 위해
val_list = []

def get_values(value):
    return value.values.reshape(-1, 1)


class CustomDataset(Dataset):
    def __init__(self, data_X, data_y, distillation=False):
        super(CustomDataset, self).__init__()
        self.data_X = data_X
        self.data_y = data_y
        self.distillation = distillation

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, index):
        if self.distillation:
            # 지식 증류 학습 시
            teacher_X = torch.Tensor(self.data_X.iloc[index])
            student_X = torch.Tensor(self.data_X[test_stage_features].iloc[index])
            y = self.data_y.values[index]
            return teacher_X, student_X, y
        else:
            if self.data_y is None:
                test_X = torch.Tensor(self.data_X.iloc[index])
                return test_X
            else:
                teacher_X = torch.Tensor(self.data_X.iloc[index])
                y = self.data_y.values[index]
                return teacher_X, y, index


class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        self.conv1d = nn.Conv1d(1, 32, kernel_size=CFG['num_preds'], stride=CFG['num_preds'])
        self.activation = nn.ELU()
        self.batchnorm1d = nn.BatchNorm1d(32)
        self.conv1d2 = nn.Conv1d(32, 24, kernel_size=1)
        self.activation = nn.ELU()
        self.batchnorm1d2 = nn.BatchNorm1d(24)
        self.conv1d3 = nn.Conv1d(24, 16, kernel_size=1)
        self.activation = nn.ELU()
        self.batchnorm1d3 = nn.BatchNorm1d(16)
        self.conv1d4 = nn.Conv1d(16, 4, kernel_size=1)
        self.activation = nn.ELU()
        self.flatten = nn.Flatten()
        self.pool = nn.AvgPool1d(2)
        self.flatten2 = nn.Flatten()
        self.batchnorm1d4 = nn.BatchNorm1d(408)
        self.out = nn.Linear(408,1)
        self.act = nn.Sigmoid()


    def forward(self, x):
        x = x.reshape((x.shape[0], 1, CFG['num_preds']*CFG['num_features']))
        x = self.conv1d(x)
        x = self.activation(x)
        x = self.batchnorm1d(x)
        x = self.conv1d2(x)
        x = self.activation(x)
        x = self.batchnorm1d2(x)
        x = self.conv1d3(x)
        x = self.activation(x)
        x = self.batchnorm1d3(x)
        x = self.conv1d4(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = x.reshape((x.shape[0], 1, CFG['num_features']*4))
        x = self.pool(x)
        x = self.flatten2(x)
        x = self.batchnorm1d4(x)
        x = self.out(x)
        x = self.act(x)
        return x


def validation_teacher(model, val_loader, criterion, device):
    model.eval()

    val_loss = []
    pred_labels = []
    true_labels = []
    threshold = 0.35

    with torch.no_grad():
        for X, y, index in tqdm(val_loader):
            X = X.float().to(device)
            y = y.float().to(device)

            model_pred = model(X.to(device))

            loss = criterion(model_pred, y.reshape(-1, 1))
            val_loss.append(loss.item())

            model_pred = model_pred.squeeze(1).to('cpu')
            pred_labels += model_pred.tolist()
            true_labels += y.tolist()

        pred_labels = np.where(np.array(pred_labels) > threshold, 1, 0)
        val_f1 = competition_metric(true_labels, pred_labels)
    return val_loss, val_f1


def model_train(device, x_origin, y_origin):
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    pred = np.zeros(len(train))

    for tr_idx, val_idx in folds.split(x_origin):
        criterion = nn.BCEWithLogitsLoss().to(device)
        best_score = 0
        best_model = None
        early_stopping_steps = CFG['EARLY_STOPPING_STEPS']
        early_step = 0
        # 모델, 옵티마이저, 스케쥴러 선언
        model = Teacher()
        model.eval()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1,
                                                               threshold_mode='abs', min_lr=1e-8, verbose=True)

        x_d = deepcopy(x_origin)
        y_d = deepcopy(y_origin)
        test_d = deepcopy(test)
        x_train, x_val = x_d.iloc[tr_idx], x_d.iloc[val_idx]
        y_train, y_val = y_d.iloc[tr_idx], y_d.iloc[val_idx]
        # Feature Scaling
        for col in x_train.columns:
            if col not in categorical_features:
                scaler = StandardScaler()
                x_train[col] = scaler.fit_transform(get_values(x_train[col]))
                x_val[col] = scaler.transform(get_values(x_val[col]))
                if col in test_d.columns:
                    test_d[col] = scaler.transform(get_values(test_d[col]))
        # Feature Engineering
        le = LabelEncoder()
        for col in categorical_features:
            x_train[col] = le.fit_transform(x_train[col])
            x_val[col] = le.transform(x_val[col])
            if col in test_d.columns:
                test_d[col] = le.transform(test_d[col])

        train_dataset = CustomDataset(x_train, y_train, False)
        val_dataset = CustomDataset(x_val, y_val, False)
        train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

        for epoch in range(CFG["EPOCHS"]):
            train_loss = []

            model.train()
            for X, y, index in tqdm(train_loader):
                X = X.float().to(device)
                y = y.float().to(device)

                optimizer.zero_grad()

                y_pred = model(X)
                loss = criterion(y_pred, y.reshape(-1, 1))
                loss.backward()

                optimizer.step()

                train_loss.append(loss.item())

            val_loss, val_score = validation_teacher(model, val_loader, criterion, device)
            print(
                f'Epoch [{epoch}], Train Loss : [{np.mean(train_loss) :.5f}] Val Loss : [{np.mean(val_loss) :.5f}] Val F1 Score : [{val_score:.5f}]')

            if scheduler is not None:
                scheduler.step(val_score)

            if best_score < val_score:
                best_model = model
                best_score = val_score

            elif (CFG['EARLY_STOP'] == True):
                early_step += 1
                if (early_step >= early_stopping_steps):
                    break

        val_list.append(best_score)

        with torch.no_grad():
            X = torch.Tensor(np.array(x_val))
            X = X.float().to(device)
            model_pred = model(X.to(device))
            model_pred = model_pred.squeeze(1).to('cpu')
            pred[val_idx] = model_pred.reshape(-1)

    return best_model, pred


teacher_model, pred = model_train(device, x, y)
print(val_list)
pred = pd.DataFrame(pred).to_csv('data/teacher_4cnn.csv', index=False)

# 0.561 -> 여기서 studnet 학습했을때