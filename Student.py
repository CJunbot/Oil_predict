import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random
import warnings
warnings.filterwarnings(action='ignore')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using: {device}')

CFG = {
    'EPOCHS': 50,
    'LEARNING_RATE': 1e-2,
    'BATCH_SIZE': 256,
    'SEED': 42
}


class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=18, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.classifier(x)
        return output


student_model = Student()
optimizer = torch.optim.Adam(student_model.parameters(), lr=CFG['LEARNING_RATE'])


def distillation(student_logits, labels, teacher_logits, alpha):
    distillation_loss = nn.BCELoss()(student_logits.float(), teacher_logits.float())
    student_loss = nn.BCELoss()(student_logits, labels.reshape(-1, 1))
    return alpha * student_loss + (1-alpha) * distillation_loss


def distill_loss(output, target, teacher_output, loss_fn=distillation, opt=optimizer):
    loss_b = loss_fn(output, target, teacher_output, alpha=0.1)
    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item()


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
            student_X = torch.Tensor(self.data_X[test_stage_features].iloc[index])
            y = self.data_y.values[index]
            return student_X, y, index
        else:
            if self.data_y is None:
                test_X = torch.Tensor(self.data_X.iloc[index])
                return test_X
            else:
                teacher_X = torch.Tensor(self.data_X.iloc[index])
                y = self.data_y.values[index]
                return teacher_X, y


def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG['SEED'])

train = pd.read_csv('data/train_after.csv')
test = pd.read_csv('data/test.csv')
train_knowledge = pd.read_csv('data/teacher_4cnn.csv')

categorical_features = ['COMPONENT_ARBITRARY', 'YEAR']
# Inference(실제 진단 환경)에 사용하는 컬럼
test_stage_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR', 'ANONYMOUS_2', 'AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V', 'V40', 'ZN']

train = train.fillna(0)
test = test.fillna(0)

y = train['Y_LABEL']
x = train.drop(['ID', 'Y_LABEL'], axis=1)
x = x[test_stage_features]
test = test.drop(['ID'], axis=1)

y_knowledge = train_knowledge['0']

train_X, val_X, train_y, val_y = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


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

def student_train(s_model, optimizer, train_loader, val_loader, scheduler, device):
    s_model.to(device)

    best_score = 0
    best_model = None

    for epoch in range(CFG["EPOCHS"]):
        train_loss = []
        s_model.train()

        for X_s, y, index in tqdm(train_loader):
            X_s = X_s.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()

            output = s_model(X_s)
            teacher_output = torch.tensor(train_knowledge['0'][list(index)].values).reshape(len(index), 1).to(device)

            loss_b = distill_loss(output, y, teacher_output, loss_fn=distillation, opt=optimizer)

            train_loss.append(loss_b)

        val_loss, val_score = validation_student(s_model, val_loader, distill_loss, device)
        print(
            f'Epoch [{epoch}], Train Loss : [{np.mean(train_loss) :.5f}] Val Loss : [{np.mean(val_loss) :.5f}] Val F1 Score : [{val_score:.5f}]')

        if scheduler is not None:
            scheduler.step(val_score)

        if best_score < val_score:
            best_model = s_model
            best_score = val_score

    return best_model


def validation_student(s_model, val_loader, criterion, device):
    s_model.eval()

    val_loss = []
    pred_labels = []
    true_labels = []
    threshold = 0.35

    with torch.no_grad():
        for X_s, y, index in tqdm(val_loader):
            X_s = X_s.float().to(device)
            y = y.float().to(device)

            model_pred = s_model(X_s)
            teacher_output = torch.tensor(train_knowledge['0'][list(index)].values).reshape(len(index), 1).to(device)

            loss_b = distill_loss(model_pred, y, teacher_output, loss_fn=distillation, opt=None)
            val_loss.append(loss_b)

            model_pred = model_pred.squeeze(1).to('cpu')
            pred_labels += model_pred.tolist()
            true_labels += y.tolist()

        pred_labels = np.where(np.array(pred_labels) > threshold, 1, 0)
        val_f1 = competition_metric(true_labels, pred_labels)
    return val_loss, val_f1


train_dataset = CustomDataset(train_X, train_y, True)
val_dataset = CustomDataset(val_X, val_y, True)

train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, threshold_mode='abs', min_lr=1e-8, verbose=True)
best_student_model = student_train(student_model, optimizer, train_loader, val_loader, scheduler, device)


# @@@@@@@@@@@@@@@@@@@@ 추론 단계 @@@@@@@@@@@@@@@@@@@@
def choose_threshold(model, val_loader, device):
    model.to(device)
    model.eval()

    thresholds = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    pred_labels = []
    true_labels = []

    best_score = 0
    best_thr = None
    with torch.no_grad():
        for x_s, y, index in tqdm(iter(val_loader)):
            x_s = x_s.float().to(device)
            y = y.float().to(device)

            model_pred = model(x_s)

            model_pred = model_pred.squeeze(1).to('cpu')
            pred_labels += model_pred.tolist()
            true_labels += y.tolist()

        for threshold in thresholds:
            pred_labels_thr = np.where(np.array(pred_labels) > threshold, 1, 0)
            score_thr = competition_metric(true_labels, pred_labels_thr)
            if best_score < score_thr:
                best_score = score_thr
                best_thr = threshold
    return best_thr, best_score

best_threshold, best_score = choose_threshold(best_student_model, val_loader, device)
print(f'Best Threshold : [{best_threshold}], Score : [{best_score:.5f}]')

test_datasets = CustomDataset(test, None, False)
test_loaders = DataLoader(test_datasets, batch_size = CFG['BATCH_SIZE'], shuffle=False)


def inference(model, test_loader, threshold, device):
    model.to(device)
    model.eval()

    test_predict = []
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.float().to(device)
            model_pred = model(x)

            model_pred = model_pred.squeeze(1).to('cpu')
            test_predict += model_pred

    test_predict = np.where(np.array(test_predict) > threshold, 1, 0)
    print('Done.')
    return test_predict

preds = inference(best_student_model, test_loaders, best_threshold, device)
submit = pd.read_csv('data/sample_submission.csv')
submit['Y_LABEL'] = preds
submit.head()
submit.to_csv('data/submit.csv', index=False)
