import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import re
import numpy as np
from sklearn.multioutput import MultiOutputRegressor

pred1 = pd.read_csv('pred1.csv')['0']

submit = np.zeros(len(pred1))
for i in range(len(submit)):
    if pred1[i] > 0.2:
        submit[i] = str(1)
    else:
        submit[i] = str(0)


rslt = pd.read_csv('sample_submission.csv')
rslt['Y_LABEL'] = submit
rslt.head()
rslt.to_csv('submit.csv', index=False)
