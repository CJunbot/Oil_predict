import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def get_values(value):
    return value.values.reshape(-1, 1)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

train = train.fillna(0)
test = test.fillna(0)

# Adding features
train['acid_element'] = train['S'] + train['H2O']  # 산화에 필요한 물질
train['acid_element_min'] = pd.concat([train['S'],train['H2O']], axis=1).min(axis=1)  # 산화에 필요한 물질
train['acid_element_max'] = pd.concat([train['S'],train['H2O']], axis=1).max(axis=1)  # 산화에 필요한 물질
train['acid_element_mean'] = pd.concat([train['S'],train['H2O']], axis=1).mean(axis=1)  # 산화에 필요한 물질
train['acid_element_std'] = pd.concat([train['S'],train['H2O']], axis=1).std(axis=1)  # 산화에 필요한 물질
train['acid_element_median'] = pd.concat([train['S'],train['H2O']], axis=1).median(axis=1)  # 산화에 필요한 물질
train['acid_element_kurt'] = pd.concat([train['S'],train['H2O']], axis=1).kurt(axis=1)  # 산화에 필요한 물질

train['acid_number'] = train['FNOX'] + train['FSO4'] + train['FOXID']  # total acid number
train['acid_number_min'] = pd.concat([train['FNOX'], train['FSO4'], train['FOXID']], axis=1).min(axis=1)  # total acid number
train['acid_number_max'] = pd.concat([train['FNOX'], train['FSO4'], train['FOXID']], axis=1).max(axis=1)  # total acid number
train['acid_number_mean'] = pd.concat([train['FNOX'], train['FSO4'], train['FOXID']], axis=1).mean(axis=1)  # total acid number
train['acid_number_std'] = pd.concat([train['FNOX'], train['FSO4'], train['FOXID']], axis=1).std(axis=1)  # total acid number
train['acid_number_median'] = pd.concat([train['FNOX'], train['FSO4'], train['FOXID']], axis=1).median(axis=1)  # total acid number
train['acid_number_kurt'] = pd.concat([train['FNOX'], train['FSO4'], train['FOXID']], axis=1).kurt(axis=1)  # total acid number

train['acid_ratio'] = train['acid_number']/(train['FTBN']+1e-10)  # 산/염기 비율
train['acid_ratio_min'] = pd.concat([train['acid_number'], train['FTBN']], axis=1).min(axis=1)  # 산/염기 비율
train['acid_ratio_max'] = pd.concat([train['acid_number'], train['FTBN']], axis=1).max(axis=1)  # 산/염기 비율
train['acid_ratio_mean'] = pd.concat([train['acid_number'], train['FTBN']], axis=1).mean(axis=1)  # 산/염기 비율
train['acid_ratio_std'] = pd.concat([train['acid_number'], train['FTBN']], axis=1).std(axis=1)  # 산/염기 비율
train['acid_ratio_median'] = pd.concat([train['acid_number'], train['FTBN']], axis=1).median(axis=1)  # 산/염기 비율
train['acid_ratio_kurt'] = pd.concat([train['acid_number'], train['FTBN']], axis=1).kurt(axis=1)  # 산/염기 비율

train['AL_SI_ratio'] = train['AL']/(train['SI']+1e-10)  # 알루미늄 실리콘 비율: 먼지 유입 혹은 피스톤 구이 파악 가능

train['metal_catalyst'] = train['CU'] + train['MO'] + train['FE'] + train['NI'] + train['CO'] + train['MN']  # 금속 촉매

train['metal_oxide'] = train['MG'] + train['V'] + train['ZN']  # 금속 산화물 촉매

train['cleaning'] = train['CA'] + train['MG'] + train['BA'] + train['NA']  # 엔진 오일 청정제 성분

train['anti_freeze'] = train['K'] + train['B'] + train['CR'] + train['P'] + train['SI']  # 엔진 오일 anti freeze 첨가제 성분

train['foreign body'] = train['FUEL'] + train['H2O']  # 이물질

train['engine_up_wear'] = train['SN'] + train['FE'] + train['CR'] + train['SI'] + train['AL']  # 엔진 상단부 마모

train['dust'] = train['PB'] + train['SN'] + train['FE'] + train['CR'] + train['SI'] + train['AL'] + train['CU']  # 먼지 유입 됌

train['cooling_water'] = train['SI'] + train['AL'] + train['NA'] + train['CU'] + train['B']  # 냉각수 유입/ 소포제 유입

train['anti_wear_agent'] = train['ZN'] + train['P']  # 마모 방지제

train['aluminum_alloy'] = train['SI'] + train['CU'] + train['MG'] + train['MN'] + train['AL']  # 알루미늄 합금 성분

train['special_cast_iron'] = train['SI'] + train['MN'] + train['S'] + train['P'] + train['NI'] + train['CR'] + train['MO'] + train['AL'] + train['TI'] + train['V']  # 특수 주철 성분

train['Silver_bearing'] = train['CU'] + train['AG']  # 은 합금 베어링 성분

train['metal'] = train['BA'] + train['AL'] + train['AG'] + train['BE'] + train['CA'] + train['CD'] + train['CO'] + \
                 train['CR'] + train['CU'] + train['FE'] + train['K'] + train['LI'] + train['MG'] + train['MN'] + \
                 train['MO'] + train['NA'] + train['NI'] + train['PB'] + train['SN'] + train['TI'] + train['V'] + train['ZN']  # 금속

train['metalloid'] = train['B'] + train['SB'] + train['SI']

train['nonmetal'] = train['P'] + train['S']

train['Particle_acid'] = train['U25'] - train['U4']

train['particle_1'] = train['U4'] - train['U6']  # 입자 크기 4~5um

train['particle_2'] = train['U6'] - train['U14']  # 입자 크기 6~14um

train['particle_3'] = train['U14'] - train['U20']  # 입자 크기 14~20um

train['particle_4'] = train['U4'] - train['U50']  # 피스톤 간극이 40~60um, 입자 크기 50~4um는 통과 가능

train['particle_5'] = train['U4'] - train['U20']  # 피스톤 링 간극이 20~50um, 입자 크기 20~4um는 통과 가능

train['particle_6'] = train['U4'] - train['U100']  # 피스톤 링 간극이 30~130um, 입자 크기 100~4um는 통과 가능

train.to_csv('../data/train_after.csv')
