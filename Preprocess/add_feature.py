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

train['acid_number'] = train['FNOX'] + train['FSO4'] + train['FOXID']  # total acid number
train['acid_number_min'] = pd.concat([train['FNOX'], train['FSO4'], train['FOXID']], axis=1).min(axis=1)  # total acid number
train['acid_number_max'] = pd.concat([train['FNOX'], train['FSO4'], train['FOXID']], axis=1).max(axis=1)  # total acid number
train['acid_number_mean'] = pd.concat([train['FNOX'], train['FSO4'], train['FOXID']], axis=1).mean(axis=1)  # total acid number
train['acid_number_std'] = pd.concat([train['FNOX'], train['FSO4'], train['FOXID']], axis=1).std(axis=1)  # total acid number
train['acid_number_median'] = pd.concat([train['FNOX'], train['FSO4'], train['FOXID']], axis=1).median(axis=1)  # total acid number

train['acid_ratio'] = train['acid_number']/(train['FTBN']+1e-10)  # 산/염기 비율
train['acid_ratio_min'] = pd.concat([train['acid_number'], train['FTBN']], axis=1).min(axis=1)  # 산/염기 비율
train['acid_ratio_max'] = pd.concat([train['acid_number'], train['FTBN']], axis=1).max(axis=1)  # 산/염기 비율
train['acid_ratio_mean'] = pd.concat([train['acid_number'], train['FTBN']], axis=1).mean(axis=1)  # 산/염기 비율
train['acid_ratio_std'] = pd.concat([train['acid_number'], train['FTBN']], axis=1).std(axis=1)  # 산/염기 비율
train['acid_ratio_median'] = pd.concat([train['acid_number'], train['FTBN']], axis=1).median(axis=1)  # 산/염기 비율

train['AL_SI_ratio'] = train['AL']/(train['SI']+1e-10)  # 알루미늄 실리콘 비율: 먼지 유입 혹은 피스톤 구이 파악 가능
train['AL_SI_ratio_min'] = pd.concat([train['AL'], train['SI']], axis=1).min(axis=1)  # 알루미늄 실리콘 비율: 먼지 유입 혹은 피스톤 구이 파악 가능
train['AL_SI_ratio_max'] = pd.concat([train['AL'], train['SI']], axis=1).max(axis=1)  # 알루미늄 실리콘 비율: 먼지 유입 혹은 피스톤 구이 파악 가능
train['AL_SI_ratio_mean'] = pd.concat([train['AL'], train['SI']], axis=1).mean(axis=1)  # 알루미늄 실리콘 비율: 먼지 유입 혹은 피스톤 구이 파악 가능
train['AL_SI_ratio_std'] = pd.concat([train['AL'], train['SI']], axis=1).std(axis=1)  # 알루미늄 실리콘 비율: 먼지 유입 혹은 피스톤 구이 파악 가능
train['AL_SI_ratio_median'] = pd.concat([train['AL'], train['SI']], axis=1).median(axis=1)  # 알루미늄 실리콘 비율: 먼지 유입 혹은 피스톤 구이 파악 가능

train['metal_catalyst'] = train['CU'] + train['MO'] + train['FE'] + train['NI'] + train['CO'] + train['MN']  # 금속 촉매
train['metal_catalyst_min'] = pd.concat([train['CU'],train['MO'], train['FE'], train['NI'], train['CO'], train['MN']], axis=1).min(axis=1)  # 금속 촉매
train['metal_catalyst_max'] = pd.concat([train['CU'],train['MO'], train['FE'], train['NI'], train['CO'], train['MN']], axis=1).max(axis=1)  # 금속 촉매
train['metal_catalyst_mean'] = pd.concat([train['CU'],train['MO'], train['FE'], train['NI'], train['CO'], train['MN']], axis=1).mean(axis=1)  # 금속 촉매
train['metal_catalyst_std'] = pd.concat([train['CU'],train['MO'], train['FE'], train['NI'], train['CO'], train['MN']], axis=1).std(axis=1)  # 금속 촉매
train['metal_catalyst_median'] = pd.concat([train['CU'],train['MO'], train['FE'], train['NI'], train['CO'], train['MN']], axis=1).median(axis=1)  # 금속 촉매

train['metal_oxide'] = train['MG'] + train['V'] + train['ZN']  # 금속 산화물 촉매
train['metal_oxide_min'] = pd.concat([train['MG'], train['V'], train['ZN']], axis=1).min(axis=1)  # 금속 산화물 촉매
train['metal_oxide_max'] = pd.concat([train['MG'], train['V'], train['ZN']], axis=1).max(axis=1)  # 금속 산화물 촉매
train['metal_oxide_mean'] = pd.concat([train['MG'], train['V'], train['ZN']], axis=1).mean(axis=1)  # 금속 산화물 촉매
train['metal_oxide_std'] = pd.concat([train['MG'], train['V'], train['ZN']], axis=1).std(axis=1)  # 금속 산화물 촉매
train['metal_oxide_median'] = pd.concat([train['MG'], train['V'], train['ZN']], axis=1).median(axis=1)  # 금속 산화물 촉매

train['cleaning'] = train['CA'] + train['MG'] + train['BA'] + train['NA']  # 엔진 오일 청정제 성분
train['cleaning_min'] = pd.concat([train['CA'], train['MG'], train['BA'], train['NA']], axis=1).min(axis=1)  # 엔진 오일 청정제 성분
train['cleaning_max'] = pd.concat([train['CA'], train['MG'], train['BA'], train['NA']], axis=1).max(axis=1)  # 엔진 오일 청정제 성분
train['cleaning_mean'] = pd.concat([train['CA'], train['MG'], train['BA'], train['NA']], axis=1).mean(axis=1)  # 엔진 오일 청정제 성분
train['cleaning_std'] = pd.concat([train['CA'], train['MG'], train['BA'], train['NA']], axis=1).std(axis=1)  # 엔진 오일 청정제 성분
train['cleaning_median'] = pd.concat([train['CA'], train['MG'], train['BA'], train['NA']], axis=1).median(axis=1)  # 엔진 오일 청정제 성분

train['anti_freeze'] = train['K'] + train['B'] + train['CR'] + train['P'] + train['SI']  # 엔진 오일 anti freeze 첨가제 성분
train['anti_freeze_min'] = pd.concat([train['K'], train['B'], train['CR'], train['P'], train['SI']], axis=1).min(axis=1)  # 엔진 오일 anti freeze 첨가제 성분
train['anti_freeze_max'] = pd.concat([train['K'], train['B'], train['CR'], train['P'], train['SI']], axis=1).max(axis=1)  # 엔진 오일 anti freeze 첨가제 성분
train['anti_freeze_mean'] = pd.concat([train['K'], train['B'], train['CR'], train['P'], train['SI']], axis=1).mean(axis=1)  # 엔진 오일 anti freeze 첨가제 성분
train['anti_freeze_std'] = pd.concat([train['K'], train['B'], train['CR'], train['P'], train['SI']], axis=1).std(axis=1)  # 엔진 오일 anti freeze 첨가제 성분
train['anti_freeze_median'] = pd.concat([train['K'], train['B'], train['CR'], train['P'], train['SI']], axis=1).median(axis=1)  # 엔진 오일 anti freeze 첨가제 성분

train['foreign body'] = train['FUEL'] + train['H2O']  # 이물질
train['foreign body_min'] = pd.concat([train['FUEL'], train['H2O']], axis=1).min(axis=1)  # 이물질
train['foreign body_max'] = pd.concat([train['FUEL'], train['H2O']], axis=1).max(axis=1)  # 이물질
train['foreign body_mean'] = pd.concat([train['FUEL'], train['H2O']], axis=1).mean(axis=1)  # 이물질
train['foreign body_std'] = pd.concat([train['FUEL'], train['H2O']], axis=1).std(axis=1)  # 이물질
train['foreign body_median'] = pd.concat([train['FUEL'], train['H2O']], axis=1).median(axis=1)  # 이물질

train['engine_up_wear'] = train['SN'] + train['FE'] + train['CR'] + train['SI'] + train['AL']  # 엔진 상단부 마모
train['engine_up_wear_min'] = pd.concat([train['SN'], train['FE'], train['CR'], train['SI'], train['AL']], axis=1).min(axis=1)  # 엔진 상단부 마모
train['engine_up_wear_max'] = pd.concat([train['SN'], train['FE'], train['CR'], train['SI'], train['AL']], axis=1).max(axis=1)  # 엔진 상단부 마모
train['engine_up_wear_mean'] = pd.concat([train['SN'], train['FE'], train['CR'], train['SI'], train['AL']], axis=1).mean(axis=1)  # 엔진 상단부 마모
train['engine_up_wear_std'] = pd.concat([train['SN'], train['FE'], train['CR'], train['SI'], train['AL']], axis=1).std(axis=1)  # 엔진 상단부 마모
train['engine_up_wear_median'] = pd.concat([train['SN'], train['FE'], train['CR'], train['SI'], train['AL']], axis=1).median(axis=1)  # 엔진 상단부 마모

train['dust'] = train['PB'] + train['SN'] + train['FE'] + train['CR'] + train['SI'] + train['AL'] + train['CU']  # 먼지 유입 됌
train['dust_min'] = pd.concat([train['PB'], train['SN'], train['FE'], train['CR'], train['SI'], train['AL'], train['CU']], axis=1).min(axis=1)  # 먼지 유입 됌
train['dust_max'] = pd.concat([train['PB'], train['SN'], train['FE'], train['CR'], train['SI'], train['AL'], train['CU']], axis=1).max(axis=1)  # 먼지 유입 됌
train['dust_mean'] = pd.concat([train['PB'], train['SN'], train['FE'], train['CR'], train['SI'], train['AL'], train['CU']], axis=1).mean(axis=1)  # 먼지 유입 됌
train['dust_std'] = pd.concat([train['PB'], train['SN'], train['FE'], train['CR'], train['SI'], train['AL'], train['CU']], axis=1).std(axis=1)  # 먼지 유입 됌
train['dust_median'] = pd.concat([train['PB'], train['SN'], train['FE'], train['CR'], train['SI'], train['AL'], train['CU']], axis=1).median(axis=1)  # 먼지 유입 됌

train['cooling_water'] = train['SI'] + train['AL'] + train['NA'] + train['CU'] + train['B']  # 냉각수 유입/ 소포제 유입
train['cooling_water_min'] = pd.concat([train['SI'], train['AL'], train['NA'], train['CU'], train['B']], axis=1).min(axis=1)  # 냉각수 유입/ 소포제 유입
train['cooling_water_max'] = pd.concat([train['SI'], train['AL'], train['NA'], train['CU'], train['B']], axis=1).max(axis=1)  # 냉각수 유입/ 소포제 유입
train['cooling_water_mean'] = pd.concat([train['SI'], train['AL'], train['NA'], train['CU'], train['B']], axis=1).mean(axis=1)  # 냉각수 유입/ 소포제 유입
train['cooling_water_std'] = pd.concat([train['SI'], train['AL'], train['NA'], train['CU'], train['B']], axis=1).std(axis=1)  # 냉각수 유입/ 소포제 유입
train['cooling_water_median'] = pd.concat([train['SI'], train['AL'], train['NA'], train['CU'], train['B']], axis=1).median(axis=1)  # 냉각수 유입/ 소포제 유입

train['anti_wear_agent'] = train['ZN'] + train['P']  # 마모 방지제
train['anti_wear_agent_min'] = pd.concat([train['ZN'], train['P']], axis=1).min(axis=1)  # 마모 방지제
train['anti_wear_agent_max'] = pd.concat([train['ZN'], train['P']], axis=1).max(axis=1)  # 마모 방지제
train['anti_wear_agent_mean'] = pd.concat([train['ZN'], train['P']], axis=1).mean(axis=1)  # 마모 방지제
train['anti_wear_agent_std'] = pd.concat([train['ZN'], train['P']], axis=1).std(axis=1)  # 마모 방지제
train['anti_wear_agent_median'] = pd.concat([train['ZN'], train['P']], axis=1).median(axis=1)  # 마모 방지제

train['aluminum_alloy'] = train['SI'] + train['CU'] + train['MG'] + train['MN'] + train['AL']  # 알루미늄 합금 성분
train['aluminum_alloy_min'] = pd.concat([train['SI'], train['CU'], train['MG'], train['MN'], train['AL']], axis=1).min(axis=1)  # 알루미늄 합금 성분
train['aluminum_alloy_max'] = pd.concat([train['SI'], train['CU'], train['MG'], train['MN'], train['AL']], axis=1).max(axis=1)  # 알루미늄 합금 성분
train['aluminum_alloy_mean'] = pd.concat([train['SI'], train['CU'], train['MG'], train['MN'], train['AL']], axis=1).mean(axis=1)  # 알루미늄 합금 성분
train['aluminum_alloy_std'] = pd.concat([train['SI'], train['CU'], train['MG'], train['MN'], train['AL']], axis=1).std(axis=1)  # 알루미늄 합금 성분
train['aluminum_alloy_median'] = pd.concat([train['SI'], train['CU'], train['MG'], train['MN'], train['AL']], axis=1).median(axis=1)  # 알루미늄 합금 성분

train['special_cast_iron'] = train['SI'] + train['MN'] + train['S'] + train['P'] + train['NI'] + train['CR'] + train['MO'] + train['AL'] + train['TI'] + train['V']  # 특수 주철 성분
train['special_cast_iron_min'] = pd.concat([train['SI'], train['MN'], train['S'], train['P'], train['NI'], train['CR'], train['MO'], train['AL'], train['TI'], train['V']], axis=1).min(axis=1)  # 특수 주철 성분
train['special_cast_iron_max'] = pd.concat([train['SI'], train['MN'], train['S'], train['P'], train['NI'], train['CR'], train['MO'], train['AL'], train['TI'], train['V']], axis=1).max(axis=1)  # 특수 주철 성분
train['special_cast_iron_mean'] = pd.concat([train['SI'], train['MN'], train['S'], train['P'], train['NI'], train['CR'], train['MO'], train['AL'], train['TI'], train['V']], axis=1).mean(axis=1)  # 특수 주철 성분
train['special_cast_iron_std'] = pd.concat([train['SI'], train['MN'], train['S'], train['P'], train['NI'], train['CR'], train['MO'], train['AL'], train['TI'], train['V']], axis=1).std(axis=1)  # 특수 주철 성분
train['special_cast_iron_median'] = pd.concat([train['SI'], train['MN'], train['S'], train['P'], train['NI'], train['CR'], train['MO'], train['AL'], train['TI'], train['V']], axis=1).median(axis=1)  # 특수 주철 성분

train['Silver_bearing'] = train['CU'] + train['AG']  # 은 합금 베어링 성분
train['Silver_bearing_min'] = pd.concat([train['CU'], train['AG']], axis=1).min(axis=1)  # 은 합금 베어링 성분
train['Silver_bearing_max'] = pd.concat([train['CU'], train['AG']], axis=1).max(axis=1)  # 은 합금 베어링 성분
train['Silver_bearing_mean'] = pd.concat([train['CU'], train['AG']], axis=1).mean(axis=1)  # 은 합금 베어링 성분
train['Silver_bearing_std'] = pd.concat([train['CU'], train['AG']], axis=1).std(axis=1)  # 은 합금 베어링 성분
train['Silver_bearing_median'] = pd.concat([train['CU'], train['AG']], axis=1).median(axis=1)  # 은 합금 베어링 성분

train['metal'] = train['BA'] + train['AL'] + train['AG'] + train['BE'] + train['CA'] + train['CD'] + train['CO'] + \
                 train['CR'] + train['CU'] + train['FE'] + train['K'] + train['LI'] + train['MG'] + train['MN'] + \
                 train['MO'] + train['NA'] + train['NI'] + train['PB'] + train['SN'] + train['TI'] + train['V'] + train['ZN']  # 금속
train['metal_min'] = pd.concat([train['BA'], train['AL'], train['AG'], train['BE'], train['CA'], train['CD'], train['CO'],
                 train['CR'], train['CU'], train['FE'], train['K'], train['LI'], train['MG'], train['MN'],
                 train['MO'], train['NA'], train['NI'], train['PB'], train['SN'], train['TI'], train['V'], train['ZN']], axis=1).min(axis=1)  # 금속
train['metal_max'] = pd.concat([train['BA'], train['AL'], train['AG'], train['BE'], train['CA'], train['CD'], train['CO'],
                 train['CR'], train['CU'], train['FE'], train['K'], train['LI'], train['MG'], train['MN'],
                 train['MO'], train['NA'], train['NI'], train['PB'], train['SN'], train['TI'], train['V'], train['ZN']], axis=1).max(axis=1)  # 금속
train['metal_mean'] = pd.concat([train['BA'], train['AL'], train['AG'], train['BE'], train['CA'], train['CD'], train['CO'],
                 train['CR'], train['CU'], train['FE'], train['K'], train['LI'], train['MG'], train['MN'],
                 train['MO'], train['NA'], train['NI'], train['PB'], train['SN'], train['TI'], train['V'], train['ZN']], axis=1).mean(axis=1)  # 금속
train['metal_std'] = pd.concat([train['BA'], train['AL'], train['AG'], train['BE'], train['CA'], train['CD'], train['CO'],
                 train['CR'], train['CU'], train['FE'], train['K'], train['LI'], train['MG'], train['MN'],
                 train['MO'], train['NA'], train['NI'], train['PB'], train['SN'], train['TI'], train['V'], train['ZN']], axis=1).std(axis=1)  # 금속
train['metal_median'] = pd.concat([train['BA'], train['AL'], train['AG'], train['BE'], train['CA'], train['CD'], train['CO'],
                 train['CR'], train['CU'], train['FE'], train['K'], train['LI'], train['MG'], train['MN'],
                 train['MO'], train['NA'], train['NI'], train['PB'], train['SN'], train['TI'], train['V'], train['ZN']], axis=1).median(axis=1)  # 금속

train['metalloid'] = train['B'] + train['SB'] + train['SI']  # 준금속
train['metalloid_min'] = pd.concat([train['B'], train['SB'], train['SI']], axis=1).min(axis=1)  # 준금속
train['metalloid_max'] = pd.concat([train['B'], train['SB'], train['SI']], axis=1).max(axis=1)  # 준금속
train['metalloid_mean'] = pd.concat([train['B'], train['SB'], train['SI']], axis=1).mean(axis=1)  # 준금속
train['metalloid_std'] = pd.concat([train['B'], train['SB'], train['SI']], axis=1).std(axis=1)  # 준금속
train['metalloid_median'] = pd.concat([train['B'], train['SB'], train['SI']], axis=1).median(axis=1)  # 준금속

train['nonmetal'] = train['P'] + train['S']  # 비금속
train['metalloid_min'] = pd.concat([train['P'], train['S']], axis=1).min(axis=1)  # 비금속
train['metalloid_max'] = pd.concat([train['P'], train['S']], axis=1).max(axis=1)  # 비금속
train['metalloid_mean'] = pd.concat([train['P'], train['S']], axis=1).mean(axis=1)  # 비금속
train['metalloid_std'] = pd.concat([train['P'], train['S']], axis=1).std(axis=1)  # 비금속
train['metalloid_median'] = pd.concat([train['P'], train['S']], axis=1).median(axis=1)  # 비금속

train['Particle_acid'] = train['U25'] - train['U4']
train['Particle_acid_min'] = pd.concat([train['U25'], train['U4']], axis=1).min(axis=1)  # 비금속
train['Particle_acid_max'] = pd.concat([train['U25'], train['U4']], axis=1).max(axis=1)  # 비금속
train['Particle_acid_mean'] = pd.concat([train['U25'], train['U4']], axis=1).mean(axis=1)  # 비금속
train['Particle_acid_std'] = pd.concat([train['U25'], train['U4']], axis=1).std(axis=1)  # 비금속
train['Particle_acid_median'] = pd.concat([train['U25'], train['U4']], axis=1).median(axis=1)  # 비금속

train['particle_1'] = train['U4'] - train['U6']  # 입자 크기 4~5um
train['particle_1_min'] = pd.concat([train['U4'], train['U6']], axis=1).min(axis=1)  # 입자 크기 4~5um
train['particle_1_max'] = pd.concat([train['U4'], train['U6']], axis=1).max(axis=1)  # 입자 크기 4~5um
train['particle_1_mean'] = pd.concat([train['U4'], train['U6']], axis=1).mean(axis=1)  # 입자 크기 4~5um
train['particle_1_std'] = pd.concat([train['U4'], train['U6']], axis=1).std(axis=1)  # 입자 크기 4~5um
train['particle_1_median'] = pd.concat([train['U4'], train['U6']], axis=1).median(axis=1)  # 입자 크기 4~5um

train['particle_2'] = train['U6'] - train['U14']  # 입자 크기 6~14um
train['particle_2_min'] = pd.concat([train['U6'], train['U14']], axis=1).min(axis=1)  # 입자 크기 4~5um
train['particle_2_max'] = pd.concat([train['U6'], train['U14']], axis=1).max(axis=1)  # 입자 크기 4~5um
train['particle_2_mean'] = pd.concat([train['U6'], train['U14']], axis=1).mean(axis=1)  # 입자 크기 4~5um
train['particle_2_std'] = pd.concat([train['U6'], train['U14']], axis=1).std(axis=1)  # 입자 크기 4~5um
train['particle_2_median'] = pd.concat([train['U6'], train['U14']], axis=1).median(axis=1)  # 입자 크기 4~5um

train['particle_3'] = train['U14'] - train['U20']  # 입자 크기 14~20um
train['particle_3_min'] = pd.concat([train['U20'], train['U14']], axis=1).min(axis=1)  # 입자 크기 4~5um
train['particle_3_max'] = pd.concat([train['U20'], train['U14']], axis=1).max(axis=1)  # 입자 크기 4~5um
train['particle_3_mean'] = pd.concat([train['U20'], train['U14']], axis=1).mean(axis=1)  # 입자 크기 4~5um
train['particle_3_std'] = pd.concat([train['U20'], train['U14']], axis=1).std(axis=1)  # 입자 크기 4~5um
train['particle_3_median'] = pd.concat([train['U20'], train['U14']], axis=1).median(axis=1)  # 입자 크기 4~5um

train['particle_4'] = train['U4'] - train['U50']  # 피스톤 간극이 40~60um, 입자 크기 50~4um는 통과 가능
train['particle_4_min'] = pd.concat([train['U4'], train['U50']], axis=1).min(axis=1)  # 입자 크기 4~5um
train['particle_4_max'] = pd.concat([train['U4'], train['U50']], axis=1).max(axis=1)  # 입자 크기 4~5um
train['particle_4_mean'] = pd.concat([train['U4'], train['U50']], axis=1).mean(axis=1)  # 입자 크기 4~5um
train['particle_4_std'] = pd.concat([train['U4'], train['U50']], axis=1).std(axis=1)  # 입자 크기 4~5um
train['particle_4_median'] = pd.concat([train['U4'], train['U50']], axis=1).median(axis=1)  # 입자 크기 4~5um

train['particle_5'] = train['U4'] - train['U20']  # 피스톤 링 간극이 20~50um, 입자 크기 20~4um는 통과 가능
train['particle_5_min'] = pd.concat([train['U4'], train['U20']], axis=1).min(axis=1)  # 입자 크기 4~5um
train['particle_5_max'] = pd.concat([train['U4'], train['U20']], axis=1).max(axis=1)  # 입자 크기 4~5um
train['particle_5_mean'] = pd.concat([train['U4'], train['U20']], axis=1).mean(axis=1)  # 입자 크기 4~5um
train['particle_5_std'] = pd.concat([train['U4'], train['U20']], axis=1).std(axis=1)  # 입자 크기 4~5um
train['particle_5_median'] = pd.concat([train['U4'], train['U20']], axis=1).median(axis=1)  # 입자 크기 4~5um

train['particle_6'] = train['U4'] - train['U100']  # 피스톤 링 간극이 30~130um, 입자 크기 100~4um는 통과 가능
train['particle_6_min'] = pd.concat([train['U4'], train['U100']], axis=1).min(axis=1)  # 입자 크기 4~5um
train['particle_6_max'] = pd.concat([train['U4'], train['U100']], axis=1).max(axis=1)  # 입자 크기 4~5um
train['particle_6_mean'] = pd.concat([train['U4'], train['U100']], axis=1).mean(axis=1)  # 입자 크기 4~5um
train['particle_6_std'] = pd.concat([train['U4'], train['U100']], axis=1).std(axis=1)  # 입자 크기 4~5um
train['particle_6_median'] = pd.concat([train['U4'], train['U100']], axis=1).median(axis=1)  # 입자 크기 4~5um

train.to_csv('../data/train_after.csv')
