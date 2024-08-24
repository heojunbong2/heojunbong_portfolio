import FinanceDataReader as fdr
print('시작')
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import time 
from scipy.stats import entropy 
import warnings 
warnings.filterwarnings('ignore')
from tqdm import tqdm 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder
from scipy.stats import linregress
from xgboost import XGBRegressor
from joblib import dump
from joblib import load


# you can create your own function or other .py file to be used here.

def get_slope(array):
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return slope

# 파생변수(년,월,일, 5일간격, 15일 간격 이동평균의 기울기, 평균, 표준편차, 최솟값, 최댓값) 포함해서 학습한 xgboost

def main():

    # ----------------Do not change this part ---------------------------------------
    train_start = '2010-01-01'
    train_end = '2024-05-16'

    # 원자재 가격
    GC = fdr.DataReader('GC=F', train_start, train_end) # 금 선물 (COMEX)
    CL = fdr.DataReader('CL=F', train_start, train_end) # WTI유 선물 Crude Oil (NYMEX)
    BZ = fdr.DataReader('BZ=F', train_start, train_end) # 브렌트유 선물 Brent Oil (NYMEX)
    NG = fdr.DataReader('NG=F', train_start, train_end) # 천연가스 선물 (NYMEX)
    SI = fdr.DataReader('SI=F', train_start, train_end) # 은 선물 (COMEX)
    HG = fdr.DataReader('HG=F', train_start, train_end) # 구리 선물 (COMEX)

    # 환율
    USDKRW = fdr.DataReader('USD/KRW', train_start, train_end) # 달러 원화
    USDEUR = fdr.DataReader('USD/EUR', train_start, train_end) # 달러 유로화
    USDCNY = fdr.DataReader('USD/CNY', train_start, train_end) # 달러 위엔화

    # 암호화폐 데이터
    BTCUSD= fdr.DataReader('BTC/USD', train_start, train_end) # 비트코인/달러
    ETHUSD = fdr.DataReader('ETH/USD', train_start, train_end) # 이더리움/달러
    #-----------------------------------------------------------------------------
    
    # TODO: preprocessing / model training
    
    GC=GC.reset_index()
    GC=GC.fillna(method='bfill')
    dt=GC['Date'].astype('str')
    month_data=pd.to_datetime(dt)
    GC['month']=month_data.dt.month
    year_data=pd.to_datetime(dt)
    GC['year']=year_data.dt.year
    GC['day']=year_data.dt.day   
    GC['wd']=year_data.dt.weekday   ## 한 주의 요일을 나타내는 단어
    
    GC['slope5'] = GC['Close'].rolling(5).apply(get_slope, raw=True)
    GC['slope15'] = GC['Close'].rolling(15).apply(get_slope, raw=True)

    GC['std5'] = GC['Close'].rolling(5).std(raw=True)
    GC['std15'] = GC['Close'].rolling(15).std(raw=True)

    GC['mean5'] = GC['Close'].rolling(5).mean(raw=True)
    GC['mean15'] = GC['Close'].rolling(15).mean(raw=True)

    GC['skew5'] = GC['Close'].rolling(5).skew()
    GC['skew15'] = GC['Close'].rolling(15).skew()

    GC['kurt5'] = GC['Close'].rolling(5).kurt()
    GC['kurt15'] = GC['Close'].rolling(15).kurt()

    GC['min5'] = GC['Close'].rolling(5).min()
    GC['min15'] = GC['Close'].rolling(15).min()

    GC['max5'] = GC['Close'].rolling(5).max()
    GC['max15'] = GC['Close'].rolling(15).max()
    
    print('파생변수 완료')
    GC_data=GC.iloc[15:,:]                    # 이동평균 15일 이전 값들을 None값이 들어 있기 때문에 없애기
    GC_data.drop(['Date'],inplace=True,axis=1)
    
    train_data=GC_data.iloc[:-11,:]            # 5월 1일부터 15일까지의 데이터들은 validation set으로 이용할 것
    valid_data=GC_data.iloc[-11:,:]            
    train_x=train_data.drop(['Close'],axis=1)
    train_y=train_data[['Close']]
    valid_x=valid_data.drop(['Close'],axis=1)
    valid_y=valid_data[['Close']]
    
    # 하이퍼파라미터 값의 범위 설정
    
    n_estimators_range = [200,400]
    max_depth_range = [5,9]
    learning_rate_range = [0.05, 0.1]
    
    # 최적의 하이퍼파라미터와 최적의 성능을 저장할 변수 초기화
    
    best_params = None
    best_score = float('inf')
    
    # 이중 for 문을 사용하여 Grid Search 수행
    print('하이퍼파라미터튜닝')
    for n_estimators in n_estimators_range:
        for max_depth in max_depth_range:
            for learning_rate in learning_rate_range:
                # 모델 초기화 및 학습
                model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
                model.fit(train_x, train_y)
        
                # 테스트 데이터에 대한 예측값 생성
                valid_y_pred = model.predict(valid_x)
        
                # 성능 평가 (MAE)
                mae = mean_absolute_error(valid_y, valid_y_pred)
        
                # 최적의 성능과 하이퍼파라미터 업데이트
                if mae < best_score:
                    best_score = mae
                    best_params = {'n_estimators': n_estimators, 'max_depth': max_depth,'learning_rate': learning_rate}
    
    xgbmodel=XGBRegressor(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth'], learning_rate=best_params['learning_rate'])
    print('모델학습')
    xgbmodel.fit(train_x,train_y)
    
    
    
    
    # TODO: save a file of your model
    
    # 모델 저장
    print('모델 2(파생변수 O) 저장하기')
    
    dump(xgbmodel, '/home/iai/heo/project_gold/model/xgb_model')
    
# 파생 변수 없이 있는 변수로 학습한 original xgboost

def main2():
    
    # ----------------Do not change this part ---------------------------------------
    train_start = '2010-01-01'
    train_end = '2024-05-16'

    # 원자재 가격
    GC = fdr.DataReader('GC=F', train_start, train_end) # 금 선물 (COMEX)
    CL = fdr.DataReader('CL=F', train_start, train_end) # WTI유 선물 Crude Oil (NYMEX)
    BZ = fdr.DataReader('BZ=F', train_start, train_end) # 브렌트유 선물 Brent Oil (NYMEX)
    NG = fdr.DataReader('NG=F', train_start, train_end) # 천연가스 선물 (NYMEX)
    SI = fdr.DataReader('SI=F', train_start, train_end) # 은 선물 (COMEX)
    HG = fdr.DataReader('HG=F', train_start, train_end) # 구리 선물 (COMEX)

    # 환율
    USDKRW = fdr.DataReader('USD/KRW', train_start, train_end) # 달러 원화
    USDEUR = fdr.DataReader('USD/EUR', train_start, train_end) # 달러 유로화
    USDCNY = fdr.DataReader('USD/CNY', train_start, train_end) # 달러 위엔화

    # 암호화폐 데이터
    BTCUSD= fdr.DataReader('BTC/USD', train_start, train_end) # 비트코인/달러
    ETHUSD = fdr.DataReader('ETH/USD', train_start, train_end) # 이더리움/달러
    #-----------------------------------------------------------------------------
    
    # TODO: preprocessing / model training
    
    GC=GC.reset_index()
    GC=GC.fillna(method='bfill')
    GC['Date']=GC['Date'].astype('str')
    
    GC.drop(['Date'],inplace=True,axis=1)
    
    train_data=GC.iloc[:-11,:]            # 5월 1일부터 15일까지의 데이터들은 validation set으로 이용할 것
    valid_data=GC.iloc[-11:,:]            
    train_x=train_data.drop(['Close'],axis=1)
    train_y=train_data[['Close']]
    valid_x=valid_data.drop(['Close'],axis=1)
    valid_y=valid_data[['Close']]
    
    # 하이퍼파라미터 값의 범위 설정
    
    n_estimators_range = [200,400]
    max_depth_range = [5,9]
    learning_rate_range = [0.05, 0.1]
    
    # 최적의 하이퍼파라미터와 최적의 성능을 저장할 변수 초기화
    
    best_params = None
    best_score = float('inf')
    
    # 이중 for 문을 사용하여 Grid Search 수행
    print('이번에는 파생변수가 없이 모델 학습시키기')

    for n_estimators in n_estimators_range:
        for max_depth in max_depth_range:
            for learning_rate in learning_rate_range:
                # 모델 초기화 및 학습
                model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
                model.fit(train_x, train_y)
        
                # 테스트 데이터에 대한 예측값 생성
                valid_y_pred = model.predict(valid_x)
        
                # 성능 평가 (MAE)
                mae = mean_absolute_error(valid_y, valid_y_pred)
        
                # 최적의 성능과 하이퍼파라미터 업데이트
                if mae < best_score:
                    best_score = mae
                    best_params = {'n_estimators': n_estimators, 'max_depth': max_depth,'learning_rate': learning_rate}
    print('하이퍼파라미터 튜닝 완료')
    xgbmodel2=XGBRegressor(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth'], learning_rate=best_params['learning_rate'])
    xgbmodel2.fit(train_x,train_y)
    
    
    
    
    # TODO: save a file of your model
    
    # 모델 저장
    print('모델 2(파생변수 X) 저장하기')
    dump(xgbmodel2, '/home/iai/heo/project_gold/model/xgb_model2')

if __name__ == "__main__":
    main()
    main2()