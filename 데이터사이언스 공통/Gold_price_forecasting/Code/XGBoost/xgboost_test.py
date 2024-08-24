import FinanceDataReader as fdr
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


# 파생변수를 포함해서 학습한 모델로 test data 평가하기
## (그러므로 test data로 예측할려면 test data도 파생변수를 구해야됨) 그래서 test data의 이전 날짜의 train data를 불러옴

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
   
    test_start = '2024-05-16'
    test_end = '2024-05-27'
    
    # 금 가격 test set (정답셋)
    GC_test = fdr.DataReader('GC=F', test_start, test_end) # 금 선물 (COMEX) 
    #-----------------------------------------------------------------------------
    
    # TODO: load your model
    
    GC_test=GC_test.reset_index()
    dt=GC_test['Date'].astype('str')
    month_data=pd.to_datetime(dt)
    GC_test['month']=month_data.dt.month
    year_data=pd.to_datetime(dt)
    GC_test['year']=year_data.dt.year
    GC_test['day']=year_data.dt.day 
    ## 한 주의 요일을 나타내는 단어  
    GC_test['wd']=year_data.dt.weekday
    
    # test data의 이동평균 값들을 구하기 위해 train data와 결합함
    
    GC_concat=GC[-20:]
    GC_testtotal=pd.concat([GC_concat,GC_test])

    GC_testtotal['slope5'] = GC_testtotal['Close'].rolling(5).apply(get_slope, raw=True)
    GC_testtotal['slope15'] = GC_testtotal['Close'].rolling(15).apply(get_slope, raw=True)

    GC_testtotal['std5'] = GC_testtotal['Close'].rolling(5).std(raw=True)
    GC_testtotal['std15'] = GC_testtotal['Close'].rolling(15).std(raw=True)

    GC_testtotal['mean5'] = GC_testtotal['Close'].rolling(5).mean(raw=True)
    GC_testtotal['mean15'] = GC_testtotal['Close'].rolling(15).mean(raw=True)

    GC_testtotal['skew5'] = GC_testtotal['Close'].rolling(5).skew()
    GC_testtotal['skew15'] = GC_testtotal['Close'].rolling(15).skew()

    GC_testtotal['kurt5'] = GC_testtotal['Close'].rolling(5).kurt()
    GC_testtotal['kurt15'] = GC_testtotal['Close'].rolling(15).kurt()

    GC_testtotal['min5'] = GC_testtotal['Close'].rolling(5).min()
    GC_testtotal['min15'] = GC_testtotal['Close'].rolling(15).min()

    GC_testtotal['max5'] = GC_testtotal['Close'].rolling(5).max()
    GC_testtotal['max15'] = GC_testtotal['Close'].rolling(15).max()
    
    GC_testtotal2=GC_testtotal[-7:]
    GC_testtotal2.drop(['Date'],inplace=True,axis=1)
    GC_test_x=GC_testtotal2.drop(['Close'],axis=1)
    GC_test_y=GC_testtotal2[['Close']]
    
    # TODO: forecast / check the performance
    
    model=load('/home/iai/heo/project_gold/model/xgb_model')
    y_pred=model.predict(GC_test_x)
    mae=mean_absolute_error(GC_test_y,y_pred)
    print('파생변수를 포함하고 학습한 xgboost model: ')
    data2=pd.DataFrame()
    data2['Real']=np.array(GC_test['Close'])
    data2['Prediction']=np.array(y_pred)
    print(data2)
    print('MAE: ',mae)    
    
    
    pd.DataFrame(GC_test).to_csv('/home/iai/heo/project_gold/result/Real.csv',index=False)
    pd.DataFrame(y_pred).to_csv('/home/iai/heo/project_gold/result/prediction(feature_engineering_xgboost).csv',index=False)
    
def main2():
    

    test_start = '2024-05-16'
    test_end = '2024-05-27'
    
    # 금 가격 test set (정답셋)
    GC_test = fdr.DataReader('GC=F', test_start, test_end) # 금 선물 (COMEX) 
    #-----------------------------------------------------------------------------
    
    # TODO: load your model
    
    GC_test=GC_test.reset_index()
    GC_test.drop(['Date'],axis=1,inplace=True)
    
    
    # TODO: forecast / check the performance
    
    model=load('/home/iai/heo/project_gold/model/xgb_model2')
    GC_test_x=GC_test.drop(['Close'],axis=1)
    GC_test_y=GC_test[['Close']]
    y_pred2=model.predict(GC_test_x)
    mae=mean_absolute_error(GC_test_y,y_pred2)
    print('파생변수를 포함하지 않고 학습한 xgboost model: ')
    data2=pd.DataFrame()
    data2['Real']=np.array(GC_test['Close'])
    data2['Prediction']=np.array(y_pred2)
    print(data2)
    print('MAE: ',mae)    

    pd.DataFrame(y_pred2).to_csv('/home/iai/heo/project_gold/result/prediction(original_xgboost).csv',index=False)

# print the results    
    
if __name__ == '__main__':
    main()
    main2()
