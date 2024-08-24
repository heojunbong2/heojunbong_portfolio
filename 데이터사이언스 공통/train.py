import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import interpolate
from scipy.stats import boxcox
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


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
    print(f"train data start date : {train_start}")
    print(f"train data end date : {train_start}")
       

    GC_org_cols = list(GC.columns)
    CL_org_cols = list(CL.columns)
    BZ_org_cols = list(BZ.columns)
    NG_org_cols = list(NG.columns)
    SI_org_cols = list(SI.columns)
    HG_org_cols = list(HG.columns)
    USDKRW_org_cols = list(USDKRW.columns)
    USDEUR_org_cols = list(USDEUR.columns)
    USDCNY_org_cols = list(USDCNY.columns)
    BTCUSD_org_cols = list(BTCUSD.columns)
    ETHUSD_org_cols = list(ETHUSD.columns)

    # 원자재
    GC_org_cols2 = ['GC_'+i for i in GC_org_cols]
    CL_org_cols2 = ['CL_'+i for i in CL_org_cols]
    BZ_org_cols2 = ['BZ_'+i for i in BZ_org_cols]
    NG_org_cols2 = ['NG_'+i for i in NG_org_cols]
    SI_org_cols2 = ['ST_'+i for i in SI_org_cols]
    HG_org_cols2 = ['HG_'+i for i in HG_org_cols]
    # 환율
    USDKRW_org_cols2 = ['USDKRW_'+i for i in USDKRW_org_cols]
    USDEUR_org_cols2 = ['USDEUR_'+i for i in USDEUR_org_cols]
    USDCNY_org_cols2 = ['USDCNY_'+i for i in USDCNY_org_cols]
    # 암호화폐 데이터
    BTCUSD_org_cols2 = ['BTDUSD_'+i for i in BTCUSD_org_cols]
    ETHUSD_org_cols2 = ['ETHUSD_'+i for i in ETHUSD_org_cols]


    # 원자재
    GC.columns = GC_org_cols2
    CL.columns = CL_org_cols2
    BZ.columns = BZ_org_cols2
    NG.columns = NG_org_cols2
    SI.columns = SI_org_cols2
    HG.columns = HG_org_cols2
    # 환율
    USDKRW.columns = USDKRW_org_cols2
    USDEUR.columns = USDEUR_org_cols2
    USDCNY.columns = USDCNY_org_cols2
    # 암호화폐 데이터
    BTCUSD.columns = BTCUSD_org_cols2
    ETHUSD.columns = ETHUSD_org_cols2

    # GC timeindex 로 공통 인덱스 생성
    common_index = GC.index

    # 결측치 처리 - 여기서는 선형 보간법 사용
    # dataframe.interpolate(method='linear', inplace=True)
    GC.interpolate(method='quadratic', inplace=True)
    CL.interpolate(method='quadratic', inplace=True)
    BZ.interpolate(method='quadratic', inplace=True)
    NG.interpolate(method='quadratic', inplace=True)
    SI.interpolate(method='quadratic', inplace=True)
    HG.interpolate(method='quadratic', inplace=True)

    USDKRW.interpolate(method='quadratic', inplace=True)
    USDEUR.interpolate(method='quadratic', inplace=True)
    USDCNY.interpolate(method='quadratic', inplace=True)

    BTCUSD.interpolate(method='quadratic', inplace=True)
    ETHUSD.interpolate(method='quadratic', inplace=True)


    # ADF 검정
    print("### START ADF ###")
    adf_result = adfuller(GC['GC_Close'])
    print('ADF Statistic:', adf_result[0])
    print('p-value:', adf_result[1])
    for key, value in adf_result[4].items():
        print('Critical Value (%s): %.3f' % (key, value))
    print("### END ADF ###")

    # KPSS 검정
    print("### START KPSS ###")
    kpss_result = kpss(GC['GC_Close'], regression='c')
    print('KPSS Statistic:', kpss_result[0])
    print('p-value:', kpss_result[1])
    for key, value in kpss_result[3].items():
        print('Critical Value (%s): %.3f' % (key, value))
    print("### END KPSS ###")


    train = GC['GC_Close']

    kpss_diffs = pmdarima.arima.ndiffs(train, alpha=0.05, test='kpss', max_d=5)
    adf_diffs = pmdarima.arima.ndiffs(train, alpha=0.05, test='adf', max_d=5)
    n_diffs = max(kpss_diffs, adf_diffs)

    print(f"Optimized 'd' = {n_diffs}")

    bc_train, lmbda = boxcox(train)
    
    print(f"LAMBDA : {lmbda}")
    
    model_bc = ARIMA(bc_train, order=(0,1,1))
    model_bc_fit = model_bc.fit()
    model_bc_fit.summary()
    
    # TODO: save a file of your model
    print(f'### MODEL SAVE ###')
    with open('model_pickle','wb') as f:
        pickle.dump(model_bc_fit,f)
       


if __name__ == "__main__":
    main()

