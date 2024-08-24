from sklearn.metrics import mean_absolute_error
import numpy as np
import pickle
import pandas as pd
import FinanceDataReader as fdr

# you can create your own function or other .py file to be used here.


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
    
    with open('model_pickle','rb') as f:
        model_bc_fit = pickle.load(f)
    

    # TODO: forecast / check the performance
    test = GC_test['Close']
    pred_bc = model_bc_fit.forecast(steps=len(test))
    
    from scipy.special import inv_boxcox

    # 역변환 함수
    def reverse_boxcox(x, lmbda):
        if lmbda == 0:
            return np.exp(x)
        else:
            return np.exp(np.log(lmbda * x + 1) / lmbda)

    # 예측값 역변환
    ### LAMBDA : -0.9558290790517501 ###
    pred_bc_orig = reverse_boxcox(pred_bc, -0.9558290790517501)

    # 데이터프레임 생성
    test_pred_bc = pd.DataFrame({'test':test, 'pred':pred_bc_orig}, index=test.index)
    test_pred_bc
    
    # print the results    

    print('actual prices : {}'.format( [i[1] for i in test.items()] ))
    print('predicted prices : {}'.format( pred_bc_orig ))
    print('MAE :', mean_absolute_error(test, pred_bc_orig))

if __name__ == '__main__':
    main()