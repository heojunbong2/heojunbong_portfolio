import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy import linalg

from datetime import datetime
import pickle
import copy
def check_stationarity(series):
    result = adfuller(series.dropna())  # NaN 값 제거
    return {'ADF Statistic': result[0], 'p-value': result[1], 'Critical Values': result[4]}

def calculate_vif(df):
    vif = pd.DataFrame()
    vif["Variable"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif

def make_stationary(series):
    diff_series = series.diff().dropna()
    return diff_series

def make_positive_definite(matrix):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, a_min=1e-6, a_max=None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def hmm_train(data):
    train_start = '2010-01-01'
    train_end = '2024-05-16'


    data_origin=copy.deepcopy(data)
    # 다중공선성 테스트 (정상성 검사 전)
    initial_vif_results = calculate_vif(data)

    # 다중공선성 결과 출력 (정상성 검사 전)
    print("\nInitial VIF Results (Before Stationarity Check):")
    print(initial_vif_results)

    # 정상성 테스트 및 차분
    diff_count = {col: 0 for col in data.columns}

    for col in data.columns:
        result = check_stationarity(data[col])
        while result['p-value'] > 0.05:
            print(f"{col} 정상성 없음, 차분 진행. {diff_count[col] + 1}번째 차분")
            data[col] = make_stationary(data[col])
            diff_count[col] += 1
            result = check_stationarity(data[col])
        print(f"{col} 정상성 확보됨. 차분 횟수: {diff_count[col]}")

    # 다중공선성 테스트 (정상성 검사 후)
    final_vif_results = calculate_vif(data.dropna())

    # 다중공선성 결과 출력 (정상성 검사 후)
    print("\nFinal VIF Results (After Stationarity Check):")
    print(final_vif_results)

    # 상관계수 히트맵 생성
    corr = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    # 금 종가와의 상관계수가 0.15보다 낮은 경우 제거
    threshold = 0.15
    low_corr_cols = corr.index[abs(corr['GC']) < threshold].tolist()
    if 'GC' in low_corr_cols:
        low_corr_cols.remove('GC')  # 금 종가는 제외
    data.drop(columns=low_corr_cols, inplace=True)
    data = data.fillna(0)
    # 결과 출력
    print("\nColumns dropped due to low correlation with Gold Close price (GC):")
    print(low_corr_cols)
    print("\nData after dropping low correlation columns:")
    print(data.head())
    print(data.columns)

    # HMM 모델 학습을 위한 데이터 준비
    X = data.dropna().values  # 이미 차분된 데이터 사용

    n_components_list = [20, 30, 40, 50, 60, 70]

    best_mae = float('inf')
    best_model = None
    best_n_components = None
    mae_scores = []

    plt.figure(figsize=(15, 40))  # 길이를 늘려서 모든 subplot을 포함하도록 변경

    for i, n_components in enumerate(n_components_list):
        # 히든 마르코프 모델 생성 및 학습
        model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1500, min_covar=1e-3)
        try:
            model.fit(X)
        except (ValueError, linalg.LinAlgError):
            continue
        
        # 예측 수행
        hidden_states = model.predict(X)
        
        # 상태별 평균값을 사용하여 예측된 금 종가 차분 생성
        predicted_diffs = np.zeros(len(hidden_states))
        for j in range(model.n_components):
            state = (hidden_states == j)
            predicted_diffs[state] = model.means_[j, 0]
        
        # 예측된 차분을 실제 가격으로 복원
        predicted_prices = np.r_[data_origin['GC'].iloc[0], predicted_diffs].cumsum()
        
        # MAE 계산
        aligned_length = min(len(data_origin['GC']) - 1, len(predicted_prices) - 1)
        mae = mean_absolute_error(data_origin['GC'].iloc[1:1 + aligned_length], predicted_prices[1:1 + aligned_length])
        mae_scores.append(mae)
        
        # 최적의 모델 저장
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_n_components = n_components
        
        # 예측 결과 시각화
        plt.subplot(len(n_components_list), 1, i + 1)
        plt.plot(data_origin.index, data_origin['GC'], label='Actual Gold Prices')
        plt.plot(data_origin.index[1:1 + aligned_length], predicted_prices[1:1 + aligned_length], label=f'Predicted Gold Prices (n={n_components})', linestyle='--')
        plt.legend()
        plt.title(f'Actual vs Predicted Gold Prices (n={n_components}) - MAE: {mae:.4f}')

    plt.tight_layout()
    plt.show()

    # 최적의 모델 정보 출력 및 저장
    print(f'Best model: {best_n_components} components with MAE: {best_mae:.4f}')
    best_model_params = {
        'n_components': best_n_components,
        'means_': best_model.means_,
        'covars_': best_model.covars_,
        'transmat_': best_model.transmat_,
        'startprob_': best_model.startprob_
    }

    # 최적의 모델 파라미터 저장
    import pickle

    with open('gold_price_hmm_model.pkl', 'wb') as file:
        pickle.dump(best_model_params, file)

    # 최적의 모델로 다시 예측 및 시각화
    hidden_states = best_model.predict(X)
    predicted_diffs = np.zeros(len(hidden_states))
    for j in range(best_model.n_components):
        state = (hidden_states == j)
        predicted_diffs[state] = best_model.means_[j, 0]

    predicted_prices = np.r_[data_origin['GC'].iloc[0], predicted_diffs].cumsum()

    # 인덱스 길이 일치화
    aligned_length = min(len(data_origin['GC']), len(predicted_prices))
    aligned_index = data_origin.index[:aligned_length]

    plt.figure(figsize=(15, 8))
    plt.plot(aligned_index, data_origin['GC'].iloc[:aligned_length], label='Actual Gold Prices')
    plt.plot(aligned_index, predicted_prices[:aligned_length], label='Predicted Gold Prices (Best Model)', linestyle='--')
    plt.legend()
    plt.title(f'Actual vs Predicted Gold Prices (Best Model: n={best_n_components}) - MAE: {best_mae:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Gold Price')
    plt.show()




def hmm_test(fore_start,fore_end):
    # 모델 파라미터 로드 (이전에 저장한 best_model_params 사용)
    with open('gold_price_hmm_model.pkl', 'rb') as file:
        best_model_params = pickle.load(file)

    best_model = GaussianHMM(n_components=best_model_params['n_components'], covariance_type="full", min_covar=1e-3)
    best_model.means_ = best_model_params['means_']
    best_model.covars_ = np.array([make_positive_definite(cov) for cov in best_model_params['covars_']])
    best_model.transmat_ = best_model_params['transmat_']
    best_model.startprob_ = best_model_params['startprob_']

    # 원본 데이터 로드
    train_start = '2010-01-01'
    train_end = '2024-05-16'

    GC = fdr.DataReader('GC=F', train_start, train_end)['Close']
    SI = fdr.DataReader('SI=F', train_start, train_end)['Close']
    HG = fdr.DataReader('HG=F', train_start, train_end)['Close']

    data = pd.DataFrame({
        'GC': GC,
        'SI': SI,
        'HG': HG
    }).dropna()

    data_origin = data.copy()

    # 차분 수행
    for col in data.columns:
        data[col] = data[col].diff().dropna()

    data = data.dropna().fillna(0)

    # 예측할 날짜 설정
    forecast_dates = pd.date_range(start=fore_start, end=fore_end)

    # 예측 시작
    hidden_states = best_model.predict(data.values)
    last_hidden_state = hidden_states[-1]
    predicted_diffs = []

    for _ in range(len(forecast_dates)):
        next_hidden_state = np.random.choice(np.arange(best_model.n_components), p=best_model.transmat_[last_hidden_state])
        predicted_diffs.append(best_model.means_[next_hidden_state, 0])
        last_hidden_state = next_hidden_state

    # 예측된 차분을 실제 가격으로 복원
    last_price = data_origin['GC'].iloc[-1]
    predicted_prices = np.r_[last_price, np.array(predicted_diffs).cumsum() + last_price][1:]

    # 실제 가격 로드
    actual_prices = fdr.DataReader('GC=F', fore_start, fore_end)['Close']
    if len(actual_prices) != len(predicted_prices):
        # 길이 조정 (경우에 따라 실제 가격 데이터가 예측 기간과 일치하지 않을 수 있음)
        actual_prices = actual_prices.reindex(forecast_dates, method='nearest')

    # MAE 계산
    mae = mean_absolute_error(actual_prices, predicted_prices)

    # 예측 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_dates, actual_prices, label='Actual Gold Prices (2024-05-17 to 2024-05-28)', linestyle='-')
    plt.plot(forecast_dates, predicted_prices, label='Predicted Gold Prices', linestyle='--')
    plt.legend()
    plt.title(f'Gold Prices Prediction from 2024-05-17 to 2024-05-28\nMAE: {mae:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Gold Price')
    plt.show()

    # 예측 결과 출력
    predicted_gold_prices = pd.DataFrame({'Date': forecast_dates, 'Predicted_Gold_Price': predicted_prices, 'Actual_Gold_Price': actual_prices.values})
    print(predicted_gold_prices)

    # print('Actual prices: {}'.format(actual_prices.values))
    # print('Predicted prices: {}'.format(predicted_prices))
    print('MAE:', mae)



# hmm_test('2024-05-17','2024-05-26')