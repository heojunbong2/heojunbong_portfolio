# 용해탱크 제조 데이터를 이용한 용해 품질 분류기 모델 구축
# 0. File and Programming Skills
- EDA.ipynb: 데이터의 각 변수들의 특징들 시각화  
- Feature_importance.ipynb: 분류모델의 변수 중요도 확인(변수중요도가 낮은 INSP 변수는 모델 구축 시 제외함)
- Model.ipynb: 여러 약분류모델(Logistic regression, KNN), 강분류모델(Soft voting 모델, Xgboost 모델, Randomforest 모델)
  
# 1. Introduction
- Goal: 용해탱크 제조 데이터를 활용해서 최적의 용해 품질 분류기 모델을 만들자
- Dataset: 용해탱크 제조 관련 데이터 변수들을(STD_DT, NUM, MELT_TEMP, MOTORSPEED, MELT_WEIGHT, INSP, TAG) 가지고 있는 정형데이터셋

# 2. Process
- 먼저 RandomForest 모델을 구축한 다음에 모델의 용해 품질 관련 변수 중요도를 확인하고 변수 중요도가 낮은 변수는 제거하기
- 수정된 데이터셋으로 matplotlib과 seaborn 라이브러리를 이용해서 EDA해서 변수들의 특성을 파악하기
- EDA 수행 후 여러 모델들을 구축하여 F-1 Score와 AUC Score가 가장 높은 모델을 선정하기

# 3. Analaysis
- F-1 Score와 AUC가 높은 XGboost model을 구축함
  
![image](https://github.com/heojunbong2/portfolio/assets/168062535/5f8e9828-062f-4b49-af90-4c9bd0ff7b45)
