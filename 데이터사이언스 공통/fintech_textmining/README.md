# 모바일 결제 서비스 앱 리뷰 분석
텍스트마이닝 기법을 통해서 모바일 결제 서비스 앱에 대한 부정 리뷰 데이터 분석
# 0. File and Programming Skills
- kakaopay.ipynb, naverpay.ipynb와 toss.ipynb: selenium 라이브러리를 이용해서 각 모바일 결제 앱의 리뷰 데이터를 수집
- kakao_preprocessing.ipynb, naverpay_preprocessing.ipynb와 toss_preprocessing.ipynb: 텍스트 데이터를 전처리
- kakao_network2.ipynb, naverpay_network2.ipynb, toss_network2.ipynb: networkx 라이브러리를 이용해서 연도별 부정 리뷰에 대한 네트워크 시각화
- kakao_topicmodeling2.ipynb, naverpay_topicmodeling2.ipynb, toss_topicmodeling2.ipynb: gensim 라이브러리를 이용해서 연도별 부정 리뷰에 대한 토픽 모델링 수행

# Introduction
- Goal: 모바일 결제 서비스 앱(카카오페이, 네이버페이, 토스)의 부정 리뷰 데이터 분석을 통해 문제점 도출
- Dataset: selenium 크롤링을 통한 네이버페이, 카카오페이, 토스 어플 리뷰 데이터

# Process
