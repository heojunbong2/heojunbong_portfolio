# 모바일 결제 서비스 앱 리뷰 분석
텍스트마이닝 기법을 통해서 모바일 결제 서비스 앱에 대한 부정 리뷰 데이터 분석
# 0. File and Programming Skills
- kakaopay.ipynb, naverpay.ipynb와 toss.ipynb: selenium 라이브러리를 이용해서 각 모바일 결제 앱의 리뷰 데이터를 수집
- kakao_preprocessing.ipynb, naverpay_preprocessing.ipynb와 toss_preprocessing.ipynb: 텍스트 데이터를 전처리
- kakao_network2.ipynb, naverpay_network2.ipynb, toss_network2.ipynb: networkx 라이브러리를 이용해서 연도별 부정 리뷰에 대한 네트워크 시각화
- kakao_topicmodeling2.ipynb, naverpay_topicmodeling2.ipynb, toss_topicmodeling2.ipynb: gensim 라이브러리를 이용해서 연도별 부정 리뷰에 대한 토픽 모델링 수행

# 1. Introduction
- Goal: 모바일 결제 서비스 앱(카카오페이, 네이버페이, 토스)의 부정 리뷰 데이터 분석을 통해 문제점 도출
- Dataset: selenium 크롤링을 통한 네이버페이, 카카오페이, 토스 어플 리뷰 데이터

# 2. Process
- 데이터 수집: selenium을 이용한 구글 플레이스토어의 KaKaoPay, Naverpay, Toss 리뷰 데이터 크롤링
- 데이터 추출: 최근 2년간(2021년 10월~2023년 10월)의 온라인 간편 결제 서비스 앱의 문제점을 알기 위해 부정적 리뷰 데이터만 추출
- 데이터 전처리1: 한글 자음, 모음, 숫자, 특수 문자 기호를 제거하여 기본적인 텍스트 데이터 전처리를 수행하고 Konlpy 툴을 시용하여 사용자 사전을 만듬
- 데이터 전처리2: 어플 이름명과 경쟁업체 회사명 불용어 처리
- 데이터 시각화: 연도별로 워드클라우드 시각화하고 키워드 간의 관계를 알기 위해서 연도별로 네트워크 시각화를 수행함
- 정적 토픽 모델링: 중심 키워드들을 통해 어플의 부정적인 기능을 추론함
- 동적 토픽 모델링: 좀 더 깊은 분석을 위해 시간에 따른 키워드 추이 그래프를 시각화함

