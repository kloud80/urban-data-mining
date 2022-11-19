"""================================
도로 통행 정보 전처리
한양대 도시공학과 데이터마이닝
구름
================================="""
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
전국 도로 표준 노드 링크 정보
https://its.go.kr/nodelink/nodelinkRef 
"""
data = gpd.read_file('data/MOCT_LINK.shp', encoding='cp949', dtype='str')
data.dtypes

"""
서울시 도로 속도 topis 정보
https://topis.seoul.go.kr/refRoom/openRefRoom_1.do 
"""
topis = pd.read_excel('data/2022년 10월 서울시 차량통행속도.xlsx', dtype='str')
topis.dtypes

"""
서울시 도로 링크 ID와 전국 도로 표준 노드 링크 ID 매칭용 데이터
https://topis.seoul.go.kr/refRoom/openRefRoom_3_3.do 
"""
mapping = pd.read_excel('data/서울시 표준링크 매핑정보_2022년4월 기준.xlsx', dtype='str')
mapping.dtypes


topis.columns
mapping.columns

#서울시 topis 맵핑 테이블에 있는 node만 빼고 삭제
data_seoul = data[data['LINK_ID'].isin(mapping['표준링크아이디'].values)]
data_seoul.plot()
plt.show()

#맵핑 테이블하고 merge하여 서울시 노드 아이디(서비스링크)로 연결
data_seoul = data_seoul.merge(mapping, how='inner', left_on='LINK_ID', right_on='표준링크아이디')
#서울시는 표준도로를 n개 합쳐서 1개의 도로로 지정하므로 표준 노드 shp를 같은 서비스 링크끼리 연결하여 합침 (dissolve)
data_seoul = data_seoul.dissolve(by = '서비스링크').reset_index()
data_seoul.plot()
plt.show()

#한글 fields명을 영문으로 변경
data_seoul = data_seoul.rename(columns={'서비스링크':'service_id', '표준링크아이디':'stand_id'})
#서울시 도로만 shp로 따로 저장 한다.
data_seoul.to_file('data/MOCT_LINK_SEOUL.shp', driver='ESRI Shapefile',)


topis.columns

#topis정보 (한달간 도로 속도 도로별, 시간대별)를 피봇 테이블로 날짜별 시간대별로 변환
topis_pv = topis.pivot_table(index='링크아이디', values=['01시', '02시', '03시', '04시', '05시', '06시', '07시',
       '08시', '09시', '10시', '11시', '12시', '13시', '14시', '15시', '16시', '17시',
       '18시', '19시', '20시', '21시', '22시', '23시', '24시'], columns='일자')


# topis_pv.to_csv('data/topis_pv.csv', index=False, encoding='cp949')

#필드명이 날짜/시간(01시/20221001)으로 되어 있으므로 날짜시간 (2022100101) 타입으로 변경
cols = np.array(topis_pv.columns)
cols_adj = np.array([])
for col in cols :
       cols_adj = np.append(cols_adj, col[1] + col[0][:-1])

# topis_pv = topis_pv.copy()
# 날짜 시간 순으로 데이터 재정렬하기 위해 column명을 소팅
topis_pv.columns = cols_adj
cols_adj = np.sort(cols_adj, axis=0)

#소팅한 컬럼명 순서대로 pandas 재 정렬
topis_pv = topis_pv[cols_adj]

#결측치 0으로 채우기.. 보간처리 해야하는데 패스..
topis_pv = topis_pv.fillna(0.0)

topis_pv = topis_pv.reset_index()
topis_pv = topis_pv.rename(columns={'링크아이디': 'linkid'})
topis_pv.columns

#학습에 사용할 데이터셋으로 저장
topis_pv.to_csv('data/topis_final.csv', index=False)

