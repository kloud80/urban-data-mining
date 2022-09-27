# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 10:19:51 2021

@author: Kloud


"""
import pandas as pd
import geopandas as gpd
import os, re, sys
from glob import glob
from tqdm import tqdm
import time
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('font', family='gulim') #한글 폰트 적용시
os.chdir('04 SVM/')


"""----------------------------------------------
data 소스 파일 다운로드 : https://www.dropbox.com/s/gqa6jxfvevbu5yx/data.zip?dl=0
---------------------------------------------"""

#%%
# 생활인구 집계구 shp 데이터 가져오기
g_생활인구 = gpd.read_file('../data/통계지역경계(2016년+기준)/집계구.shp', dtype='str')
g_생활인구 = g_생활인구.set_crs('EPSG:5179') #생활인구 집계구 좌표계
g_생활인구 = g_생활인구.to_crs('EPSG:5174')  #지적도 좌표계로 변환
g_생활인구.plot('ADM_NM', figsize=(20,20))

g_생활인구[g_생활인구['ADM_NM']=='사근동'].plot('TOT_REG_CD', figsize=(20,20))

#%%
#2018년 생활인구 6월
flist = glob('data/LOCAL_PEOPLE_201806/*.csv')
data = pd.read_csv(flist[0], sep=',', encoding='utf-8', dtype='str')
print(data.dtypes)
print(data['기준일ID'].value_counts())
print(data['시간대구분'].value_counts())

생활인구18년6월 = pd.DataFrame(columns=['기준일ID', '집계구코드', 'total'])

for file in tqdm(flist):
    data = pd.read_csv(file, sep=',', encoding='utf-8', dtype='str')
    
    #요일을 가져온다
    wday = datetime.date(int(data.loc[0,'기준일ID'][:4]), int(data.loc[0,'기준일ID'][4:6]), int(data.loc[0,'기준일ID'][6:])).weekday()
    
    if wday in [5, 6] : #토요일, 일요일은 패스
        continue
    
        
    #데이터 타입정의 *표기 내용 0으로 변경, 문자타입>실수>정수로 변경(56.0 으로 표기된 셀 존재)
    data[data.columns[4:]] = data[data.columns[4:]].replace('*', '0')
    data[data.columns[4:]] = data[data.columns[4:]].astype('float').astype('int')
    
    #20~24, 25~29세 남/여 생활인구만 합쳐서 자름
    data['total'] = data['남자20세부터24세생활인구수'] + data['남자25세부터29세생활인구수'] + data['여자20세부터24세생활인구수'] + data['여자25세부터29세생활인구수']
    data = data[['기준일ID', '시간대구분', '집계구코드', 'total']]
 
    #19시~22시 생활인구만 자름
    data = data[data['시간대구분'].isin(['19', '20', '21', '22'])]
    
    #평균 생활인구수 계산
    data = data.groupby(['기준일ID', '집계구코드'])['total'].mean().reset_index()
    
    #모두 합친다
    생활인구18년6월 = pd.concat([생활인구18년6월, data], axis=0)


#전체 데이터 출력
summary = 생활인구18년6월.groupby(['기준일ID'])['total'].sum()
plt.figure(figsize=(20,10))
plt.bar(summary.index, summary)

#최종 18년 6월 평일 19~22시까지의 각 집계구별 평균 생활인구수
생활인구18년6월 = 생활인구18년6월.groupby(['집계구코드'])['total'].mean().reset_index()

del data, file, flist, summary, wday

#%%
#2021년 생활인구 6월
flist = glob('LOCAL_PEOPLE_202106\*.csv')
data = pd.read_csv(flist[0], sep=',', encoding='cp949', dtype='str')
print(data.dtypes)
print(data['?"기준일ID"'].value_counts())
print(data['시간대구분'].value_counts())

생활인구21년6월 = pd.DataFrame(columns=['?"기준일ID"', '집계구코드', 'total'])

for file in tqdm(flist):
    data = pd.read_csv(file, sep=',', encoding='cp949', dtype='str')
    
    #요일을 가져온다
    wday = datetime.date(int(data.loc[0,'?"기준일ID"'][:4]), int(data.loc[0,'?"기준일ID"'][4:6]), int(data.loc[0,'?"기준일ID"'][6:])).weekday()
    
    if wday in [5, 6] : #토요일, 일요일은 패스
        continue
    
        
    #데이터 타입정의 *표기 내용 0으로 변경, 문자타입>실수>정수로 변경(56.0 으로 표기된 셀 존재)
    data[data.columns[4:]] = data[data.columns[4:]].replace('*', '0')
    data[data.columns[4:]] = data[data.columns[4:]].astype('float').astype('int')
    
    #20~24, 25~29세 남/여 생활인구만 합쳐서 자름
    data['total'] = data['남자20세부터24세생활인구수'] + data['남자25세부터29세생활인구수'] + data['여자20세부터24세생활인구수'] + data['여자25세부터29세생활인구수']
    data = data[['?"기준일ID"', '시간대구분', '집계구코드', 'total']]
 
    #19시~22시 생활인구만 자름
    data = data[data['시간대구분'].isin(['19', '20', '21', '22'])]
    
    #평균 생활인구수 계산
    data = data.groupby(['?"기준일ID"', '집계구코드'])['total'].mean().reset_index()
    
    #모두 합친다
    생활인구21년6월 = pd.concat([생활인구21년6월, data], axis=0)


#전체 데이터 출력
summary = 생활인구21년6월.groupby(['?"기준일ID"'])['total'].sum()
plt.figure(figsize=(20,10))
plt.bar(summary.index, summary)

#최종 18년 6월 평일 19~22시까지의 각 집계구별 평균 생활인구수
생활인구21년6월 = 생활인구21년6월.groupby(['집계구코드'])['total'].mean().reset_index()

del data, file, flist, summary, wday
#%%
""" 데이터를 합쳔디 """
생활인구18년6월 = 생활인구18년6월.rename(columns={'집계구코드' : 'TOT_REG_CD', 'total' : '18년6월'})
생활인구21년6월 = 생활인구21년6월.rename(columns={'집계구코드' : 'TOT_REG_CD', 'total' : '21년6월'})

g_생활인구 = g_생활인구.merge(생활인구18년6월, how='left', left_on='TOT_REG_CD', right_on='TOT_REG_CD')
g_생활인구 = g_생활인구.merge(생활인구21년6월, how='left', left_on='TOT_REG_CD', right_on='TOT_REG_CD')


fig, ax = plt.subplots(1, 1, figsize=(20,15))
g_생활인구.plot('18년6월', ax=ax, cmap='Reds', legend=True)

fig, ax = plt.subplots(1, 1, figsize=(20,15))
g_생활인구.plot('21년6월', ax=ax, cmap='Reds', legend=True)


g_생활인구['인구차이'] = g_생활인구['21년6월'] - g_생활인구['18년6월']


fig, ax = plt.subplots(1, 1, figsize=(20,15))
g_생활인구.plot('인구차이', ax=ax, cmap='bwr', legend=True)

g_생활인구['area'] = g_생활인구.geometry.area

g_생활인구['인구차이'] = g_생활인구['인구차이'] / g_생활인구['area']


fig, ax = plt.subplots(1, 1, figsize=(20,15))
g_생활인구.plot('인구차이', ax=ax, cmap='bwr', legend=True, vmin=-0.005, vmax=0.005)


g_생활인구.to_file('data/results/g_생활인구.shp', encoding='cp949')


#%%

"""연속지적도를 불러온다 """ #**********오래 걸림
연속지적 = gpd.read_file('AL_11_D002_20210904.shp', encoding='cp949')
연속지적 = 연속지적.rename(columns={'A0' : 'code', 'A1' : '고유번호', 'A2' : '법정동코드', 'A3' : '법정동명', 'A4' : '지번', 'A5' : '지목', 'A6' :'날짜'})
연속지적 = 연속지적.set_crs('EPSG:5174')
연속지적['지목'] = 연속지적['지목'].apply(lambda x : re.sub('[0-9\- ]', '', x))
tmp = 연속지적.head(10)


""" 결과 표시 지적도는 역삼동만 표시 느려서"""
ax = 연속지적[연속지적['법정동코드']=='1168010100'].plot('지목', figsize=(15,15), cmap='Greys')
ax = g_생활인구[g_생활인구['ADM_NM'].isin(['역삼1동','역삼2동'])].plot('TOT_REG_CD', ax=ax, alpha=0.5)

#**********오래 걸림
생활_지적 = gpd.overlay(g_생활인구, 연속지적, how='intersection')


""" 각 필지별 참여 비율 계산"""
연속지적['면적'] = 연속지적.geometry.area
생활_지적['면적'] = 생활_지적.geometry.area

필지면적 = 연속지적[['고유번호','면적']]
필지면적 = 필지면적.rename(columns={'면적' : '원래면적'})

생활_지적 = pd.merge(생활_지적, 필지면적, how='left', left_on='고유번호', right_on='고유번호')

tmp = 생활_지적.head(10)

""" 쉐이프로 저장 오래 걸림"""
# 생활_지적.to_file('results\g_생활_지적.shp', encoding='cp949')

del ax,  연속지적, 필지면적

#%%
""" 층별개요 데이터를 서울시만 자른 후 불러 낸다"""
f = open ('mart_djy_04.txt', 'r') #층별개요 전체
d = open ('mart_djy_04_서울.txt', 'w') #서울만 저장

num = 0
while True:
    line = f.readline()
    if not line:
        break
    
    if line.split('|')[4][:2] == '11' :
        d.writelines(line)
    if num % 10000 == 0:
        sys.stdout.write('\r진행 : ' + str(num))
    num+=1

sys.stdout.write('\n')
f.close()
d.close()

del f, d, line, num
#%%
""" 층별개요를 불러낸 후 필지별, 용도별 면적 정보 데이터를 만든다"""
colnames=['관리_건축물대장_PK', '대지_위치', '도로명_대지_위치', '건물_명', '시군구_코드', '법정동_코드', '대지_구분_코드', '번', '지', '특수지_명', '블록', '로트', '새주소_도로_코드', '새주소_법정동_코드', '새주소_지상지하_코드', '새주소_본_번', '새주소_부_번', '동_명', '층_구분_코드', '층_구분_코드_명', '층_번호', '층_번호_명', '구조_코드', '구조_코드_명', '기타_구조', '주_용도_코드', '주_용도_코드_명', '기타_용도', '면적(㎡)', '주_부속_구분_코드', '주_부속_구분_코드_명', '면적_제외_여부', '생성_일자']

층별개요 = pd.read_csv('mart_djy_04_서울.txt', sep='|', encoding='cp949', dtype='str', names=colnames)

층별개요.dtypes


층별개요 = 층별개요[~층별개요['시군구_코드'].isnull()]
층별개요 = 층별개요[~층별개요['법정동_코드'].isnull()]
층별개요 = 층별개요[~층별개요['대지_구분_코드'].isnull()]
층별개요 = 층별개요[~층별개요['번'].isnull()]
층별개요 = 층별개요[~층별개요['지'].isnull()]


층별개요 = 층별개요[['시군구_코드', '법정동_코드', '대지_구분_코드', '번', '지', '주_용도_코드_명', '면적(㎡)']]

start = time.time() #**********오래 걸림
층별개요['고유번호'] = 층별개요.apply(lambda x : x['시군구_코드'] + x['법정동_코드'] + str(int(x['대지_구분_코드'])+1) + x['번'] + x['지'], axis=1)
print("time :", time.time() - start)

층별개요 = 층별개요.drop(['시군구_코드', '법정동_코드', '대지_구분_코드', '번', '지'], axis=1)

층별개요['면적(㎡)'] = 층별개요['면적(㎡)'].astype('float')

층별개요 = 층별개요.groupby(['고유번호', '주_용도_코드_명'])['면적(㎡)'].sum().reset_index()
층별개요 = 층별개요.rename(columns={'면적(㎡)' : '층면적'})

층별개요.to_csv('data/results/층별개요.txt', sep='|', encoding='cp949', index=False)

#%%
"""독립변수 만들기"""
생활인구_x1 = pd.pivot_table(생활_지적, index='TOT_REG_CD', columns='지목', values='면적', aggfunc='sum').fillna(0.0) #집계구x지목으로 피봇
생활인구_x1['전체면적'] = 생활인구_x1[생활인구_x1.columns[1:]].apply(lambda x: sum(x), axis=1) #전체 면적 합계
생활인구_x1[생활인구_x1.columns[1:-1]] = 생활인구_x1[생활인구_x1.columns[1:]].apply(lambda x: x[:-1] / x[-1], axis=1) #전체면적으로 나눔 /비율


층별개요_매칭 = pd.merge(층별개요, 생활_지적[['고유번호', 'TOT_REG_CD', '면적', '원래면적']], how='left', left_on='고유번호', right_on='고유번호')
층별개요_매칭['비율'] = 층별개요_매칭['면적'] / 층별개요_매칭['원래면적']
층별개요_매칭['층면적'] = 층별개요_매칭['층면적'] * 층별개요_매칭['비율']


생활인구_x2 = pd.pivot_table(층별개요_매칭, index='TOT_REG_CD', columns='주_용도_코드_명', values='층면적', aggfunc='sum').fillna(0.0) #집계구x지목으로 피봇
생활인구_x2 = pd.concat([생활인구_x2, 생활인구_x1['전체면적']], axis=1) #독립변수1에서 계산한 토지 전체 면적 가져오기

생활인구_x2[생활인구_x2.columns[1:-1]] = 생활인구_x2[생활인구_x2.columns[1:]].apply(lambda x: x[:-1] / x[-1], axis=1) #전체면적으로 나눔 /비율
생활인구_x2 = 생활인구_x2.drop('전체면적', axis=1)

g_생활인구.dtypes
생활인구_y = g_생활인구[['TOT_REG_CD', '18년6월', '21년6월', '인구차이']]
생활인구_y = 생활인구_y.set_index('TOT_REG_CD')


학습데이터셋 =  pd.concat([생활인구_x1, 생활인구_x2, 생활인구_y], axis=1)

학습데이터셋.to_csv('data/생활인구_학습데이터.txt', sep='|', encoding='cp949', index=True)

#%%
tmp = 생활인구_x1.sum().reset_index()
tmp = 생활인구_x2.sum().reset_index()

g_생활인구 = g_생활인구.merge(생활인구_x1.reset_index()[['TOT_REG_CD', '공']], how='left', left_on='TOT_REG_CD', right_on='TOT_REG_CD')
g_생활인구 = g_생활인구.merge(생활인구_x2.reset_index()[['TOT_REG_CD', '아파트', '사무소', '고시원']], how='left', left_on='TOT_REG_CD', right_on='TOT_REG_CD')




fig, ax = plt.subplots(1, 1, figsize=(20,15))
g_생활인구.plot('공', ax=ax, cmap='Greens', legend=True, vmax=0.5)

fig, ax = plt.subplots(1, 1, figsize=(20,15))
g_생활인구.plot('아파트', ax=ax, cmap='Reds', legend=True, vmax=5)

fig, ax = plt.subplots(1, 1, figsize=(20,15))
g_생활인구.plot('사무소', ax=ax, cmap='Blues', legend=True, vmax=1.5)

fig, ax = plt.subplots(1, 1, figsize=(20,15))
g_생활인구.plot('고시원', ax=ax, cmap='Oranges', legend=True, vmax=0.2)

