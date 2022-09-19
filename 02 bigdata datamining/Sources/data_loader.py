"""
@author: Kloud

"""
import pandas as pd
import geopandas as gpd
import os, sys, re
from glob import glob
from tqdm import tqdm
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import contextily as cx #xyz 타일맵 호출하는 라이브러리
from xyzservices import TileProvider #타일맵 소스를 네이버로 변경하기 위해 클래스

mpl.rc('font', family='NanumGothic') #한글 폰트 적용시
os.chdir('02 Decision Tree/')

"""
데이터 : https://www.dropbox.com/s/a6ogktjetbbsrnw/data.zip?dl=0
"""

#%%

# %%
"""sdot 위치정보와 공개 자료 가져오기"""
sdot_loc = pd.read_excel('data/(공개용)도시데이터센서(S-DoT)_환경정보_설치_위치정보_210615.xlsx', sheet_name='S-DoT 설치 위치 정보',
                         skiprows=3, header=[0, 1, 2], dtype='str')

cols = list(map(lambda x: tuple(map(lambda y: '' if 'Unnamed' in y else y, x)), sdot_loc.columns))
sdot_loc.columns = list(map(lambda x: ''.join(x), cols))

sdot_list = glob('data/sdot/*.csv')
sdot = pd.read_csv(sdot_list[0], sep=',', encoding='cp949', dtype='str', index_col=False)

for file in tqdm(sdot_list[1:]) :
    sdot = pd.concat([sdot, pd.read_csv(file, sep=',', encoding='cp949', dtype='str', index_col=False)])

# del sdot_list, cols

# %%
"""위치정보를 geopandas point 로 변경"""
gsdot = gpd.GeoDataFrame(
    sdot_loc[['시리얼번호', '주소', '높이']], geometry=gpd.points_from_xy(sdot_loc['경도'], sdot_loc['위도']))

gsdot = gsdot.set_crs('EPSG:4326')  # WGS84
gsdot = gsdot.to_crs('EPSG:5174')

ax = gsdot.plot('높이', figsize=(15, 15), cmap='Greens')
ax.set_axis_off()
plt.show()



# %%
""" 시간대별 온도 변화 확인"""
sdot['기온(℃)'] = sdot['기온(℃)'].astype('float')
sdot['Date'] = sdot['전송시간'] #.apply(lambda x: pd.to_datetime(x, format='%Y%m%d%H%M', errors='ignore'))
sdot = sdot[['시리얼', '전송시간', 'Date', '기온(℃)']]

#데이터  피봇
sdot_pivot = pd.pivot_table(sdot, index='시리얼', columns='Date', values='기온(℃)', aggfunc='mean')
sdot_pivot = sdot_pivot.fillna(-40.0)
sdot_pivot.columns = [pd.to_datetime(x, format='%Y%m%d%H%M', errors='ignore') for x in sdot_pivot.columns]

#온도 정보가 35회 이상 누락된 센서 제거
tmp = sdot_pivot[sdot_pivot == -40].count(axis=1)
tmpcnt = tmp.value_counts().reset_index()
tmp = tmp.reset_index()

sdot_pivot = sdot_pivot[sdot_pivot.index.isin(tmp[tmp[0] < 35].시리얼.values)]

#온도 정보가 1개 이상 누락된 시간의 센서 정보는 삭제
tmp = sdot_pivot[sdot_pivot == -40].count()
tmp = tmp.reset_index()

sdot_pivot = sdot_pivot[tmp[tmp[0] == 0]['index'].values]


#geo 데이터도 축소
gsdot = gsdot[gsdot['시리얼번호'].isin(sdot_pivot.index.values)]


gsdot_temp = gsdot.merge(sdot_pivot.reset_index(), how='inner', left_on='시리얼번호', right_on='시리얼')

vmax = 40  # max(sdot_pivot.max())
vmin = 10  # min(sdot_pivot.min())

for t in gsdot_temp.columns[6:]:
    ax = gsdot_temp.plot(t, figsize=(15, 15), cmap='coolwarm', legend=True, vmax=vmax, vmin=vmin)
    ax.set_title(t, fontsize=20)
    plt.show()
    q = input('멈춤(q):')
    if q == 'q': break

# del sdot_pivot, vmax, vmin, ax, t, q
# %%



# %%
"""연속지적도를 불러온다 """  # **********오래 걸림
연속지적 = gpd.read_file('data/AL_11_D002_20210904.shp', encoding='cp949')
연속지적 = 연속지적.rename(
    columns={'A0': 'code', 'A1': '고유번호', 'A2': '법정동코드', 'A3': '법정동명', 'A4': '지번', 'A5': '지목', 'A6': '날짜'})
연속지적 = 연속지적.set_crs('EPSG:5174')
연속지적['지목'] = 연속지적['지목'].apply(lambda x: re.sub('[0-9\- ]', '', x))
tmp = 연속지적.head(10)

"""sdot 포인트에 반경 200m 원 버퍼를 생성"""
gsdot_buffer = gpd.GeoDataFrame(gsdot[['시리얼번호', '주소']], geometry=gsdot.buffer(200))

""" 결과 표시 지적도는 역삼동만 표시 느려서"""
ax = 연속지적[연속지적['법정동코드'] == '1168010100'].plot('지목', figsize=(15, 15), cmap='Greys')
ax = gsdot_buffer[gsdot_buffer['주소'].str.contains('역삼동')].plot(ax=ax, alpha=0.5, color='red')
plt.show()

""" 200m 버퍼 폴리곤과 연속지적도 폴리곤 교집합만 남김"""  # **********오래 걸림
gsdot_지적 = gpd.overlay(gsdot_buffer, 연속지적, how='intersection')

""" 각 필지별 참여 비율 계산"""
연속지적['면적'] = 연속지적.geometry.area
gsdot_지적['면적'] = gsdot_지적.geometry.area

필지면적 = 연속지적[['고유번호', '면적']]
필지면적 = 필지면적.set_index('고유번호')

gsdot_지적['원래면적'] = gsdot_지적['고유번호'].apply(lambda x: 필지면적.loc[x])
tmp = gsdot_지적.head(10)

""" 쉐이프로 저장"""
# gsdot_지적.to_file('data/sdot_지적.shp', encoding='cp949')

# del ax, gsdot_buffer, 연속지적, 필지면적


# %%
# gsdot_지적 = gpd.read_file('sdot_지적.shp', encoding='cp949')
# gsdot_지적.set_crs('EPSG:5174')
gsdot[gsdot['주소'].str.contains('역삼동')]

tmp = gsdot_지적.head(10)
tmp = gsdot_지적['시리얼번호'].value_counts().reset_index()
tmp = gsdot_지적[gsdot_지적['시리얼번호'] == 'OC3CL200020']  # 역삼동

gsdot_지적['시리얼번호']

gsdot_지적[gsdot_지적['시리얼번호'] == 'OC3CL200012'].plot('지목', figsize=(15, 15), cmap='Reds')
plt.show()

#%%
"""건축물대장 표제부"""  # **********오래 걸림


# def file_filter_cols(filepath, sep, columns, dest_filepath):
#     f = open(filepath, 'r')
#     d = open(dest_filepath, 'w')
#
#     max_l = len(open(filepath).readlines())
#
#     for l in tqdm(range(max_l)):
#         line = f.readline()
#         data = line.split(sep)
#         if len(data) < max(columns):
#             continue
#         d.writelines(sep.join([data[idx] for idx in columns]) + '\n')
#
#     f.close()
#     d.close()
#
#
# file_filter_cols('mart_djy_03.txt', '|', [8, 9, 10, 11, 12, 26, 29, 32, 35, 38, 42, 43, 60], '요약_표제부.txt')

# %%
# colnames = ['시군구코드', '법정동코드', '대지구분', '본번', '부번', '건축면적', '지상연면적', '구조', '주용도', '지붕', '높이', '지상층수','사용승인일']
# start = time.time()
# 표제부 = pd.read_csv('요약_표제부.txt', sep='|', encoding='cp949', dtype='str', names=colnames)
# print("time :", time.time() - start)
# 표제부['대지구분'].value_counts()

# 표제부 = 표제부[~표제부['시군구코드'].isnull()]
# 표제부 = 표제부[~표제부['법정동코드'].isnull()]
# 표제부 = 표제부[~표제부['대지구분'].isnull()]
# 표제부 = 표제부[~표제부['본번'].isnull()]
# 표제부 = 표제부[~표제부['부번'].isnull()]



# 표제부 = 표제부[표제부['시군구코드'].str[:2] == '11']
# 표제부.to_csv('data/요약_표제부.txt', sep='|', encoding='cp949')

표제부 = pd.read_csv('data/요약_표제부.txt', sep='|', encoding='cp949', dtype='str')
start = time.time() #**********오래 걸림
표제부['고유번호'] = 표제부.apply(lambda x : x['시군구코드'] + x['법정동코드'] + str(int(x['대지구분'])+1) + x['본번'] + x['부번'], axis=1)
print("time :", time.time() - start)
표제부 = 표제부.drop(['시군구코드', '법정동코드', '대지구분', '본번', '부번'], axis=1)

start = time.time()
sdot_건축물 = pd.merge(gsdot_지적[['시리얼번호', '고유번호', '면적', '원래면적']],
                    표제부, how='inner', left_on='고유번호', right_on='고유번호')
print("time :", time.time() - start)

sdot_건축물['건축면적'] = sdot_건축물['건축면적'].astype('float')
sdot_건축물['지상연면적'] = sdot_건축물['지상연면적'].astype('float')
sdot_건축물['높이'] = sdot_건축물['높이'].astype('float')
sdot_건축물['지상층수'] = sdot_건축물['지상층수'].astype('int')

# sdot_건축물.to_csv('sdot_건축물.csv', sep='|', encoding='cp949', index=False)
# %%
# sdot_건축물 = pd.read_csv('sdot_건축물.csv', sep='|', encoding='cp949')


"""독립변수 만들기"""
sdot_x1 = pd.pivot_table(gsdot_지적, index='시리얼번호', columns='지목', values='면적', aggfunc='sum').fillna(0.0)
sdot_x1['전체면적'] = sdot_x1[sdot_x1.columns[1:]].apply(lambda x: sum(x), axis=1)

tmp = sdot_x1['전체면적'].reset_index()
gsdot_지적[gsdot_지적['시리얼번호'] == 'OC3CL200026'].plot('지목', figsize=(15, 15))
sdot_loc[sdot_loc['시리얼번호'] == 'OC3CL200026']['주소']
gsdot_지적[gsdot_지적['시리얼번호'] == 'OC3CL200026']['고유번호']

연속지적[연속지적['법정동코드'] == '1150010500'].plot('지목', figsize=(15, 15))

gsdot_지적[gsdot_지적['시리얼번호'] == 'OC3CL200131'].plot('지목', figsize=(15, 15))
sdot_loc[sdot_loc['시리얼번호'] == 'OC3CL200131']['주소']
gsdot_지적[gsdot_지적['시리얼번호'] == 'OC3CL200131']['고유번호']
연속지적[연속지적['법정동코드'] == '1174011000'].plot('지목', figsize=(15, 15))

for col in sdot_x1.columns[1:-1]: sdot_x1[col] = sdot_x1[col] / sdot_x1['전체면적']

# %%
"""대상 필지의 버퍼 참여율을 계산"""
sdot_건축물.dtypes
sdot_건축물['비율'] = sdot_건축물['면적'] / sdot_건축물['원래면적']
sdot_건축물['비율_건축면적'] = sdot_건축물['건축면적'] * sdot_건축물['비율']
sdot_건축물['비율_지상연면적'] = sdot_건축물['지상연면적'] * sdot_건축물['비율']

"""사용승인일 결측 및 오류 정제"""
sdot_건축물['사용승인일'] = sdot_건축물['사용승인일'].fillna('19800101')
sdot_건축물['사용승인일'] = sdot_건축물['사용승인일'].apply(lambda x: '19800101' if len(x) != 8 else x)
sdot_건축물['사용승인일'] = sdot_건축물['사용승인일'].apply(lambda x: '19800101' if (x[:2] != '19') & (x[:2] != '20') else x)
sdot_건축물['건축년한'] = sdot_건축물['사용승인일'].apply(lambda x: 2021 - int(x[:4]))

"""독립변수 만들기 합계"""
sdot_x2 = sdot_건축물.groupby('시리얼번호')[['비율_건축면적', '비율_지상연면적']].sum()

sdot_x3 = pd.pivot_table(sdot_건축물, index='시리얼번호', columns='구조', values='비율_지상연면적', aggfunc='sum').fillna(0.0)

sdot_x4 = pd.pivot_table(sdot_건축물, index='시리얼번호', columns='주용도', values='비율_지상연면적', aggfunc='sum').fillna(0.0)

sdot_x5 = pd.pivot_table(sdot_건축물, index='시리얼번호', columns='지붕', values='비율_건축면적', aggfunc='sum').fillna(0.0)

"""독립변수 만들기 평균"""
sdot_건축물['건축년한_비율'] = sdot_건축물['건축년한'] * sdot_건축물['비율_지상연면적']
sdot_건축물['높이_비율'] = sdot_건축물['높이'] * sdot_건축물['비율_지상연면적']
sdot_건축물['지상층수_비율'] = sdot_건축물['지상층수'] * sdot_건축물['비율_지상연면적']

sdot_x6 = sdot_건축물.groupby('시리얼번호')[['건축년한_비율', '높이_비율', '지상층수_비율']].sum()

sdot_x2_6 = pd.concat([sdot_x2, sdot_x6], axis=1)
sdot_x2_6['건축년한'] = sdot_x2_6.apply(lambda x: x['건축년한_비율'] / x['비율_지상연면적'] if x['비율_지상연면적'] > 0 else 0, axis=1)
sdot_x2_6['높이'] = sdot_x2_6.apply(lambda x: x['높이_비율'] / x['비율_지상연면적'] if x['비율_지상연면적'] > 0 else 0, axis=1)
sdot_x2_6['지상층수'] = sdot_x2_6.apply(lambda x: x['지상층수_비율'] / x['비율_지상연면적'] if x['비율_지상연면적'] > 0 else 0, axis=1)

"""높이 오류 발견"""
gsdot_지적[gsdot_지적['시리얼번호'] == 'OC3CL200183'].plot('지목', figsize=(15, 15))
sdot_loc[sdot_loc['시리얼번호'] == 'OC3CL200183']['주소']
tmp = sdot_건축물[sdot_건축물['시리얼번호'] == 'OC3CL200183']

gsdot_지적[gsdot_지적['시리얼번호'] == 'V02Q1940820'].plot('지목', figsize=(15, 15))
sdot_loc[sdot_loc['시리얼번호'] == 'V02Q1940820']['주소']
tmp = sdot_건축물[sdot_건축물['시리얼번호'] == 'V02Q1940820']

""" 변환"""
sdot_건축물[sdot_건축물['고유번호'] == '1150010100102650003']['높이']
sdot_건축물.iloc[46462, 9] = 34.697
# %%
"""최종 입력 테이블 만들기"""
sdot_x_total = pd.concat([sdot_x1, sdot_x2_6, sdot_x3, sdot_x4, sdot_x5], axis=1)
sdot_x_total = sdot_x_total[~sdot_x_total['높이'].isnull()]

sdot_x_total.to_csv('sdot입력변수.csv', sep='|', encoding='cp949')

tmp = sdot_x_total.dtypes.reset_index()
# %%
sdot_y = gsdot_temp[gsdot_temp.columns[5:-1]]
sdot_y = pd.concat([gsdot_temp['시리얼번호'], sdot_y], axis=1)
sdot_y = sdot_y.set_index('시리얼번호')
temp_mean = sdot_y.mean()

sdot_y_비율 = sdot_y / temp_mean
sdot_y_온도 = sdot_y - temp_mean

sdot_y1 = sdot_y_온도.apply(lambda x: x.mean(), axis=1)
sdot_y2 = sdot_y_비율.apply(lambda x: x.mean(), axis=1)

sdot_y_total = pd.concat([sdot_y1, sdot_y2], axis=1)
sdot_y_total = sdot_y_total.rename(columns={0: '온도차이', 1: '온도비율차이'})

"""입력변수와 종속변수 합치기"""
sdot_data_total = pd.concat([sdot_y_total, sdot_x_total], axis=1)
sdot_data_total = sdot_data_total[~sdot_data_total['온도차이'].isnull()]

sdot_data_total = sdot_data_total.drop('', axis=1)

sdot_data_total.to_csv('sdot학습데이터.csv', sep='|', encoding='cp949')

"""오류 발견 """
sdot[sdot['시리얼'] == 'OC3CL200011']
gsdot_지적[gsdot_지적['시리얼번호'] == 'OC3CL200011'].plot('지목', figsize=(15, 15))
sdot_loc[sdot_loc['시리얼번호'] == 'OC3CL200011']['주소']
tmp = sdot_건축물[sdot_건축물['시리얼번호'] == 'OC3CL200011']

"""경기도 지우기"""
sdot_loc[sdot_loc['주소'].str.contains('경기도')]

sdot_y_total.loc['OC3CL200010']
sdot_y_total.loc['OC3CL200011']
sdot_y_total.loc['OC3CL200061']

sdot_y_total = sdot_y_total.drop('OC3CL200010', axis=0)
sdot_y_total = sdot_y_total.drop('OC3CL200011', axis=0)
sdot_y_total = sdot_y_total.drop('OC3CL200061', axis=0)