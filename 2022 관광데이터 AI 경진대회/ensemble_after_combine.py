import pandas as pd
import os
import numpy as np

arr = ['5일장', 'ATV', 'MTB', '강', '게스트하우스', '계곡', '고궁', '고택', '골프', '공연장',
       '공예,공방', '공원', '관광단지', '국립공원', '군립공원', '기념관', '기념탑/기념비/전망대',
       '기암괴석', '기타', '기타행사', '농.산.어촌 체험', '다리/대교', '대중콘서트', '대형서점',
       '도립공원', '도서관', '동굴', '동상', '등대', '래프팅', '면세점', '모텔', '문', '문화관광축제',
       '문화원', '문화전수시설', '뮤지컬', '미술관/화랑', '민물낚시', '민박', '민속마을', '바/까페',
       '바다낚시', '박람회', '박물관', '발전소', '백화점', '번지점프', '복합 레포츠', '분수', '빙벽등반',
       '사격장', '사찰', '산', '상설시장', '생가', '서비스드레지던스', '서양식', '섬', '성',
       '수련시설', '수목원', '수상레포츠', '수영', '스노쿨링/스킨스쿠버다이빙', '스카이다이빙', '스케이트',
       '스키(보드) 렌탈샵', '스키/스노보드', '승마', '식음료', '썰매장', '안보관광', '야영장,오토캠핑장',
       '약수터', '연극', '영화관', '온천/욕장/스파', '외국문화원', '요트', '윈드서핑/제트스키',
       '유람선/잠수함관광', '유명건물', '유스호스텔', '유원지', '유적지/사적지', '이색거리', '이색찜질방',
       '이색체험', '인라인(실내 인라인 포함)', '일반축제', '일식', '자동차경주', '자연생태관광지',
       '자연휴양림', '자전거하이킹', '전문상가', '전시관', '전통공연', '종교성지', '중식', '채식전문점',
       '카약/카누', '카지노', '카트', '컨벤션', '컨벤션센터', '콘도미니엄', '클래식음악회', '클럽',
       '터널', '테마공원', '트래킹', '특산물판매점', '패밀리레스토랑', '펜션', '폭포', '학교', '한식',
       '한옥스테이', '항구/포구', '해수욕장', '해안절경', '헬스투어', '헹글라이딩/패러글라이딩', '호수',
       '홈스테이', '희귀동.식물']

BASE_PATH = '/home/jovyan/playGround/ziippy/dl_data/open_sightseeing/'

sample_submission = pd.read_csv(BASE_PATH + 'sample_submission.csv')

value_submissions = [# BASE_PATH + '_old/my_best_a5_fold_0_e21_with_all.csv',
                     # BASE_PATH + '_old/my_best_a5_fold_1_e22_with_all.csv',
                     # BASE_PATH + '_old/my_best_a5_fold_2_with_all.csv',
                     # BASE_PATH + '_old/my_best_a5_fold_3_with_all.csv',
    BASE_PATH + 'my_best_d1_fold_0_with_value.csv',
    BASE_PATH + 'my_best_d1_fold_1_with_value.csv',
    BASE_PATH + 'my_best_d1_fold_2_with_value.csv',
    BASE_PATH + 'my_best_d1_fold_3_with_value.csv',
    BASE_PATH + 'my_best_d1_fold_4_with_value.csv',
    
    BASE_PATH + 'my_best_d2_fold_0_with_value.csv',
    BASE_PATH + 'my_best_d2_fold_1_with_value.csv',
    BASE_PATH + 'my_best_d2_fold_2_with_value.csv',
    BASE_PATH + 'my_best_d2_fold_3_with_value.csv',
    BASE_PATH + 'my_best_d2_fold_4_with_value.csv',
    
    BASE_PATH + 'my_best_d3_fold_0_with_value.csv',
    BASE_PATH + 'my_best_d3_fold_1_with_value.csv',
    BASE_PATH + 'my_best_d3_fold_2_with_value.csv',
    BASE_PATH + 'my_best_d3_fold_3_with_value.csv',
    BASE_PATH + 'my_best_d3_fold_4_with_value.csv',
    
    BASE_PATH + 'my_best_d5_fold_0_with_value.csv',
    BASE_PATH + 'my_best_d5_fold_1_with_value.csv',
    BASE_PATH + 'my_best_d5_fold_2_with_value.csv',
    BASE_PATH + 'my_best_d5_fold_3_with_value.csv',
    BASE_PATH + 'my_best_d5_fold_4_with_value.csv',
    
    BASE_PATH + 'my_best_d6_fold_0_with_value.csv',
    BASE_PATH + 'my_best_d6_fold_1_with_value.csv',
    BASE_PATH + 'my_best_d6_fold_2_with_value.csv',
    BASE_PATH + 'my_best_d6_fold_3_with_value.csv',
    BASE_PATH + 'my_best_d6_fold_4_with_value.csv',
    
    BASE_PATH + 'my_best_d7_fold_0_with_value.csv',
    BASE_PATH + 'my_best_d7_fold_1_with_value.csv',
    BASE_PATH + 'my_best_d7_fold_2_with_value.csv',
    BASE_PATH + 'my_best_d7_fold_3_with_value.csv',
    BASE_PATH + 'my_best_d7_fold_4_with_value.csv',
                    ]

#OUTPUT_FILE_PATH = BASE_PATH + 'my_submission_d1_d2_d3_d6_d7_221031_soft_vote.csv'
OUTPUT_FILE_PATH = BASE_PATH + 'my_submission_final3_vote.csv'

dataFrame = pd.concat(
   map(pd.read_csv, value_submissions), ignore_index=True)

dataFrame = dataFrame.groupby(dataFrame['id']).mean()
dataFrame.drop('cat3_value', inplace=True, axis=1)
# print(dataFrame)

maxValueIndex  = dataFrame.idxmax(axis=1)
cat3 = [arr[int(x)] for x in maxValueIndex]
#print(cat3)

sample_submission['cat3'] = cat3
sample_submission.to_csv(OUTPUT_FILE_PATH,index=False)

print('ensemble_after_combine done')
