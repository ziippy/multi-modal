import pandas as pd
import os

BASE_PATH = '/home/jovyan/playGround/ziippy/dl_data/open_sightseeing/'

sample_submission = pd.read_csv(BASE_PATH + 'sample_submission.csv')

value_submissions = [
    BASE_PATH + '_old/my_submission_a5_0-3.csv',
    # BASE_PATH + 'my_submission_d1_d2_d3_d5_d6_221030_vote.csv',
    # BASE_PATH + '_old/my_submission_a5_0-3_221027_vote_2.csv',
    #BASE_PATH + '_old/my_submission_a6_0-2.csv',
    BASE_PATH + 'my_submission_b7.csv',
    BASE_PATH + 'my_submission_d1_d2_d3_d6_d7_221031_vote.csv',
                    ]

OUTPUT_FILE_PATH = BASE_PATH + 'my_submission_final2_vote.csv'

dfs = []
for value_submission in value_submissions:
    df = pd.read_csv(value_submission)
    dfs.append(df)

# changed_count = 0
using_vote_count = 0
using_first_count = 0

for i, row in sample_submission.iterrows():
    max_index = 0
    max_cat3 = None
    max_value = 0
    #   
    result_dict = {}
    
    
    
    # 일단 값 저장
    for index in range(len(value_submissions)):
        temp_cat3 = dfs[index].loc[i,'cat3']
        if temp_cat3 not in result_dict.keys():
            # 득표
            result_dict[temp_cat3] = 1
        else:
            # 득표 누적
            result_dict[temp_cat3] += 1
    
    # 다수결의 원칙 우선
    
    # 만약 result_dict 의 크기가 len(value_submissions) 이라면, 다수결로 판단할 게 없다는 뜻이므로. 0 번째 항목 사용
    # print(result_dict)
    if len(result_dict) == len(value_submissions):
        using_first_count += 1
        #
        print(result_dict)
        final_iter = next(iter(result_dict))
        final_cat3 = final_iter
        #
        print(f'FIRST > row: {i}, cat3: {final_cat3}')
        sample_submission.loc[i,'cat3'] = final_cat3
        

    else:
        using_vote_count += 1
        #
        sorted_dict = sorted(result_dict.items(), key = lambda item: item[1], reverse = True)
        # print('sorted: ', sorted_dict)
        final_iter = next(iter(sorted_dict))
        final_cat3 = final_iter[0]
        final_cat3_vote = int(final_iter[1])
        #
        # print(f'VOTE > row: {i}, cat3: {final_cat3}')
        sample_submission.loc[i,'cat3'] = final_cat3
    
sample_submission.to_csv(OUTPUT_FILE_PATH,index=False)

print(f'using_vote_count: {using_vote_count}')
print(f'using_first_count: {using_first_count}')
print('ensemble done')
