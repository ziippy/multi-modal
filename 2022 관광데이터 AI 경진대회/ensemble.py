import pandas as pd
import os

BASE_PATH = '/home/jovyan/playGround/ziippy/dl_data/open_sightseeing/'

sample_submission = pd.read_csv(BASE_PATH + 'sample_submission.csv')

value_submissions = [
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
    
    # BASE_PATH + '_old/my_submission_a5_0-3.csv',
    # BASE_PATH + '_old/my_submission_a6_0-2.csv',
    # BASE_PATH + 'my_submission_b7.csv',
    # BASE_PATH + 'my_submission_d1_d2_d3_d6_d7_221031_vote.csv',
                    ]

OUTPUT_FILE_PATH = BASE_PATH + 'my_submission_final2_vote.csv'

dfs = []
for value_submission in value_submissions:
    df = pd.read_csv(value_submission)
    dfs.append(df)

# changed_count = 0
using_vote_count = 0
using_half_half_count = 0
using_value_count = 0
for i, row in sample_submission.iterrows():
    max_index = 0
    max_cat3 = None
    max_value = 0
    #
    using_vote = True
    using_max_value = False
    using_sum_value = False
    
    result_dict = {}
    result_max_value_dict = {}
    result_sum_value_dict = {}
    
    # 일단 값 저장
    for index in range(len(value_submissions)):
        temp_cat3 = dfs[index].loc[i,'cat3']
        temp_value = dfs[index].loc[i,'cat3_value']
        if temp_cat3 not in result_dict.keys():
            # 득표
            result_dict[temp_cat3] = 1
            # max_value 저장
            result_max_value_dict[temp_cat3] = temp_value
            # sum_value 저장
            result_sum_value_dict[temp_cat3] = temp_value
        else:
            # 득표 누적
            result_dict[temp_cat3] += 1
            # max_value 갱신
            if temp_value > result_max_value_dict[temp_cat3]:
                result_max_value_dict[temp_cat3] = temp_value
            # sum_value 누적
            result_sum_value_dict[temp_cat3] += temp_value

    #using_vote = False
    #using_max_value = True
    
    # 다수결의 원칙 우선
    if using_vote is True:
        # 만약 result_dict 의 크기가 len(value_submissions) 이라면, 다수결로 판단할 게 없다는 뜻이므로. 값으로 판단하자.
        if len(result_dict) == len(value_submissions):
            using_max_value = True
            # using_sum_value = True
        else:
            sorted_dict = sorted(result_dict.items(), key = lambda item: item[1], reverse = True)
            # print(sorted_dict)
            final_iter = next(iter(sorted_dict))
            final_cat3 = final_iter[0]
            final_cat3_vote = int(final_iter[1])
            #
            # using_vote_count += 1
            # print(f'VOTE > row: {i}, cat3: {final_cat3}')
            #sample_submission.loc[i,'cat3'] = final_cat3
            # 만약 2:2 인 경우에는 둘 중에 max 사용
            if len(result_dict) == 2 and final_cat3_vote == 2:
                #using_max_value = True
                #using_half_half_count += 1
                using_vote_count += 1
                #
                # print(sorted_dict)
                # sorted_dict = sorted(result_max_value_dict.items(), key = lambda item: item[1], reverse = True)
                # print(sorted_dict)
                # final_iter = next(iter(sorted_dict))
                # final_cat3 = final_iter[0]
                #
                # sorted_dict = sorted(result_sum_value_dict.items(), key = lambda item: item[1], reverse = True)
                # print(sorted_dict)
                # final_iter = next(iter(sorted_dict))
                # final_cat3 = final_iter[0]
                final_iter = next(iter(result_dict))
                final_cat3 = final_iter
                #
                #print(f'half-half VOTE > row: {i}, cat3: {final_cat3}')
                sample_submission.loc[i,'cat3'] = final_cat3
            else:
                using_vote_count += 1
                #            
                # print(f'VOTE > row: {i}, cat3: {final_cat3}')
                sample_submission.loc[i,'cat3'] = final_cat3
    
    # 가장 높은 값 사용
    # 위에서 만든 result_max_value_dict 를 이용하자.
    if using_max_value is True:
        using_value_count += 1
        #
        sorted_dict = sorted(result_max_value_dict.items(), key = lambda item: item[1], reverse = True)
        # print(sorted_dict)
        final_iter = next(iter(sorted_dict))
        final_cat3 = final_iter[0]
        sample_submission.loc[i,'cat3'] = final_cat3
        # print(f'using max > row: {i}, class3: {final_cat3}')
        
    # 합쳐서 가장 높은 값 사용
    # 위에서 만든 result_sum_value_dict 를 이용하자.
    if using_sum_value is True:
        using_value_count += 1
        #
        sorted_dict = sorted(result_sum_value_dict.items(), key = lambda item: item[1], reverse = True)
        # print(sorted_dict)
        final_iter = next(iter(sorted_dict))
        final_cat3 = final_iter[0]
        sample_submission.loc[i,'cat3'] = final_cat3
        print(f'using sum max > row: {i}, class3: {final_cat3}')
        
#     if using_max_value_vote is True:
#         using_max_value_count += 1
#         for index in range(len(value_submissions)):
#             temp_cat3 = dfs[index].loc[i,'cat3']
#             temp_value = dfs[index].loc[i,'cat3_value']
#             if temp_value > max_value:
#                 max_index = index
#                 max_value = temp_value
#                 if max_cat3 is None:
#                     max_cat3 = temp_cat3
#                 else:
#                     if max_cat3 != temp_cat3:
#                         # print(f'CHANGED > row: {i}, max_index: {max_index}, max_cat3: {max_cat3} -> {temp_cat3}')
#                         max_cat3 = temp_cat3
#                         changed_count += 1
#    
#         final_cat3 = dfs[max_index].loc[i,'cat3']
#         sample_submission.loc[i,'cat3'] = dfs[max_index].loc[i,'cat3']        
#         print(f'row: {i}, max_index: {max_index}, class3: {final_cat3}')

sample_submission.to_csv(OUTPUT_FILE_PATH,index=False)

print(f'using_vote_count: {using_vote_count}')
print(f'using_half_half_count: {using_half_half_count}')
print(f'using_value_count: {using_value_count}')
# print(f'changed_count: {changed_count}')
print('ensemble done')