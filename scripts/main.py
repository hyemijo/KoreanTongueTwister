import filereader, preprocess, analyze, mystats
import numpy as np
from collections import Counter, defaultdict
import unicodedata
import pandas as pd
from scipy import stats


filename_phon = "data_min.txt"
filename_phon_error = "data_min_error.txt"
filename_feature = "korean_feature_table.tsv"
features = ['syll', 'cons', 'son', 'cont', 'asp', 'tense', 'nas', 'spread', 
            'lab', 'cor', 'ant', 'strid', 'dors', 'high', 'low', 'back', 'round']


# 1. get data
data_sent = filereader.get_data_raw(filename_phon)
data_phon_list = filereader.get_data_phon(filename_phon) # 문장별 발음 음소 단위 전사 결과물.
data_phon_featurized_list = filereader.get_data_phon_featurized(filename_phon, filename_feature) # 음소 -> 자질의 다발로 변환
data_error_list = filereader.get_data_error_tagged(filename_phon_error, data_phon_featurized_list)  #data_phon_featurized_list에 error 정보 태깅. error 리스트 반환.
data_feature = filereader.get_data_feature_dict(filename_feature) # feature table

#전사 결과물
"""for sent in data_phon_featurized_list:
    for syl in sent:
        for phon in syl:
            print(phon.jamo, end = "")
        print(end= " ")
    print()"""


#======================================================
# 문장별 onset, coda 타입 수, 토큰 수 집계
onsets_data = []
codas_data = []
onsets_nums = []
codas_nums = []
for sent in data_phon_featurized_list:
    onsets_type = set()
    codas_type = set()
    onsets_num = 0
    codas_num = 0

    for syl in sent:
        for phon in syl:
            if phon.syl_str_info == "onset":
                onsets_type.add(phon.jamo)
                onsets_num += 1
            elif phon.syl_str_info == "coda":
                codas_type.add(phon.jamo)
                codas_num += 1
            else:
                continue
    
    onsets_data.append(onsets_type)
    codas_data.append(codas_type)
    onsets_nums.append(onsets_num)
    codas_nums.append(codas_num)
    
#print("onset")
onset_type = [len(i) for i in onsets_data] #sum([len(i) for i in onsets_data])) #type
#print(onsets_nums, sum(onsets_nums)) #token
onset_token_type = [ round(onsets_nums[i] / len(num), 2) for i, num in enumerate(onsets_data)]


#print("coda")
coda_type = [len(i) for i in codas_data] #sum([len(i) for i in codas_data]))
#print(codas_nums, sum(codas_nums))
coda_token_type = [ round(codas_nums[i] / len(num), 2) for i, num in enumerate(codas_data)]



# 문장별 & 음절 내 위치별 오류 빈도 집계
error_onset_nums = []
error_nucleus_nums = []
error_coda_nums = []
error_sum_nums = []

for sent in data_phon_featurized_list:
    error_onset = 0
    error_nucleus = 0
    error_coda = 0

    for syl in sent:
        for phon in syl:
            if phon.syl_str_info == "onset":
                error_onset += phon.error_freq

            elif phon.syl_str_info == "coda":
                error_coda += phon.error_freq

            else:
                error_nucleus += phon.error_freq
    error_onset_nums.append(error_onset)
    error_nucleus_nums.append(error_nucleus)
    error_coda_nums.append(error_coda)
    error_sum_nums.append(error_onset + error_nucleus + error_coda)

#print(error_onset_nums, sum(error_onset_nums))
#print(error_nucleus_nums)
#print(error_coda_nums, sum(error_coda_nums))
#print(error_sum_nums)
#print(273 / 66)


# 선형회귀분석. token/type ~ error
import numpy as np
from sklearn.linear_model import LinearRegression
x_onset = np.array(onset_token_type).reshape((-1,1))
#x_onset = np.array(onset_type).reshape((-1,1))
#x_onset = np.array(onsets_nums).reshape((-1,1))
y_onset = np.array(error_onset_nums)
x_coda = np.array(coda_token_type).reshape((-1,1))
#x_coda = np.array(coda_type).reshape((-1,1))
#x_coda = np.array(codas_nums).reshape((-1,1))
y_coda = np.array(error_coda_nums)

model_onset = LinearRegression().fit(x_onset, y_onset)
model_coda = LinearRegression().fit(x_coda, y_coda)
r_sq_onset = model_onset.score(x_onset, y_onset)
r_sq_coda = model_coda.score(x_coda,y_coda)
#print(r_sq_onset, r_sq_coda)

# ======================================================
# 2. basic stats
# 오류 집계
#mystats.get_error_freq(data_phon_featurized_list, data_sent) # [2, 8, 1, 10, 1, 2, 1, 30, 12, 3, 3, 2, 1, 2, 4, 1, 24, 10, 2] 119 #print(c.count("null"), len(c)): 탈락 오류 25개.

# onset, nucleus, coda 개수 집계
#mystats.get_syl_str_num(data_phon_featurized_list)
#mystats.get_syl_type_num(data_phon_featurized_list)


# 음절 내 위치별 음소 개수 집계
#mystats.get_phon_num(data_phon_featurized_list)

# target -> error 변화한 feature 집계 (substitution error 대상)

# 주석 해제
#features_shared_data, features_nonshared_data = analyze.get_featured_shared_nonshared(data_phon_featurized_list, data_feature)
#mystats.print_shared_nonshared(features_shared_data, features_nonshared_data, data_feature)


# 변화한/유지된 자질 범주 해석
# target-error 간 공유x 자질 (바뀐 자질) natural class 기준 집계
#print("============changed features: natural class")
#feature_changes = mystats.get_feature_changes(features_nonshared_data)
#print(features_nonshared_data[0])
#print(feature_changes)
#print(Counter(feature_changes))
#print(sum(feature_changes.values()), len(features_nonshared_data))

# target-error가 공유하는 자질의 natural class 기준 집계
#print("============sustained features: natural class")
#print(features_shared_data[0])
#feature_unchanged = mystats.get_feature_unchanged(features_shared_data)
#print(Counter(feature_unchanged))
#print(sum(feature_unchanged.values()), len(features_shared_data))


# ============================================
# trigger 찾기
from copy import deepcopy

trigger_data = analyze.find_trigger_data(data_phon_featurized_list, data_feature)
error_type_dir_syl_str = {"anticipatory":{"onset":0, "nucleus":0, "coda":0},
                  "perseveratory": {"onset":0, "nucleus":0, "coda":0},
                  "bidirectional": {"onset":0, "nucleus":0, "coda":0},
                  "no_trigger": {"onset":0, "nucleus":0, "coda":0}}
error_type_dir_min_dist = {"anticipatory":{"onset":0, "nucleus":0, "coda":0},
                  "perseveratory": {"onset":0, "nucleus":0, "coda":0},
                  "bidirectional": {"onset":0, "nucleus":0, "coda":0},
                  "no_trigger": {"onset":0, "nucleus":0, "coda":0}}


def get_error_type(phon, triggers, sent):
    phon_idx = 0
    trigger_idx = 0
    temp = []
    dist = 0

    if (len(triggers) > 1): # 모두 bidirectional로 간주 가능 (len == 2, temp에 +, -로 담김)
        for t in triggers:
            phon_idx = analyze.find_phon_idx_stream(phon, sent)
            trigger_idx = analyze.find_phon_idx_stream(t, sent)
            dist = trigger_idx - phon_idx
            temp.append(dist)

        if temp[0] * temp[1] < 0:
            return "bidirectional"

    elif (len(triggers) == 0):
        return "no_trigger"

    else:    # trigger 1개인 경우
        t = triggers[0]
        phon_idx = analyze.find_phon_idx_stream(phon, sent)
        trigger_idx = analyze.find_phon_idx_stream(t, sent)
        dist = trigger_idx - phon_idx #if dist == 0: # 이 경우 없었음.
    
        if dist > 0: # dist < 0: trigger가 phon 이후에 등장
            return "anticipatory"
        else:
            return "perseveratory"
    return

for phon, sent, (trigger_syl_str, trigger_min_dist) in trigger_data:
    print("phon", phon.jamo, phon.syl_str_info, phon.syl_idx)
    #print([p.jamo for syl in sent for p in syl])
    # print("syl str", end="\t")
    print("triggers", [(t.jamo, t.syl_idx, t.syl_str_info) for t in trigger_syl_str])
    error_type = get_error_type(phon, trigger_syl_str, sent)
    print("error", error_type)
    error_type_dir_syl_str[error_type][phon.syl_str_info] += 1
    print()        
    # print("min dist", end="\t")
    #for t in trigger_min_dist:
    #    if get_error_type(phon, t, sent):
    #        print(t.jamo, t.syl_str_info, t.syl_idx, end="\t")
    # print()
    # print()

print(error_type_dir_syl_str)



