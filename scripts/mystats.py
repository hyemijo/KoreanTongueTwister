#get statistics
import analyze
from collections import Counter
from scipy import stats
import pandas as pd
from collections import defaultdict

features = ['syll', 'cons', 'son', 'cont', 'asp', 'tense', 'nas', 'spread', 
            'lab', 'cor', 'ant', 'strid', 'dors', 'high', 'low', 'back', 'round']

o_features = features[:-4] # features for onset, coda
n_features = features[:3] + [features[8]] + features[-4:] # features for nucleus

"""
lens = []
for sent in data_phon_list:
    lens.append(len(sent))
print(np.mean(lens))
print(np.median(lens))
"""

def get_error_freq(data_phon_featurized_list, data_sent):
    errors = []
    error_freqs = []
    for i, sent in enumerate(data_phon_featurized_list): 
        error_freq_sent = 0
        sent_syl = [syl for syl in data_sent[i] if syl.strip()]
        for syl in sent:
            for phon in syl:
                if phon.error_freq:
                    error_freq_sent += phon.error_freq
                    for error in phon.error_token:
                        try:
                            errors.append(error.jamo)
                        except:
                            errors.append("null")
        error_freqs.append(error_freq_sent)
    return (error_freqs, sum(error_freqs), errors)


def get_syl_str_num(data_phon_featurized_list):
    num_onset = 0 #495 
    num_nucleus = 0 #548 #반모음 63개. 모음 548개. 총합 611개.
    num_coda = 0 #273
    num_syl = 0 #548 

    for i, sent in enumerate(data_phon_featurized_list): 
        for syl in sent:
            if syl:
                num_syl += 1
                for phon in syl:
                    if (phon.syl_str_info == "onset"):
                        num_onset += 1
                    elif (phon.syl_str_info == "nucleus"): #반모음까지 집계
                    # 모음만 집계(phon.syl_str_info == "nucleus" and phon.jamo not in "jw"):
                        num_nucleus += 1
                    elif (phon.syl_str_info == "coda"):
                        num_coda += 1
                    else:
                        print("error: syl str info")

    print(num_syl, num_onset, num_nucleus, num_coda)
    return 


def syl2syltype(syl):
    syl_type = ""

    if (len(syl) ==1 and syl[0].syl_str_info == "nucleus"):
        syl_type = "V"

    elif (len(syl) ==2 and syl[0].jamo in "jw" and syl[1].syl_str_info == "nucleus"):
        syl_type = "GV"
    elif (len(syl) ==2 and syl[1].jamo in "jw" and syl[0].syl_str_info == "nucleus"):
        syl_type = "GV"
    elif (len(syl) ==2 and syl[0].syl_str_info == "nucleus" and syl[1].syl_str_info == "coda"):    
        syl_type = "VC"
    elif (len(syl) ==2 and syl[0].syl_str_info == "onset" and syl[1].syl_str_info == "nucleus"):    
        syl_type = "CV"

    elif (len(syl) ==3 and syl[0].jamo in "jw" and syl[1].syl_str_info == "nucleus" and syl[2].syl_str_info == "coda"):    
        syl_type = "GVC"
    elif (len(syl) ==3 and syl[0].syl_str_info == "onset" and syl[1].jamo in "jw" and syl[2].syl_str_info == "nucleus"):    
        syl_type = "CGV"
    elif (len(syl) ==3 and syl[0].syl_str_info == "onset" and syl[2].jamo in "jw" and syl[1].syl_str_info == "nucleus"):    
        syl_type = "CGV"
    elif (len(syl) ==3 and syl[0].syl_str_info == "onset" and syl[1].syl_str_info == "nucleus" and syl[2].syl_str_info == "coda"):    
        syl_type = "CVC"
        
    elif (len(syl) ==4 and syl[0].syl_str_info == "onset" and syl[1].jamo in "jw" and syl[2].syl_str_info == "nucleus" and syl[3].syl_str_info == "coda"):    
        syl_type = "CGVC"

    else:
        print(f"error: syl type {analyze.syl2jamos(syl)}")
        print([phon.syl_str_info for phon in syl])
        print()

    return syl_type


def get_syl_type_num(data_phon_featurized_list):
    syltypes = []
    for i, sent in enumerate(data_phon_featurized_list): 
        for syl in sent:
            if syl:
                syltypes.append(syl2syltype(syl))


    syltypes = Counter(syltypes)

    num_v, num_gv, num_cv, num_cgv = syltypes["V"], syltypes["GV"], syltypes["CV"], syltypes["CGV"]
    num_vc, num_gvc, num_cvc, num_cgvc = syltypes["VC"], syltypes["GVC"], syltypes["CVC"], syltypes["CGVC"]

    num_open_syl = num_v + num_gv + num_cv + num_cgv
    num_closed_syl = num_vc + num_gvc + num_cvc + num_cgvc

    print(f"open_syl: {num_open_syl} [num_v: {num_v}, num_gv: {num_gv}, num_cv: {num_cv}, num_cgv: {num_cgv}]")
    print(f"closed_syl: {num_closed_syl} [num_vc: {num_vc}, num_gvc: {num_gvc}, num_cvc: {num_cvc}, num_cgvc: {num_cgvc}]")

    return 


def find_feature_phon(feature_tuples, data_feature):
    phons = []
    for k, v  in data_feature.items():
        if (set(feature_tuples).issubset(set(v.items()))):
            phons.append(k)
    return phons


def get_phon_num(data_phon_featurized_list):
    ## 빈 dataframe 만들기
    df_onset = get_init_df("error", "onset").append(get_init_df("nonerror", "onset"))
    df_nucleus = get_init_df("error", "nucleus").append(get_init_df("nonerror", "nucleus"))
    df_coda = get_init_df("error", "coda").append(get_init_df("nonerror", "coda"))


    ## + 자질, - 자질 따로 저장.
    df_onset_feature_plus = get_init_feature_df("error", "onset").append(get_init_feature_df("nonerror", "onset"))
    df_onset_feature_minus = get_init_feature_df("error", "onset").append(get_init_feature_df("nonerror", "onset"))
    df_nucleus_feature_plus = get_init_feature_df("error", "nucleus").append(get_init_feature_df("nonerror", "nucleus"))
    df_nucleus_feature_minus = get_init_feature_df("error", "nucleus").append(get_init_feature_df("nonerror", "nucleus"))
    df_coda_feature_plus = get_init_feature_df("error", "coda").append(get_init_feature_df("nonerror", "coda"))
    df_coda_feature_minus = get_init_feature_df("error", "coda").append(get_init_feature_df("nonerror", "coda"))

    ## 1. phoneme-level cnt
    count_error_nonerror_freq(df_onset, data_phon_featurized_list, "onset")
    count_error_nonerror_freq(df_nucleus, data_phon_featurized_list, "nucleus")
    count_error_nonerror_freq(df_coda, data_phon_featurized_list, "coda")

    df_onset = add_sum_row(df_onset)
    df_nucleus =add_sum_row(df_nucleus)
    df_coda = add_sum_row(df_coda)

    cont_tables_onset = get_cont_table_phon_error(df_onset)
    cont_tables_nucleus = get_cont_table_phon_error(df_nucleus)
    cont_tables_coda = get_cont_table_phon_error(df_coda)

    print(df_onset)
    print(df_nucleus)
    print(df_coda)

    ## 2. feature-level cnt
    #df_syl_str_info_plus, df_syl_str_info_minus, data_phon_featurized_list, syl_str_info
    count_error_nonerror_feature_freq(df_onset_feature_plus, df_onset_feature_minus, data_phon_featurized_list, "onset")
    count_error_nonerror_feature_freq(df_nucleus_feature_plus, df_nucleus_feature_minus, data_phon_featurized_list, "nucleus")
    count_error_nonerror_feature_freq(df_coda_feature_plus, df_coda_feature_minus, data_phon_featurized_list, "coda")

    df_onset_feature_plus = add_sum_row(df_onset_feature_plus)
    df_onset_feature_minus =add_sum_row(df_onset_feature_minus)
    df_nucleus_feature_plus = add_sum_row(df_nucleus_feature_plus)
    df_nucleus_feature_minus = add_sum_row(df_nucleus_feature_minus)
    df_coda_feature_plus = add_sum_row(df_coda_feature_plus)
    df_coda_feature_minus = add_sum_row(df_coda_feature_minus)

    print(df_onset_feature_plus)
    print(df_onset_feature_minus)
    print(df_nucleus_feature_plus)
    print(df_nucleus_feature_minus)
    print(df_coda_feature_plus)
    print(df_coda_feature_minus)





def print_shared_nonshared(features_shared_data, features_nonshared_data, data_feature):
    print("==============================shared features")
    print("총", len(features_shared_data), "건")

    for k, v in sorted(Counter(features_shared_data).items(), key= lambda x : x[1], reverse=True):
        #공유되는 자질
        print(v, end = "\t")
        for (feature, value) in k:
            print((feature, value), end = "\t")

        #해당 feature 가진 음소 출력
        phons = find_feature_phon(k, data_feature)
        print(phons)

        #print(k, v)

    print("==============================nonshared features")
    print("총", len(features_nonshared_data), "건")
    for k, v in sorted(Counter(features_nonshared_data).items(), key= lambda x : x[1], reverse=True):
        print(v, end = "\t")
        for (feature, value) in k:
            if (value != "0"):
                print((feature, value), end = "\t")
        print()
    print()
    print()



place = ["lab", "cor", "ant", "dors", "strid",  "high", "low", "back", "round"] 
manner = ["cont", "nas", "son"] # strid: manner 말고 place로 분류. (Hayes 기준 따름)
laryngeal = ["asp", "tense", "spread"]
syllabic = ["syll"]


def get_feature_changes(features_nonshared_data):
    feature_changes = defaultdict(int)

    for k, v in sorted(Counter(features_nonshared_data).items(), key= lambda x : x[1], reverse=True):
        features = set()

        for (feature, (target, error)) in k:
            if feature in place:
                features.add("place")
            if feature in manner:
                features.add("manner")
            if feature in laryngeal:
                features.add("laryngeal")
            if feature in syllabic:
                features.add("syllabic")
        
        features = tuple(features)
        feature_changes[features] += v

    return feature_changes


def get_feature_unchanged(features_shared_data):
    feature_unchanged = defaultdict(int)

    for k, v in sorted(Counter(features_shared_data).items(), key= lambda x : x[1], reverse=True):
        features = set()
        shared_features_set = set([f[0] for f in k])

        for (feature, a) in k:
            if set(place).issubset(shared_features_set):
                features.add("place")
            if set(manner).issubset(shared_features_set):
                features.add("manner")
            if set(laryngeal).issubset(shared_features_set):
                features.add("laryngeal")
            if set(syllabic).issubset(shared_features_set):
                features.add("syllabic")
        
        features = tuple(features)
        feature_unchanged[features] += v
    return feature_unchanged




def get_syl_type_dict(data_phon_featurized_list):
    target_is_error_syl = False
    target_syl_type_dict = defaultdict(int)
    data_syl_type_dict = {"onset": defaultdict(int), "nucleus":defaultdict(int), "coda": defaultdict(int)}

    for sent in data_phon_featurized_list:
        sent_syl = [syl for syl in sent if syl] #유효한 음절의 리스트

        for i, syl in enumerate(sent_syl):
            error_freq = 0
            error_freq_onset, error_freq_nucleus, error_freq_coda = 0, 0, 0

            syl_before_target = ""
            syl_after_target = ""
            syl_before_target_syl_type = ""
            syl_after_target_syl_type = ""

            for phon in syl:
                if (phon.error_freq): # 음절에 포함된 음소에서 오류 발생했었으면
                    target_is_error_syl = True # 그 음절은 target 음절
                    error_freq += phon.error_freq

                    # 오류 발생 빈도 세부 집계
                    if (phon.syl_str_info == "onset"):
                        error_freq_onset += phon.error_freq
                    elif (phon.syl_str_info == "nucleus"):
                        error_freq_nucleus += phon.error_freq
                    elif (phon.syl_str_info == "coda"):
                        error_freq_coda += phon.error_freq
                    else:
                        continue

            if (target_is_error_syl):
                target_syl_type = syl2syltype(syl) # target의 syllable type 구하기
                
                # environment: before
                try:
                    syl_before_target = sent_syl[i-1]
                    syl_before_target_syl_type = syl2syltype(syl_before_target)
                except:
                    pass
                
                # environment: after
                try:
                    syl_after_target = sent_syl[i+1]
                    syl_after_target_syl_type = syl2syltype(syl_after_target)
                except:
                    pass

                key = (syl_before_target_syl_type, target_syl_type, syl_after_target_syl_type) #key = target_syl_type
                # 전체 집계
                target_syl_type_dict[key] += 1 #token: target_syl_type_dict[key] += error_freq #type: target_syl_type_dict[target_syl_type] += 1
                
                # 세부 집계
                data_syl_type_dict["onset"][key] += error_freq_onset
                data_syl_type_dict["nucleus"][key] += error_freq_nucleus
                data_syl_type_dict["coda"][key] += error_freq_coda
                
                target_is_error_syl = False

                continue
            else:
                continue
    return target_syl_type_dict, data_syl_type_dict














def get_init_df(row_name, syl_str_info):
    # 빈 df 만드는 함수.
    init_df = dict()
    
    onset = list("ㅂㅍㅃㄷㅌㄸㅈㅊㅉㄱㅋㄲㅅㅆㅎㅁㄴㅇㄹ")
    nucleus = list("jwㅣㅔㅡㅓㅏㅗㅜ")
    coda = list("ㅂㄷㄱㅁㄴㅇㄹ") 

    if (syl_str_info == "onset"):
        for phon in onset:
            init_df[phon] = 0
    elif (syl_str_info== "nucleus"):
        for phon in nucleus:
            init_df[phon] = 0
    elif (syl_str_info== "coda"):
        for phon in coda:
            init_df[phon] = 0
    else:
        return

    init_df = pd.DataFrame([init_df])
    init_df.index = [row_name]

    return init_df


def get_init_feature_df(row_name, syl_str_info):
    #df_onset_feature = mystats.get_init_feature_df("error", "onset").append(mystats.get_init_feature_df("nonerror", "onset"))

    init_feature_df = dict()

    if (syl_str_info == "onset"):
        for f in o_features : #dors까지!
            init_feature_df[f] = 0
    elif (syl_str_info== "nucleus"):
        for f in n_features:
            init_feature_df[f] = 0
    elif (syl_str_info== "coda"):
        for f in o_features : #dors까지!
            init_feature_df[f] = 0
    else:
        return

    init_feature_df = pd.DataFrame([init_feature_df])
    init_feature_df.index = [row_name]

    return init_feature_df


def count_error_nonerror_freq(df_syl_str_info, data_phon_featurized_list, syl_str_info):
    #mystats.count_error_nonerror_freq(df_onset)
    #요소 접근: print(df_onset["ㅂ"]["error"])

    for sent in data_phon_featurized_list:
        for syl in sent:
            for phon in syl:
                #print("phon: ", phon.jamo, phon.syl_str_info, phon.error_freq, phon.error_token)
                if (phon.syl_str_info == syl_str_info and phon.error_freq != 0): # onset. 오류 발생 음절에서 음소 등장 빈도 +1
                    df_syl_str_info[phon.jamo]["error"] += 1
                elif (phon.syl_str_info == syl_str_info and phon.error_freq == 0): # onset. 오류 미발생 음절에서 음소 등장 빈도 +1
                    df_syl_str_info[phon.jamo]["nonerror"] += 1
                else: # onset 아닌 것들. 집계 x.
                    continue
    return


def add_feature_error_freq(phon, df_syl_str_info_plus, df_syl_str_info_minus, row_name, num):
    # 음소의 error_freq != 0 인 경우, df에서 해당 음소의 각 자질 cell에 빈도 +1씩.
    features = df_syl_str_info_plus.columns
    for f_phon, value_phon in phon.__dict__.items():
        #print("f_phon, value_phon: ", f_phon, value_phon)
        if (value_phon == "+"):
            df_syl_str_info_plus[f_phon][row_name] += num
            #print(df_syl_str_info_plus[f_phon][row_name])
        elif (value_phon == "-"):
            df_syl_str_info_minus[f_phon][row_name] += num
            #print(df_syl_str_info_minus[f_phon][row_name])
        else: #value_phon == "0"
            continue
    return

def count_error_nonerror_feature_freq(df_syl_str_info_plus, df_syl_str_info_minus, data_phon_featurized_list, syl_str_info):
    #mystats.count_error_nonerror_freq(df_onset)
    #요소 접근: print(df_onset["ㅂ"]["error"])

    for sent in data_phon_featurized_list:
        for syl in sent:
            for phon in syl:
                #print("phon: ", phon.jamo, phon.error_freq)
                #try:
                if (phon.syl_str_info != syl_str_info):
                    continue
                elif (phon.syl_str_info == syl_str_info and phon.error_freq != 0): # ex. onset. 오류 발생 음절에서 음소 등장 빈도 +1 -> 자질 등장 빈도 +1
                    add_feature_error_freq(phon, df_syl_str_info_plus, df_syl_str_info_minus, "error", 1)        
                elif (phon.syl_str_info == syl_str_info and phon.error_freq == 0): # ex. onset. 오류 미발생 음절에서 음소 등장 빈도 +1 -> 자질 등장 빈도 +1
                    add_feature_error_freq(phon, df_syl_str_info_plus, df_syl_str_info_minus, "nonerror", 1)
                else:
                    continue
                #except:  #  phon의 attribute 중 feature 아닌 것들. syl_idx 등!
                    #pass
    return



def add_sum_row(df):
    # df에 행/열별 합계 덧붙이는 함수.
    df["error_sum"] = df.sum(axis=1)
    df = df.T
    df["phon_sum"] = df.sum(axis=1)
    
    return df


def get_cont_table_phon_error(df):
    #tables = mystats.get_cont_table_phon_error(df_onset)
    tables = []
    row_sum = df.iloc[-1]

    for index, row_phon in df.iterrows():
        if (row_phon == row_sum).all():
            break
        
        row_not_phon = row_sum - row_phon
        table = pd.concat([row_phon, row_not_phon], axis=1)
        table.columns = [table.columns[0], "non-"+table.columns[0]]
        tables.append(table)

    return tables


def print_cont_tables(tables, syl_str_info):
    print(f"=============={syl_str_info}==============")
    for table in tables:
        #https://thduddl2486.tistory.com/181
        #행: 독립변수, 열: 종속변수
        print(table.T)
        print(stats.fisher_exact(table.T, alternative="two-sided"))
        #try:
        #    print(stats.chi2_contingency(table.T))
        #except:
        #    pass

    print()
    print("======================================")
    
    return


def check_freqdicts_phon(freqdicts_phon):
    for i, sent_ngram in enumerate(freqdicts_phon[0]):
        print(f"=========={i+1}th sentence")
        print(sent_ngram)
        print(freqdicts_phon[1][i])
        print(freqdicts_phon[2][i])
        print(freqdicts_phon[3][i])
        print()
        print()
    return


def check_freqdicts_phon_unigram_syl(freqdicts_phon, syl_str):
    for i, sent_ngram in enumerate(freqdicts_phon[0]):
        print(f"======{i+1}th sentence======")
        items = sorted(sent_ngram.items(), key=lambda x : x[1], reverse=True)
        for phon_unigram, freq in items:
            if (phon_unigram[0][1] == syl_str):
                print(phon_unigram[0], freq)
        print()
    return


def get_ngram_ratio(freqdicts_phon):
    ratio = 0
    
    for ngram in freqdicts_phon:
        type_sum = 0
        token_sum = 0
        for sent_ngram in ngram:
            type_sum += len(sent_ngram)
            token_sum += sum(sent_ngram.values())

        ratio += (type_sum / token_sum)
    
    return ratio / 4



def count_len(data_onset):
    sum = 0
    for sent in data_onset:
        sum += len(sent)
    return sum


def get_feature_dicts(data_types, features, data_syl_str):
    feature_dicts = dict()

    for i, data in enumerate(data_syl_str):
        feature_dict = dict()
        for feature in features:
            feature_dict[feature] = analyze.get_ngrams_features(data, feature)
        feature_dicts[data_types[i]] = feature_dict

    return feature_dicts


def count_features(data_type, data_types, features, data_syl_str):
    print("========================")
    print(data_type)

    feature_dicts = get_feature_dicts(data_types, features, data_syl_str)

    for feature in features:
        for i in range(4):
            cnt_ith = Counter(feature_dicts[data_type][feature][i])
            print(f"{feature}: {cnt_ith})")

    return