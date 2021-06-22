from typing import final
import filereader, preprocess
import pandas as pd
from collections import Counter
import unicodedata

features = ['syll', 'cons', 'son', 'cont', 'asp', 'tense', 'nas', 'spread', 
            'lab', 'cor', 'ant', 'strid', 'dors', 'high', 'low', 'back', 'round']

def get_syl_frac(syl, syl_str_info):
    #onset, nucleus, coda 리스트 형태로 반환
    syl_frac = []

    for phon in syl:
        if (phon.syl_str_info == syl_str_info):
            syl_frac.append(phon)
            
    return syl_frac


#phoneme analysis
def syl2jamos(syl):
    jamos = []
    for phoneme in syl:
        jamos.append(phoneme.jamo)
    return "".join(jamos)

def syl2phontuple(syl):
    phon_tuple = []
    for phoneme in syl:
        phon_tuple.append((phoneme.jamo, phoneme.syl_str_info))
    return tuple(phon_tuple)




# target - error 간 공유, 비공유 자질 셋
# substitution error에 대해서만 구하기 (addition, omission x. )
# addition, omission은 수기로 분석하기. 몇 개 안 됨.

def get_featured_shared_nonshared(data_phon_featurized_list, data_feature):
    features_shared_data = []
    features_nonshared_data = []

    for i, sent in enumerate(data_phon_featurized_list): 
        for syl in sent:
            for phon in syl:
                features_phon = data_feature[phon.jamo]
                if phon.error_freq: # 오류 발생한 음소면
                    for j, error in enumerate(phon.error_token): # 해당 음소에서 발생한 오류 토큰 각각 살핌.
                        try: # addition 제외. 총 3건. (j 첨가 오류)
                            if (error.jamo in "jw"):
                                continue
                            #    print(syl2jamos(syl), phon.jamo, error.jamo)  #ㅊㅣ ㅣ j , ㅆㅡ ㅡ j, ㄱㅗ ㅗ j
                        except:
                            pass

                        try: 
                            # substitution 91건 대상으로만.
                            #addition, omission: 집계에서 제외됨
                            features_error = data_feature[error.jamo]
                            features_shared = {k: features_phon[k] for k in features_phon.keys() if features_phon[k] == features_error[k]}
                            features_nonshared = {k: (features_phon[k], features_error[k]) for k in features_phon.keys() if features_error[k] != features_phon[k]}
                            
                            if features_shared:
                                features_shared_data.append(tuple(features_shared.items()))

                            if features_nonshared:
                                features_nonshared_data.append(tuple(features_nonshared.items()))

                        except:
                           # omission 제외: 총 25건. (null로 처리)             
                            continue
                     
    return features_shared_data, features_nonshared_data




# =======================================================
# TRIGGER FINDER
# =======================================================


def is_substitution_error(error, phon):
    if (not error or not error.jamo): #omission
        return False
    elif (error.jamo in "jw"): #addition
        return False
    elif (error.jamo and error.jamo != phon.jamo):
        return True
    else:
        print("error: invalid categorization")
        print(error.jamo, phon.jamo)
        return False

def has_nonshared_features(p, features_nonshared):
    dict_p = p.__dict__.items()

    for (feature, (value_target, value_error)) in features_nonshared:
        if (p.__dict__[feature] != value_error):
            return False
        else:
            continue
    return True


def add_trigger_syl_str(phon, i, sent, triggers, features_nonshared, direction):
    # Phoneme 객체
    if direction == "left":
        syl_candidate = sent[phon.syl_idx - (i+1)] #sent[phon.syl_idx - (i+1)] # 1칸씩 옮겨가면서 syl 탐색.
    elif direction == "right":
        syl_candidate = sent[phon.syl_idx + (i+1)]

    for p in syl_candidate:
        if (phon.syl_str_info == p.syl_str_info and has_nonshared_features(p, features_nonshared)): #nonshared feature 중 error의 feature 모두 가지면
            #print("jamos:", phon.jamo, phon.syl_idx, p.jamo, p.syl_idx)
            triggers.append(p) # 해당 음소를 trigger로 간주
        else:
            continue
    return triggers


def add_trigger_min_dist(phon, phon_idx, i, segment_stream, triggers, features_nonshared, direction):
    segment_candidate = ""

    if direction == "left":
        segment_candidate = segment_stream[phon_idx - (i+1)] #sent[phon.syl_idx - (i+1)] # 1칸씩 옮겨가면서 syl 탐색.
    elif direction == "right":
        segment_candidate = segment_stream[phon_idx + (i+1)]
    else:
        print("ERROR: invalid direction")

    if (has_nonshared_features(segment_candidate, features_nonshared)): #nonshared feature 중 error의 feature 모두 가지면
        triggers.append(segment_candidate) # 해당 음소를 trigger로 간주
        #print("jamos:", phon.jamo, phon.syl_idx, segment_candidate.jamo, segment_candidate.syl_idx)
    else:
        pass

    return triggers


def find_trigger_by_syl_str(phon, sent, features_nonshared):
    triggers = []
    
    # 1. window size 설정.
    left_window_size = 7
    right_window_size = 7
    triggers_left = []
    triggers_right = []

    if (phon.syl_idx < 7): # syl이 왼쪽 끝에 있는 경우.
        left_window_size = phon.syl_idx
    if (len(sent)-1 - phon.syl_idx < 7): # syl이 오른쪽 끝에 있는 경우.
        right_window_size = len(sent)-1 - phon.syl_idx
    
    #print("window sizes: ", left_window_size, right_window_size)
    #print("idx: ", phon.syl_idx)

    # 2. 음절구조 맞게 trigger 탐색.
    #print("left")
    for i in range(left_window_size):
        triggers_left = add_trigger_syl_str(phon, i, sent, triggers, features_nonshared, direction ="left")

    #print("right")
    for i in range(right_window_size):
        triggers_right = add_trigger_syl_str(phon, i, sent, triggers, features_nonshared, direction ="right")

    triggers_sorted = sorted(list(set(triggers_left + triggers_right)), key = lambda x: abs(x.syl_idx - phon.syl_idx)) # phoneme 객체의 리스트
    #print("sorted triggers:", [(phon.jamo, phon.syl_idx) for phon in triggers_sorted])

    dist = [abs(x.syl_idx - phon.syl_idx) for x in triggers_sorted]
    #print("target-trigger dist: ", dist)

    # trigger 후보 중 가장 target과 가까운 trigger 선별
    #print("phon", phon.syl_idx)
    final_triggers =  [triggers_sorted[i] for i, d in enumerate(dist) if d == min(dist)]
    #print("final triggers:", [(p.jamo, p.syl_idx) for p in final_triggers])


    return final_triggers #triggers_sorted


def find_phon_idx_stream(phon, sent):
    phon_idx = 0
    length = 0
    for i, syl in enumerate(sent):
        if (phon.syl_idx == i):
            for j, p in enumerate(syl):
                if p.syl_str_info == phon.syl_str_info:
                    phon_idx = j + length # phon의 segment stream에서의 위치 idx
    
        length += len(syl)
    return phon_idx

def get_segment_stream(sent):
    segment_stream = []

    for syl in sent:
        segment_stream.extend(syl)
    
    return segment_stream


def find_trigger_by_min_dist(phon, sent, features_nonshared):
    triggers_left = []
    triggers_right = []
    triggers = []
    

    # sent를 segment의 stream으로 변환
    # segment_stream = []
    # phon_idx = 0
    # length = 0 
    # stream에서 phon 위치 찾기
    # for i, syl in enumerate(sent):
    #     if (phon.syl_idx == i):
    #         for j, p in enumerate(syl):
    #             if p.syl_str_info == phon.syl_str_info:
    #                 phon_idx = j + length # phon의 segment stream에서의 위치 idx
        
    #     segment_stream.extend(syl)
    #     length += len(syl)
    
    phon_idx = find_phon_idx_stream(phon, sent)
    segment_stream = get_segment_stream(sent)

    #print([seg.jamo for seg in segment_stream])

    left_window_size = 21 # 수정
    right_window_size = 21 # 수정

    if (phon_idx < 21): # syl이 왼쪽 끝에 있는 경우.
        left_window_size = phon_idx
    if (len(segment_stream)-1 - phon_idx < 21): # syl이 오른쪽 끝에 있는 경우.
        right_window_size = len(segment_stream)-1 - phon_idx

    #print("window sizes: ", left_window_size, right_window_size)
    #print("len: ", len(segment_stream), phon_idx)
    #print("left")

    for i in range(left_window_size):
        triggers_left = add_trigger_min_dist(phon, phon_idx, i, segment_stream, triggers, features_nonshared, direction ="left")
        #print("left triggers:", [(phon.jamo, phon.syl_idx) for phon in triggers_left])

    #print("right")
    for i in range(right_window_size):
        triggers_right = add_trigger_min_dist(phon, phon_idx, i, segment_stream, triggers, features_nonshared, direction ="right")
        #print("right triggers:", [(phon.jamo, phon.syl_idx) for phon in triggers_right])

    triggers_sorted = sorted(list(set(triggers_left + triggers_right)), key = lambda x: abs(segment_stream.index(x) - phon_idx)) #중복제거 # phoneme 객체의 리스트
    #print("sorted triggers:", [(phon.jamo, phon.syl_idx) for phon in triggers_sorted])

    dist = [abs(segment_stream.index(x) - phon_idx) for x in triggers_sorted]
    #print("target-trigger dist: ", dist)

    # trigger 후보 중 가장 target과 가까운 trigger 선별
    #print("phon", phon.syl_idx)
    final_triggers =  [triggers_sorted[i] for i, d in enumerate(dist) if d == min(dist)]
    #print("final triggers:", [(p.jamo, p.syl_idx) for p in final_triggers])

    return final_triggers #triggers_sorted

    
def find_trigger(phon, sent, features_nonshared, mode): 
    #phon 주변에서 features_nonshared 가진 phon을 mode 방식으로 찾음.
    triggers = []

    if (mode == "syl_str"):
        triggers = find_trigger_by_syl_str(phon, sent, features_nonshared) # phoneme 객체의 리스트
    
    elif (mode =="min_dist"):
        triggers = find_trigger_by_min_dist(phon, sent, features_nonshared)
    
    else:
        print("MODE ERROR: INVALID MODE INPUT")
    
    return triggers


def find_trigger_data(data_phon_featurized_list, data_feature): #syl_str, min_dist
    features_nonshared_data = get_featured_shared_nonshared(data_phon_featurized_list, data_feature)[1] # trigger clue

    trigger_data = []
    error_idx = 0

    for i, sent in enumerate(data_phon_featurized_list):
        #print(f"====={i+1}th sent====================================================")
        for syl in sent:
            for phon in syl:
                if phon.error_freq: # 오류 발생한 음소면
                    for error in phon.error_token: #오류 개수 == trigger 탐색 횟수 
                        if is_substitution_error(error, phon): #오류가 교체 오류면
                            # try:
                            #     print(f"---------{error_idx+1}. finding triggers of {error.jamo} in {syl2jamos(syl)}")
                            # except:
                            #     print(f"---------{error_idx+1}. finding triggers of {error} in {syl2jamos(syl)}")

                            # target->error에서 달라진 자질
                            features_nonshared = features_nonshared_data[error_idx] # trigger clue
                            #print(features_nonshared)
                            #print("phon_idx:", phon.syl_idx)

                            # trigger 찾기
                            # nonshared feature 가지고 있는 음소 탐색
                            #print("MODE: SYL STR")
                            triggers_syl_str = find_trigger(phon, sent, features_nonshared, mode ="syl_str" ) #리스트 자료형.
                            #print("TRIGGER syl_str: ", [(trigger.jamo, trigger.syl_idx) for trigger in triggers_syl_str]) #, find_error_type(triggers_syl_str, phon))
                            #print()
                            #print()
                            #print("MODE: MIN DIST")
                            triggers_min_dist = find_trigger(phon, sent, features_nonshared, mode = "min_dist") #리스트 자료형
                            #print("TRIGGER min dist: ", [(trigger.jamo, trigger.syl_idx) for trigger in triggers_min_dist]) #, find_error_type(triggers_min_dist, phon))
                            
                            trigger_data.append((phon, sent, (triggers_syl_str, triggers_min_dist) )) # 튜플 자료형

                            error_idx += 1
                            #print()
                            #print()

        #print()
        #print()
        
    return trigger_data








# ====================================================================================
# OBSOLETE
# ====================================================================================

def get_ngrams(syl, n):
    ngrams = []
    if len(syl) < n:
        return ngrams #emtpy list

    phon_tuple = syl2phontuple(syl)
    loop_num = len(phon_tuple) - n + 1

    #jamos = syl2jamos(syl)
    #loop_num = len(jamos) - n + 1

    for i in range(loop_num): #loop_num회 반복문.
        ngram = phon_tuple[i:i+n]
        #ngram = jamos[i:i+n]
        ngrams.append(ngram)

    return ngrams


def get_ngrams_feature(sent, feature, n): #get_ngrams_feature(sent, feature, 1) #ex. 문장 1에 있는 onset들 모음
    ngrams = []
    feature_list = []

    if (len(sent) < n):
        return ngrams

    loop_num = len(sent) - n + 1

    for phon in sent:
        phon_feature = phon.__dict__[feature]
        feature_list.append(phon_feature)

    for i in range(loop_num): #loop_num회 반복문.
        ngram = "".join(feature_list[i:i+n])
        #print(f"ngram: {ngram}")
        ngrams.append(ngram)

    return ngrams
    
def get_ngrams_phon_sent(sent):
    # 문장별로 phoneme의 ngram 모으기
    unigrams_sent = []
    bigrams_sent = []
    trigrams_sent = []
    fourgrams_sent = []
    
    for syl in sent:
        unigrams_sent.extend(get_ngrams(syl, 1))
        bigrams_sent.extend(get_ngrams(syl, 2))
        trigrams_sent.extend(get_ngrams(syl, 3))
        fourgrams_sent.extend(get_ngrams(syl, 4))

    return unigrams_sent, bigrams_sent, trigrams_sent, fourgrams_sent
    

def get_ngrams_phon(data_phon_list):
    unigrams = []
    bigrams = []
    trigrams = []
    fourgrams = []
    ngrams= []
    
    for sent in data_phon_list:
        unigrams_sent, bigrams_sent, trigrams_sent, fourgrams_sent = get_ngrams_phon_sent(sent)

        # 전체 ngram 목록에 문장별 phoneme ngram들 문장별로 리스트로 저장
        unigrams.append(unigrams_sent)
        bigrams.append(bigrams_sent)
        trigrams.append(trigrams_sent)
        fourgrams.append(fourgrams_sent)

    ngrams.append(unigrams)
    ngrams.append(bigrams)
    ngrams.append(trigrams)
    ngrams.append(fourgrams)

    return ngrams



def get_ngrams_freqdict(data_phon_list):
    freqdicts = []
    
    ngrams = get_ngrams_phon(data_phon_list)

    for ngram in ngrams:
        freqdict_ngram = []
        for ngram_sent in ngram:
            freqdict_sent = Counter(ngram_sent)
            freqdict_ngram.append(freqdict_sent)
        freqdicts.append(freqdict_ngram)

    return freqdicts




# feature analysis
def get_syl_str_list(data_phon_featurized_list, syl_str_info):
    data_syl_str_list = []

    for sent in data_phon_featurized_list:
        sent_syl_str_list = []
        for syl in sent:
            for phon in syl:
                if (phon.syl_str_info == syl_str_info):
                    sent_syl_str_list.append(phon) #ex. 문장별로 onset만 모음
                else:
                    continue
        data_syl_str_list.append(sent_syl_str_list) #ex. 데이터의 onset만 문장별로 모아둠.

    return data_syl_str_list


def get_ngrams_features(data_syl_str_list, feature): #ex. onset의 문장별 모음.
    #특정 feature에 대한 ngram 구하기
    unigrams = []
    bigrams = []
    trigrams = []
    fourgrams = []
    
    for sent in data_syl_str_list:
        unigrams.extend(get_ngrams_feature(sent, feature, 1))
        bigrams.extend(get_ngrams_feature(sent, feature, 2))
        trigrams.extend(get_ngrams_feature(sent, feature, 3))
        fourgrams.extend(get_ngrams_feature(sent, feature, 4))

    ngrams = [unigrams, bigrams, trigrams, fourgrams]
    return ngrams





#Frisch Similarity
"""
features = ['syll', 'cons', 'son', 'cont', 'asp', 'tense', 'nas', 'spread', 
            'lab', 'cor', 'ant', 'strid', 'dors', 'high', 'low', 'back', 'round']

"""
def get_frisch_similarity(phoneme1, phoneme2, natural_classes):
    feature_names = list(phoneme1.__dict__.keys())[4:]
    features_phoneme1 = list(phoneme1.__dict__.values())[4:]
    features_phoneme2 = list(phoneme2.__dict__.values())[4:]

    shared_feature_idx = []
    shared_feature_names = set()
    nonshared_feature_names = set()

    # 공유하는 자질 인덱스 모으기
    for i, feature1 in enumerate(features_phoneme1):
        feature2 = features_phoneme2[i]
        if (feature1 == feature2):
            shared_feature_idx.append(i)

    # 공유하는/공유하지 않는 자질 인덱스를 문자열로 변환
    for i in range(len(features_phoneme1)):
        if (i in shared_feature_idx):
            shared_feature_names.add(feature_names[i]) #ex. [son, cont, ...]
        else:
            nonshared_feature_names.add(feature_names[i]) #ex. [nasal, cor, ...]


    # frisch similarity 집계
    num_natural_class_and = 0
    num_natural_class_or = 0

    for natural_class in natural_classes:
        natural_class = set(natural_class)
        if (natural_class.issubset(shared_feature_names)):
            num_natural_class_and += 1
            continue
        else:
            num_natural_class_or += 1
            continue

    #print(len(natural_classes), num_natural_class_and, num_natural_class_or)
    frisch_similarity = (num_natural_class_and) / (num_natural_class_and + num_natural_class_or)

    return frisch_similarity
            






