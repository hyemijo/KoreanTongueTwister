import csv
import numpy as np
import preprocess, analyze
import unicodedata

# 1. syllabification
def get_data_raw(filename):
    # get raw transcription data from the file
    with open(filename, "r", encoding ="utf-8") as f:
        data_raw = f.readlines()
        data_raw = [line.strip() for line in data_raw if line.strip()]
    return data_raw


def get_data_syl(filename):
    #get data of valid syllables
    data_raw = get_data_raw(filename)
    data_syl = []

    for sent in data_raw:
        sent_syl = [syl for syl in list(sent) if syl.strip()]
        data_syl.append(sent_syl)
    return data_syl


def get_data_syl_hangul(filename):
    #get data of valid hangul syllables
    data_syl = get_data_syl(filename)
    data_syl_hangul = []
    for sent in data_syl:
        sent_syl_hangul = [syl for syl in list(sent) if syl.strip() and unicodedata.name(syl).startswith('HANGUL SYLLABLE')] # '' 제거
        data_syl_hangul.append(sent_syl_hangul)
    return data_syl_hangul


def get_data_phon(filename):
    data_syl = get_data_syl(filename)
    data_phon_list = []

    for i, sent_syl in enumerate(data_syl):
        sent_phon_list = []
        for j, syl in enumerate(sent_syl):
            phon_list = preprocess.get_phon_list(sent_syl[j], j)
            sent_phon_list.append(phon_list)
        data_phon_list.append(sent_phon_list)
    return data_phon_list


def compare_sent(sent1, sent2, sent2_idx):
    #errors = compare_sent(sent_error, sent_init, sent_init_idx)
    diffs = []
    sent1_syl = [syl for syl in list(sent1) if syl.strip()]
    sent2_syl = [syl for syl in list(sent2) if syl.strip()]

    for i, char in enumerate(sent1_syl):
        char2 = sent2_syl[i]

        if (char != char2):
            sent2_idx = int(sent2_idx)
            info_diff = (char, sent2_idx, i)
            diffs.append(info_diff)
        else:
            continue

    return diffs

def compare_syl(syl1, syl2):
    list1 = []
    list2 = []

    for phon in syl1:
        list1.append(phon.jamo)
    for phon in syl2:
        list2.append(phon.jamo)

    if (list1 == list2):
        return True
    else:
        return False


def tag_error_phon_init(phon_init, phon_error):
    if not compare_syl(phon_init, phon_error): # phon_init, phon_error 구성 다르면?
        phon_init[0].error_freq += 1 #전제: onset은 1개 #빈 리스트라면? [] [ㄱ]. 없던 onset 첨가된 경우 있을 수 있음.
        if (not phon_error):
            phon_error = [""]
        
        try:
            phon_error_net = [error for error in phon_error if error.jamo not in [p.jamo for p in phon_init]]
            phon_init[0].error_token.extend(phon_error_net)
        except:
            phon_init[0].error_token.extend(phon_error)
    
    return


def tag_error(error, data_phon_featurized_list):
    syl_init = data_phon_featurized_list[error[1]][error[2]] #원래 syl. 정확한 발음
    syl_error =  preprocess.get_phon_list(error[0], error[1]) #발음 오류 syl.

    onset_init = analyze.get_syl_frac(syl_init, "onset")
    nucleus_init = analyze.get_syl_frac(syl_init, "nucleus")
    coda_init = analyze.get_syl_frac(syl_init, "coda")

    onset_error = analyze.get_syl_frac(syl_error, "onset")
    nucleus_error = analyze.get_syl_frac(syl_error, "nucleus")
    coda_error = analyze.get_syl_frac(syl_error, "coda")

    syl_fracs = (onset_init, nucleus_init, coda_init, onset_error, nucleus_error, coda_error)        
    if (len(nucleus_init) > len(nucleus_error)): # glide 탈락
        nucleus_init[0].error_freq += 1
        nucleus_init[0].error_token.append("")
    else: #addition, substitution
        tag_error_phon_init(nucleus_init, nucleus_error) #[ㅗ] [j, ㅗ]: j-> 첨가. "ㅗ".error_token = [j]
        if (len(nucleus_init) < len(nucleus_error) and nucleus_init[0].jamo != nucleus_error[1].jamo): #glide 첨가
            nucleus_init[0].error_freq += 1
        
    # onset, coda 오류 태깅
    tag_error_phon_init(onset_init, onset_error)
    tag_error_phon_init(coda_init, coda_error) #전제: coda은 1개 #빈 리스트라면? [] [ㄱ]. 없던 coda 첨가된 경우 있을 수 있음.

    return


def get_data_error_tagged(filename, data_phon_featurized_list):
    data_error_raw = get_data_raw(filename)
    data_error_list = []
    idxs_sents_init = []

    #원문의 인덱스
    for i, line in enumerate(data_error_raw):
        if line.isdigit():
            idxs_sents_init.append(i+1) 

    for i, idx in enumerate(idxs_sents_init):
        sent_errors = []
        idx1 = idxs_sents_init[i] # ex. 0번째 문장

        if (i == len(idxs_sents_init) - 1):
            idx2 = len(data_error_raw) + 1
        else:
            idx2 = idxs_sents_init[i+1] # ex. 1번째 문장

        sent_init = data_error_raw[idx1]

        for j in range(idx1+1, idx2-1): #발음 오류 문장들
            sent_error = data_error_raw[j]
            sent_init_idx = data_error_raw[idx1-1]
            errors = compare_sent(sent_error, sent_init, sent_init_idx)
            
            for error in errors:
                tag_error(error, data_phon_featurized_list)
                data_error_list.append(error)

    return data_error_list


# 2. featurization
def get_data_feature_raw(filename):
    # get raw feature table data from the file
    data_feature_raw = []
    with open(filename, "r", encoding ="utf-8") as f:
        reader = csv.DictReader(f, delimiter = "\t")
        for row in reader:
            data_feature_raw.append(row)
    return data_feature_raw


def get_data_feature_dict(filename):
    # get dictionary-typed feature table data
    data_feature = dict()
    data_feature_raw = get_data_feature_raw(filename)
    for row in data_feature_raw:
        phon = row["phon"]
        row.pop("phon")
        data_feature[phon] = row
    return data_feature


def get_data_phon_featurized(filename_phon, filename_feature):
    # get featurized phoneme data
    data_phon_list = get_data_phon(filename_phon)
    data_feature_dict = get_data_feature_dict(filename_feature)
    data_phon_featurized_list = []

    for sent in data_phon_list:
        sent_phon_featurized_list = []
        for syl in sent:
            syl_phon_featurized_list = []
            for phon in syl:
                phon = preprocess.featurize(phon, data_feature_dict)
                syl_phon_featurized_list.append(phon)
            sent_phon_featurized_list.append(syl_phon_featurized_list)
        data_phon_featurized_list.append(sent_phon_featurized_list)

    return data_phon_featurized_list



