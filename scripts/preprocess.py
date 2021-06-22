import unicodedata
import itertools

class Phoneme:
    def __init__(self, syl_idx, syl_str_info, jamo, jamo_type):
        self.syl_idx = syl_idx
        self.syl_str_info = syl_str_info
        self.jamo = jamo
        self.jamo_type = jamo_type

        #common features
        self.syll = '0'
        self.cons = '0'
        self.son = '0'

        #consonant-only features (except lab) 
        self.cont = '0'
        self.asp = '0'
        self.tense = '0' 
        self.nas = '0'
        self.spread = '0'
        self.lab = '0'
        self.cor = '0'
        self.ant = '0'
        self.strid = '0'
        self.dors = '0'
    
        #vowel-only features
        self.high = '0'
        self.low = '0'
        self.back = '0'
        self.round = '0'

        #error info
        self.error_freq = 0
        self.error_token = []
        

def decompose(syl):
    LEADING = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
    VOWEL = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
    TRAILING = ('',) + tuple('ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ')

    n_cnt = len(VOWEL) * len(TRAILING)
    t_cnt = len(TRAILING)
    jamos = ''

    try:
        if unicodedata.name('가').startswith('HANGUL SYLLABLE'):
            s_idx = ord(syl) - ord('가')
            l_idx = s_idx // n_cnt
            v_idx = s_idx % n_cnt // t_cnt
            t_idx = s_idx % n_cnt % t_cnt          
            jamos += LEADING[l_idx] + VOWEL[v_idx] + TRAILING[t_idx]
            return jamos
        else:
            return jamos

    except:
        return jamos


def get_syl_str_info(jamo, phon_idx, jamos):
    try:
        #jamo_idx = jamos.index(jamo) #ㄹ, ㄹ 두 번 나오는 경우 사용 불가능. phon_idx로 수정
        jamo_len = len(jamos)
        syl_str_info = ''

        if (phon_idx == 0): # "아"에서 ㅇ 살아 있는 상태.
            syl_str_info = "onset"
        elif (phon_idx == 1):
            syl_str_info = "nucleus"
        elif (phon_idx == 2):
            syl_str_info = "coda"

        else:
            return syl_str_info

        return syl_str_info

    except:
        return syl_str_info


def convert_monophthong(jamo):
        if (jamo == "ㅐ"):
            return "ㅔ"
        elif (jamo == "ㅙ"):
            return "ㅞ"
        elif (jamo == "ㅒ"):
            return "ㅖ"
        elif (jamo == "ㅚ"):
            return "ㅞ"
        else:
            return jamo


def decompose_diphthong(jamo):
    try:
        jamo = convert_monophthong(jamo)

        monophthongs = "ㅏㅓㅗㅜㅔㅏㅓㅔㅣ"
        diphthongs_j = "ㅑㅕㅛㅠㅖ"
        diphthongs_w = "ㅘㅝㅞㅟ"
        diphthongs_ui = "ㅢ"
        diphthongs = diphthongs_j + diphthongs_w + diphthongs_ui

        if (jamo not in diphthongs): #monophthong
            return jamo

        jamo_idx = diphthongs.index(jamo)

        if (jamo in diphthongs_j):
            jamo = "j"+ monophthongs[jamo_idx]
        elif (jamo in diphthongs_w):
            jamo = "w"+ monophthongs[jamo_idx]
        elif (jamo == diphthongs_ui):
            jamo = "ㅡ"+"j"
        else:
            return jamo
        return jamo

    except:
        return jamo


def get_phon_list(syl, syl_idx):
    phon_list = []
    phonemes = []

    try:
        #give syllable structure info to Phoneme object
        jamos = decompose(syl)

        for i, jamo in enumerate(jamos):
            phon_idx = i
            syl_str_info = get_syl_str_info(jamo, phon_idx, jamos) # onset, nucleus, coda 결정시 활용.
            nucleus_decomposed = decompose_diphthong(jamo)
            is_diphthong = (len(nucleus_decomposed) > 1)

            # check diphthongs
            if (is_diphthong):
                phoneme_glide = Phoneme(syl_idx, syl_str_info, nucleus_decomposed[0], "glide")
                phoneme_vowel =  Phoneme(syl_idx, syl_str_info, nucleus_decomposed[1], "vowel")

                phonemes.append(phoneme_glide)
                phonemes.append(phoneme_vowel)

            # check non-diphthongs
            if (syl_str_info == "onset") or (syl_str_info == "coda"):
                phoneme = Phoneme(syl_idx, syl_str_info, jamo, "consonant")
            elif (syl_str_info == "nucleus"):
                jamo = convert_monophthong(jamo)
                phoneme = Phoneme(syl_idx, syl_str_info, jamo, "vowel")
            else:
                raise ValueError


        #remove invalid onset
            invalid_onset = (phoneme.syl_str_info == "onset") and (phoneme.jamo == "ㅇ")

            if (invalid_onset) : #"아" == "ㅏ"
                continue
            elif (is_diphthong):
                phon_list.extend(phonemes)
            else:
                phon_list.append(phoneme)

        return phon_list

    except:
        return phon_list



def featurize(phoneme, data_feature_dict): #Phoneme object
    phon = phoneme.jamo
    features = data_feature_dict[phon]

    phoneme.syll = features["syll"]
    phoneme.cons = features["cons"]
    phoneme.son = features["son"]
    phoneme.cont = features["cont"]

    phoneme.asp = features["asp"]
    phoneme.tense = features["tense"]
    phoneme.nas = features["nas"]
    phoneme.spread = features["spread"]
    phoneme.lab = features["lab"]
    phoneme.cor = features["cor"]
    phoneme.ant = features["ant"]
    phoneme.strid = features["strid"]
    phoneme.dors = features["dors"]

    phoneme.high = features["high"]
    phoneme.low = features["low"]
    phoneme.back = features["back"]
    phoneme.round = features["round"]

    return phoneme


def get_natural_classes(features):
    natural_classes = []
    for L in range(0, len(features)+1):
        for subset in itertools.combinations(features, L):
            if subset:
                natural_classes.append(subset)

    return natural_classes


def print_natural_classes(natural_classes_cons_only, natural_classes_vowel_only, natural_classes_cons_vowel):
    print("======natural classes_cons only======")
    for i in natural_classes_cons_only:
        print(i)
    print()
    print()
    print("======natural classes_vowel only======")
    for i in natural_classes_vowel_only:
        print(i)
    print()
    print()
    print("======natural classes_cons & vowel======")
    for i in natural_classes_cons_vowel:
        print(i)
    return