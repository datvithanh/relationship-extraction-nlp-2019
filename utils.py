#utils
import re
import numpy as np
import nltk
import pandas as pd

class2label = {'Other': 0,
               'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
               'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
               'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
               'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
               'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
               'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
               'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
               'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
               'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}

pos2id = { 0: 0, 13: 155,  14: 156,  15: 157,  16: 158,  17: 159,  18: 160,  19: 161,  20: 70,  21: 71,  22: 72,  23: 73,  24: 74,  25: 75,  26: 76,  27: 77,  28: 78,  29: 79,  30: 80,  31: 81,  32: 82,  33: 83,  34: 84,  35: 85,  36: 86,  37: 87,  38: 88,  39: 89,  40: 90,  41: 91,  42: 92,  43: 93,  44: 94,  45: 95,  46: 96,  47: 97,  48: 98,  49: 99,  50: 100,  51: 101,  52: 102,  53: 103,  54: 104,  55: 105,  56: 106,  57: 107,  58: 108,  59: 109,  60: 110,  61: 111,  62: 63,  63: 64,  64: 65,  65: 66,  66: 67,  67: 68,  68: 69,  69: 47,  70: 48,  71: 49,  72: 50,  73: 51,  74: 52,  75: 53,  76: 1,  77: 2,  78: 3,  79: 4,  80: 5,  81: 6,  82: 7,  83: 8,  84: 9,  85: 10,  86: 11,  87: 12,  88: 13,  89: 14,  90: 15,  91: 16,  92: 17,  93: 18,  94: 19,  95: 20,  96: 21,  97: 22,  98: 23,  99: 24,  100: 25,  101: 26,  102: 27,  103: 28,  104: 29,  105: 30,  106: 31,  107: 32,  108: 33,  109: 34,  110: 35,  111: 36,  112: 37,  113: 38,  114: 39,  115: 40,  116: 41,  117: 42,  118: 43,  119: 44,  120: 45,  121: 46,  122: 54,  123: 55,  124: 56,  125: 57,  126: 58,  127: 59,  128: 60,  129: 61,  130: 62,  131: 112,  132: 113,  133: 114,  134: 115,  135: 116,  136: 117,  137: 118,  138: 119,  139: 120,  140: 121,  141: 122,  142: 123,  143: 124,  144: 125,  145: 126,  146: 127,  147: 128,  148: 129,  149: 130,  150: 131,  151: 132,  152: 133,  153: 134,  154: 135,  155: 136,  156: 137,  157: 138,  158: 139,  159: 140,  160: 141,  161: 142,  162: 143,  163: 144,  164: 145,  165: 146,  166: 147,  167: 148,  168: 149,  169: 150,  170: 151,  171: 152,  172: 153,  173: 154}

def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()

def transform_pos(pos, seq_len = 90):
    ret = [0] * seq_len
    for i,v in enumerate(pos.strip().split(' ')):
        ret[i] = pos2id[int(v)]
    return np.array(ret)

def load_data_and_labels(path):
    data = []
    all_tokens = []
    lines = [line.strip() for line in open(path)]
    max_sentence_length = 0
    for idx in range(0, len(lines), 4):
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = sentence.replace('<e1>', ' _e11_ ')
        sentence = sentence.replace('</e1>', ' _e12_ ')
        sentence = sentence.replace('<e2>', ' _e21_ ')
        sentence = sentence.replace('</e2>', ' _e22_ ')

        sentence = clean_str(sentence)
        tokens = nltk.word_tokenize(sentence)
        all_tokens.append(tokens)

        if max_sentence_length < len(tokens):
            max_sentence_length = len(tokens)

        e1 = tokens.index("e12") - 1
        e2 = tokens.index("e22") - 1
        sentence = " ".join(tokens)
        data.append([id, sentence, e1, e2, relation])

    df = pd.DataFrame(data=data, columns=["id", "sentence", "e1", "e2", "relation"])
    pos1, pos2 = get_relative_position(df, 90)

    df['label'] = [class2label[r] for r in df['relation']]

    # Text Data
    e1 = df['e1'].tolist()
    e2 = df['e2'].tolist()

    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()

    # Transform pos
    p1, p2 = [], []
    for i in range(len(pos1)):
        p1.append(transform_pos(pos1[i]))
        p2.append(transform_pos(pos2[i]))

    return all_tokens, labels_flat, e1, e2, p1, p2

def get_relative_position(df, max_sentence_length):
    # Position data
    pos1 = []
    pos2 = []
    for df_idx in range(len(df)):
        sentence = df.iloc[df_idx]['sentence']
        tokens = nltk.word_tokenize(sentence)
        e1 = df.iloc[df_idx]['e1']
        e2 = df.iloc[df_idx]['e2']
        p1 = ""
        p2 = ""
        for word_idx in range(len(tokens)):
            p1 += str((max_sentence_length - 1) + word_idx - e1) + " "
            p2 += str((max_sentence_length - 1) + word_idx - e2) + " "
        pos1.append(p1)
        pos2.append(p2)

    return pos1, pos2