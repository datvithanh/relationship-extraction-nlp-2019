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

    return all_tokens, labels_flat, e1, e2, pos1, pos2

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

if __name__ == "__main__":
    all_tokens, labels_flat, e1, e2, pos1, pos2 = load_data_and_labels('data/SemEval2010_task8_training/TRAIN_FILE.TXT')