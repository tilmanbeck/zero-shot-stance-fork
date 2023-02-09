import os
import nltk
import pandas as pd
import uuid

def contains(text, characters=['-', "'" ,'/']):
    for keyword in characters:
        if keyword in text:
            return True
    return False


def preprocess_text(text):
    sents_tokenized = []

    for sent in nltk.sent_tokenize(text):
        # tokenize text
        tokens = nltk.tokenize.word_tokenize(sent, language='english')
        # convert tokens to lowercase
        tokens = [token.lower() for token in tokens]
        # remove stopwords
        tokens = [token for token in tokens if token not in nltk.corpus.stopwords.words('english')]
        # remove punctuation
        tokens = [token for token in tokens if token.isalnum() or contains(token)]

        sents_tokenized.append(tokens)
    if sents_tokenized == [[]]: # in the rare case if all tokens are removed, we ignore stopword removal
        sents_tokenized = []
        for sent in nltk.sent_tokenize(text):
            # tokenize text
            tokens = nltk.tokenize.word_tokenize(sent, language='english')
            # convert tokens to lowercase
            tokens = [token.lower() for token in tokens]
            # remove punctuation
            removed_punctuation = [token for token in tokens if token.isalnum() or contains(token)]
            if removed_punctuation != []:
                sents_tokenized.append(removed_punctuation)
            else:
                sents_tokenized.append(tokens)
    assert sents_tokenized != [[]]
    return sents_tokenized


def concatenate_text(texts):
    return ". ".join([" ".join(i) for i in texts])


def preprocess_topic(topic):
    # tokenize text
    tokens = nltk.tokenize.word_tokenize(topic, language='english')
    # convert tokens to lowercase
    tokens = [token.lower() for token in tokens]
    # remove stopwords
    tokens = [token for token in tokens if token not in nltk.corpus.stopwords.words('english')]
    # remove punctuation
    tokens = [token for token in tokens if token.isalnum() or contains(token)]
    return tokens


def flatten_list_of_lists(l):
    return [item for sublist in l for item in sublist]


def load_and_preprocess_data(fp, output_path_fp):
    df = pd.read_json(fp, lines=True, orient='records')
    # create unique ID for each target
    target2id = {target: target.encode('utf-8').hex() for target in df['target'].unique()}
    df['ori_id'] = df['target'].apply(lambda x: target2id[x])
    df.rename(columns={'target': 'ori_topic'}, inplace=True)
    # according to paper by Allaway and McKeown (2020): preprocess text by tokenizing and
    # removing stopwords and punctuation using NLTK
    df['post'] = df['text']
    df['text'] = df['text'].apply(preprocess_text)
    df['text_s'] = df['text'].apply(concatenate_text)
    df['topic'] = df['ori_topic'].apply(preprocess_topic)
    df['topic_str'] = df.apply(lambda x: " ".join(x.topic), axis=1)
    df['new_id'] = df.apply(lambda x: uuid.uuid4(), axis=1)
    # df['topic'] = json.dumps(df['topic'].values)
    # df['text'] = json.dumps(df['text'].values)

    df['contains_topic?'] = df.apply(lambda x: int(x.topic_str in x.text_s), axis=1)
    df['seen?'] = 0

    for split in df["split"].unique():
        df_split = df[df["split"] == split]
        df_split.to_csv(os.path.join(output_path_fp, f"{split}.csv"), index=False)


if __name__ == "__main__":
    input_path = "/home/tilman/Repositories/context-stance/data/benchmark"
    output_path = "/home/tilman/Repositories/zero-shot-stance-fork/data/benchmark"
    for folder in os.listdir(input_path):
        for file in os.listdir(os.path.join(input_path, folder)):
            if file.endswith(".jsonl"):
                fp = os.path.join(input_path, folder, file)
                print(fp)
                output_path_fp = os.path.join(output_path, folder)
                os.makedirs(output_path_fp, exist_ok=True)
                load_and_preprocess_data(fp, output_path_fp)