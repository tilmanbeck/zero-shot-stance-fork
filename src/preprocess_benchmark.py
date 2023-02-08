import os
import nltk
import pandas as pd


def contains(text, characters=['-', "'" ,'/']):
    for keyword in characters:
        if keyword in text:
            return True
    return False


def preprocess_text(text):
    sents = []
    for sent in nltk.sent_tokenize(text):
        # tokenize text
        tokens = nltk.tokenize.word_tokenize(sent, language='english')
        # convert tokens to lowercase
        tokens = [token.lower() for token in tokens]
        # remove stopwords
        tokens = [token for token in tokens if token not in nltk.corpus.stopwords.words('english')]
        # remove punctuation
        tokens = " ".join([token for token in tokens if token.isalnum() or contains(token)])
        sents.append(tokens)
    return ". ".join(sents)


def load_and_preprocess_data(fp, output_path_fp):
    df = pd.read_json(fp, lines=True, orient='records')
    # according to paper by Allaway and McKeown (2020): preprocess text by tokenizing and
    # removing stopwords and punctuation using NLTK
    df['text_s'] = df['text'].apply(preprocess_text)
    # create unique ID for each target
    target2id = {target: target.encode('utf-8').hex() for target in df['target'].unique()}
    df['ori_id'] = df['target'].apply(lambda x: target2id[x])
    df.rename(columns={'target': 'ori_topic'}, inplace=True)
    for split in df["split"].unique():
        df_split = df[df["split"] == split]
        df_split.to_csv(os.path.join(output_path_fp, f"{split}.csv"), index=False)


if __name__ == "__main__":
    input_path = "/home/beck/Repositories/context-stance/data/benchmark"
    output_path = "/home/beck/Repositories/zero-shot-stance-fork/data/benchmark"
    for folder in os.listdir(input_path):
        for file in os.listdir(os.path.join(input_path, folder)):
            if file.endswith(".jsonl"):
                fp = os.path.join(input_path, folder, file)
                print(fp)
                output_path_fp = os.path.join(output_path, folder)
                os.makedirs(output_path_fp, exist_ok=True)
                load_and_preprocess_data(fp, output_path_fp)