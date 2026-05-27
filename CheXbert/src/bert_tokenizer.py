import os
import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm


def get_impressions_from_csv(path):
    df = pd.read_csv(path)
    imp = df['Report Impression']
    imp = imp.str.strip()
    imp = imp.replace('\n', ' ', regex=True)
    imp = imp.replace(r'\s+', ' ', regex=True)
    imp = imp.str.strip()
    return imp


def tokenize(impressions, tokenizer):
    new_impressions = []
    print("\nTokenizing report impressions. All reports are cut off at 512 tokens.")
    for i in tqdm(range(impressions.shape[0])):
        try:
            tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
        except Exception:
            tokenized_imp = None
        if tokenized_imp:
            token_ids = tokenizer.convert_tokens_to_ids(tokenized_imp)
            res = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]
            if len(res) > 512:
                res = res[:511] + [tokenizer.sep_token_id]
            new_impressions.append(res)
        else:
            new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
    return new_impressions


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Tokenize radiology report impressions.')
    parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                        help='path to csv containing reports')
    parser.add_argument('-o', '--output_path', type=str, nargs='?', required=True,
                        help='path to intended output file')
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
    impressions = get_impressions_from_csv(args.data)
    new_impressions = tokenize(impressions, tokenizer)
    with open(args.output_path, 'w') as filehandle:
        json.dump(new_impressions, filehandle)
