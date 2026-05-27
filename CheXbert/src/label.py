import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import utils
from models.bert_labeler import bert_labeler
from datasets_chexbert.unlabeled_dataset import UnlabeledDataset
from constants import *
from collections import OrderedDict
from transformers import BertTokenizer
from tqdm import tqdm


def collate_fn_no_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len."""
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                  batch_first=True,
                                                  padding_value=PAD_IDX)
    len_list = [s['len'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list}
    return batch


def load_unlabeled_data(csv_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=False):
    """Create UnlabeledDataset object for the input reports."""
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(csv_path)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    return loader


def label(checkpoint_path, csv_path):
    """Labels a dataset of reports
    @param checkpoint_path (string): location of saved model checkpoint
    @param csv_path (string): location of csv with reports

    @returns y_pred (List[List[int]]): Labels for each of the 14 conditions, per report
    """
    ld = load_unlabeled_data(csv_path)

    model = bert_labeler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]

    print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    print("The batch size is %d" % BATCH_SIZE)
    with torch.no_grad():
        for i, data in enumerate(tqdm(ld)):
            batch = data['imp']
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)

            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1)
                y_pred[j].append(curr_y_pred)

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)

    if was_training:
        model.train()

    y_pred = [t.tolist() for t in y_pred]
    return y_pred


def save_preds(y_pred, csv_path, out_path):
    """Save predictions as out_path/labeled_reports.csv"""
    y_pred = np.array(y_pred)
    y_pred = y_pred.T

    df = pd.DataFrame(y_pred, columns=CONDITIONS)
    reports = pd.read_csv(csv_path)['Report Impression']

    df['Report Impression'] = reports.tolist()
    new_cols = ['Report Impression'] + CONDITIONS
    df = df[new_cols]

    df.replace(0, np.nan, inplace=True)   # blank class is NaN
    df.replace(3, -1, inplace=True)       # uncertain class is -1
    df.replace(2, 0, inplace=True)        # negative class is 0

    df.to_csv(os.path.join(out_path, 'labeled_reports.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label a csv file containing radiology reports')
    parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                        help='path to csv containing reports (column: "Report Impression")')
    parser.add_argument('-o', '--output_dir', type=str, nargs='?', required=True,
                        help='path to intended output folder')
    parser.add_argument('-c', '--checkpoint', type=str, nargs='?', required=True,
                        help='path to the pytorch checkpoint')
    args = parser.parse_args()
    csv_path = args.data
    out_path = args.output_dir
    checkpoint_path = args.checkpoint

    y_pred = label(checkpoint_path, csv_path)
    save_preds(y_pred, csv_path, out_path)
