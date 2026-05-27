import torch
from transformers import BertTokenizer
import bert_tokenizer
from torch.utils.data import Dataset


class UnlabeledDataset(Dataset):
    """The dataset to contain report impressions without any labels."""

    def __init__(self, csv_path):
        """Initialize the dataset object
        @param csv_path (string): path to csv with 'Report Impression' column
        """
        tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
        impressions = bert_tokenizer.get_impressions_from_csv(csv_path)
        self.encoded_imp = bert_tokenizer.tokenize(impressions, tokenizer)

    def __len__(self):
        return len(self.encoded_imp)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imp = self.encoded_imp[idx]
        imp = torch.LongTensor(imp)
        return {"imp": imp, "len": imp.shape[0]}
