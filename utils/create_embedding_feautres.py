import pandas as pd
import os
import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import gc
from torch.utils.data import DataLoader, Dataset
from utils.general_utils import get_min_max_scores


def create_embedding_features(data_path, prompt_id, attribute_name, embedding_model_name, device) -> tuple:
    # Load data
    print(f'load data from {data_path}...')
    data = load_data(data_path)

    y_train = np.array(data['train']['label'])
    y_dev = np.array(data['dev']['label'])
    y_test = np.array(data['test']['label'])

    minscore, maxscore = get_min_max_scores()[prompt_id][attribute_name]
    y_train = (y_train - minscore) / (maxscore - minscore)
    y_dev = (y_dev - minscore) / (maxscore - minscore)
    y_test = (y_test - minscore) / (maxscore - minscore)

    # Create embedding
    os.makedirs(data_path + 'cache/', exist_ok=True)
    pkl_files = [file for file in os.listdir(data_path + 'cache/') if file.endswith('.pkl')]
    model_name = embedding_model_name
    if len(pkl_files) == 0:
        # Load embedding model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)

        train_loader = create_data_loader(data['train']['feature'], tokenizer, max_length=512, batch_size=32)
        dev_loader = create_data_loader(data['dev']['feature'], tokenizer, max_length=512, batch_size=32)
        test_loader = create_data_loader(data['test']['feature'], tokenizer, max_length=512, batch_size=32)

        # Create embedding
        print('[Train]')
        train_features = run_embedding_model(train_loader, model, device)
        print('[Dev]')
        dev_features = run_embedding_model(dev_loader, model, device)
        print('[Test]')
        test_features = run_embedding_model(test_loader, model, device)
        
        torch.cuda.empty_cache()
        gc.collect()

        # Save embedding
        with open(data_path + 'cache/train_features.pkl', 'wb') as f:
            pickle.dump(train_features, f)
        with open(data_path + 'cache/dev_features.pkl', 'wb') as f:
            pickle.dump(dev_features, f)
        with open(data_path + 'cache/test_features.pkl', 'wb') as f:
            pickle.dump(test_features, f)
    else:
        print('Loading embedding from cache...')
        train_features = pickle.load(open(data_path + 'cache/train_features.pkl', 'rb'))
        dev_features = pickle.load(open(data_path + 'cache/dev_features.pkl', 'rb'))
        test_features = pickle.load(open(data_path + 'cache/test_features.pkl', 'rb'))

    return train_features, dev_features, test_features, y_train, y_dev, y_test


def load_data(data_path: str) -> dict:
    data = {}
    for file in ['train', 'dev', 'test']:
        feature = []
        label = []
        essay_id = []
        essay_set = []
        read_data = pd.read_pickle(data_path + file + '.pkl')
        for i in range(len(read_data)):
            feature.append(read_data[i]['content_text'])
            label.append(int(read_data[i]['score']))
            essay_id.append(int(read_data[i]['essay_id']))
            essay_set.append(int(read_data[i]['prompt_id']))
        data[file] = {'feature': feature, 'label': label, 'essay_id': essay_id, 'essay_set': essay_set}

    return data

def run_embedding_model(data_loader: DataLoader, model, device) -> np.array:
    model.eval()
    progress_bar = tqdm(data_loader, desc="Create Embedding", unit="batch", ncols=100)
    with torch.no_grad():
        features = []
        for d in progress_bar:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)
            features.extend(outputs.last_hidden_state[:, 0, :].cpu().tolist())
    return np.array(features)


class EssayDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
    

# データローダーの定義
def create_data_loader(text, tokenizer, max_length, batch_size):
    ds = EssayDataset(
        texts=np.array(text),
        tokenizer=tokenizer,
        max_length=max_length
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)