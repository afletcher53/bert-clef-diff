import json
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def title_abstract(data):
    data = json.loads(data)
    return data['title'] + " " + data['abstract']

def preprocess_text(text, tokenizer, max_length=None):
    tokens = tokenizer.tokenize(text)
    if max_length is not None:
        tokens = tokens[:max_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    preprocessed_text = ' '.join([token[2:] if token.startswith("##") else token for token in tokens])
    return preprocessed_text.strip()

def process_batch(batch, tokenizer, model, device):
    original_texts = [title_abstract(datapoint) for datapoint in batch]
    preprocessed_full = [preprocess_text(text, tokenizer) for text in original_texts]
    
    encoded_512 = tokenizer(original_texts, max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model(encoded_512['input_ids'], attention_mask=encoded_512['attention_mask'])
    
    decoded_512 = tokenizer.batch_decode(encoded_512['input_ids'], skip_special_tokens=True)
    
    results = []
    for i, (orig, full, decoded) in enumerate(zip(original_texts, preprocessed_full, decoded_512)):
        full = full.replace(" ' ", "'")
        for punct in ['.', ',']:
            full = full.replace(f' {punct}', punct)
        
        if decoded != full:
            full_words = full.split()
            decoded_words = decoded.split()
            diff = len(full_words) - len(decoded_words)
            percent_full = diff / len(full_words) * 100
        else:
            diff = 0
            percent_full = 0
        
        results.append({'id': i, 'percent_full': percent_full, 'diff': diff})
    
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()

    with open('all_clean.jsonl', 'r') as f:
        data = f.readlines()

    dataset = TextDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=cpu_count())

    results = []
    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_results = process_batch(batch, tokenizer, model, device)
        results.extend(batch_results)

    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()