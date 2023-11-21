from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from tqdm.notebook import tqdm

import torch
import numpy as np
import pandas as pd

MAX_LEN = 256
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 40
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10

# convert df
def convert_df(df):
    # combine each sentence_id
    title = list(df.groupby("Record Number")['Title'].agg('first'))
    tags = list(df.groupby("Record Number")['Tag'].apply(list))
    train_df = pd.DataFrame({'Title':title, 'Tags':tags})
    return train_df

# dataset
class NERDataset(Dataset):
    def __init__(self, dataframe, label2id, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        
    def __getitem__(self, index):
        sentence = self.data.Title[index]
        word_labels = self.data.Tags[index]
        tokenized_sentence, labels = self._tokenize(sentence, word_labels)
        
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        labels.insert(0, "O")
        labels.append("O")
        
        if len(tokenized_sentence) > self.max_len:
            tokenized_sentence = tokenized_sentence[:self.max_len]
            labels = labels[:self.max_len]
        else:
            tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(self.max_len - len(tokenized_sentence))]
            labels = labels + ["O" for _ in range(self.max_len - len(labels))]
                                                               
        attn_mask = [1 if t != '[PAD]' else 0 for t in tokenized_sentence]
        
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        label_ids = [self.label2id[l] for l in labels]
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.data)
                                                       
    def _tokenize(self, sentence, labels):
        ts = []
        l = []
        s = sentence.strip()
        for word, label in zip(s.split(), labels):
            word = self.tokenizer.tokenize(word)
            ts += word
            l += [label] * len(word)
        return ts, l

def training_loop(model, dataloader, optimizer, device):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()

    pbar = tqdm(dataloader, 'loss: n/a')
    for idx, batch in enumerate(pbar):
        ids = batch['ids'].to(device, dtype = torch.long)
        mask = batch['mask'].to(device, dtype = torch.long)
        targets = batch['targets'].to(device, dtype = torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        loss, tr_logits = outputs.loss, outputs.logits
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)
        
        if idx % 100==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")
           
        # compute training accuracy
        flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
        active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tr_preds.extend(predictions)
        tr_labels.extend(targets)
        
        #tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
        #tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f'loss: {loss.item()}')

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    #print(f"Training loss epoch: {epoch_loss}")
    #print(f"Training accuracy epoch: {tr_accuracy}")
    return epoch_loss

def create_dl(train_df, df, tokenizer, train_batch_size=TRAIN_BATCH_SIZE):
    labels2id = {df['Tag'].unique()[i - 1]: i for i in range(1, len(df['Tag'].unique()) + 1)}
    labels2id.update({"O": 0})
    id2labels = {labels2id[i]: i for i in labels2id}
    train_ds = NERDataset(train_df, labels2id, tokenizer, MAX_LEN)
    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    
    return train_dl, labels2id, id2labels, train_ds
    
def ex(): # ex
    path = "Train_Tagged_Titles.tsv"
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)
    train_df = convert_df(df)
    train_dl, labels2id, id2labels, train_ds, create_dl = create_dl(train_df, df)
    
    device = torch.device('cpu')
    
    model = BertForTokenClassification.from_pretrained(
    'bert-base-german-cased', num_labels=len(labels2id), id2label=id2labels, label2id=labels2id)
    
    model.to(device)
    
    torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    pbar = tqdm(range(EPOCHS), desc="Loss: N/A")
    for e in enumerate(pbar):
        pbar.set_description(f'Loss: f{training_loop(model, train_dl, optimizer, device)}')
        if (e + 1) % 10 == 0:
            torch.save({'model_state_dict': model.state_dict(), 'optim_state_dict': optim.state_dict()}, f'bert_german_0_e{e + 1}.p')
    
    
    