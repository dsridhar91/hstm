import os
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import numpy as np
import argparse
from sklearn import metrics
from data.dataset import TextResponseDataset

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

class ProcDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        text = str(self.data.text[index])
        text=" ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.label[index], dtype=torch.float)
        } 
    
    def __len__(self):
        return self.len


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)
    
    def forward(self, ids, mask):
        output_1 = self.l1(ids, mask)[0]
        output_2 = self.l2(output_1[:,0]) #self.l2(output_1.mean(1))
        output = self.l3(output_2)
        return output


def train(model, training_loader, epoch, lr=1e-5, label_is_bool=True):
    if label_is_bool:
        loss_function = torch.nn.BCEWithLogitsLoss()
    else:
        loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr=lr)
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask).squeeze()
        optimizer.zero_grad()
        loss = loss_function(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def valid(model, testing_loader, label_is_bool=False):
    model.eval()
    eval_targets = []
    eval_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask).squeeze()
            if label_is_bool:
                outputs = torch.sigmoid(outputs)
            eval_targets += targets.cpu().detach().numpy().tolist()
            eval_outputs += outputs.cpu().detach().numpy().tolist()
    return np.array(eval_outputs), np.array(eval_targets)

def main(seed=10):
    np.random.seed(seed)
    df = pd.read_csv(in_file)
    df['split'] = np.random.randint(0,10,size=df.shape[0])

    label_is_bool = False
    if dataset in TextResponseDataset.CLASSIFICATION_SETTINGS:
        label_is_bool=True

    train_dataset = df[df.split!=split].reset_index(drop=True)
    test_dataset = df[df.split==split].reset_index(drop=True)
    print("Training data:", train_dataset.shape)
    print("Test data:", test_dataset.shape)
    
    training_set = ProcDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = ProcDataset(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = BERTClass()
    model.to(device)

    for epoch in range(EPOCHS):
        train(model, training_loader, epoch, lr=LEARNING_RATE, label_is_bool=label_is_bool)

    print("testing..")
    eval_outputs, eval_targets = valid(model, testing_loader, label_is_bool=label_is_bool)
    log_loss = mse = accuracy = auc = 0.0

    if label_is_bool:
        auc = metrics.roc_auc_score(eval_targets, eval_outputs)
        log_loss = metrics.log_loss(eval_targets, eval_outputs)
        preds = (eval_outputs >= 0.5)
        accuracy = metrics.accuracy_score(eval_targets, preds)

    else:
        mse = metrics.mean_squared_error(eval_targets, eval_outputs)

    print("MSE:", mse, "Acc:", accuracy, "Log loss:", log_loss, "AUC:", auc)
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'bert.result.' + 'split'+str(split)), np.array([mse, auc, log_loss, accuracy]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", action="store", default="")
    parser.add_argument("--data", action="store", default="amazon")
    parser.add_argument("--split", action='store', default=0)
    parser.add_argument("--n-folds", action='store', default=10)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument('--outdir', action='store', default='../out/')
    args = parser.parse_args()
    
    dataset = args.data
    in_file = args.in_file
    outdir = args.outdir
    split = int(args.split)
    n_folds = int(args.n_folds)
    is_load_mode = bool(args.load)
    is_save_mode = bool(args.save)
    
    if in_file == "":
        # in_file = '/proj/sml/projects/text-response/csv_proc/' + dataset + '.csv'
        in_file = '../dat/csv_proc/' + dataset + '.csv'

    model_dir = '../out/model/'
    os.makedirs(model_dir, exist_ok=True)
    model_file = model_dir + dataset +'.bert-model.' + str(split)

    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-05
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    main()
