import sys
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
from util import get_cv_split_assignments
from absl import flags
from absl import app

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
		# self.l2 = torch.nn.Dropout(0.3)
		self.l3 = torch.nn.Linear(768, 1)
	
	def forward(self, ids, mask):
		output_1 = self.l1(ids, mask)[0]
		# output_2 = self.l2(output_1.mean(1))
		output = self.l3(output_1.mean(1))
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
			print(f'Epoch: {epoch}, Loss:  {loss.item()}', flush=True)
		
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

def main(argv):
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	df = pd.read_csv(FLAGS.in_file)

	label_is_bool = False
	if FLAGS.data in TextResponseDataset.CLASSIFICATION_SETTINGS:
		label_is_bool=True

	if FLAGS.train_test_mode:
		train_dataset = df.iloc[:FLAGS.train_size].reset_index(drop=True)
		test_dataset = df.iloc[FLAGS.train_size:].reset_index(drop=True)
	else:
		n_docs = df.shape[0]
		df['split'] = get_cv_split_assignments(n_docs, num_splits=FLAGS.n_folds)
		train_dataset = df[df.split!=FLAGS.split].reset_index(drop=True)
		test_dataset = df[df.split==FLAGS.split].reset_index(drop=True)

	training_set = ProcDataset(train_dataset, tokenizer, FLAGS.max_len)
	testing_set = ProcDataset(test_dataset, tokenizer, FLAGS.max_len)

	train_params = {'batch_size': FLAGS.batch_size,
					'shuffle': True,
					'num_workers': 0
					}

	test_params = {'batch_size': FLAGS.batch_size,
					'shuffle': True,
					'num_workers': 0
					}

	training_loader = DataLoader(training_set, **train_params)
	testing_loader = DataLoader(testing_set, **test_params)

	model = BERTClass()
	model.to(device)

	for epoch in range(FLAGS.epochs):
		train(model, training_loader, epoch, lr=FLAGS.learning_rate, label_is_bool=label_is_bool)

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
	os.makedirs(FLAGS.outdir, exist_ok=True)
	np.save(os.path.join(FLAGS.outdir, 'bert.result.' + 'split'+str(FLAGS.split)), np.array([mse, auc, log_loss, accuracy]))

if __name__ == '__main__':
	FLAGS = flags.FLAGS
	flags.DEFINE_string("in_file", "", "path to processed data file that contains pairs of (untokenized) text and label.")
	flags.DEFINE_string("data", "amazon", "name of dataset")
	flags.DEFINE_string("outdir", "../out/", "path to directory where output is saved.")

	flags.DEFINE_integer("split", 0, "for cross validation, indicates which split is used as the test split.")
	flags.DEFINE_integer("n_folds", 10, "for cross validation, number of splits (i.e., folds) to use.")
	flags.DEFINE_integer("train_size", 10000, "number of samples to set aside for training split (only valid if train/test setting is used)")
	flags.DEFINE_integer("max_len", 128, "max length of sequences -- longer sequences will be trimmed.")
	flags.DEFINE_integer("batch_size", 32, "batch size for training and evaluation.")
	flags.DEFINE_integer("epochs", 5, "number of full passes to do over dataset when training.")

	flags.DEFINE_float("learning_rate", 1e-5, "optimization learning rate.")
	
	flags.DEFINE_boolean("train_test_mode", False, "flag to use to run a train/test experiment instead of cross validation (default).")
	app.run(main)