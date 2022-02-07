import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
import pandas as pd
import gzip
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import sys
from collections import Counter
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import argparse

from data.load_data_from_text import load_peerread, load_semantic_scholar, load_amazon, load_yelp, load_framing_corpus, load_mixed_corpus

class LemmaTokenizer:
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		# return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if str.isalpha(t)]
		return [t for t in word_tokenize(doc) if str.isalpha(t)]

class TextResponseDataset(Dataset):
	def __init__(self, dataset_name, data_file, processed_data_file, **kwargs):
		super(Dataset, self).__init__()

		CLASSIFICATION_SETTINGS = {'peerread', 'yelp', 'yelp_full_tr', 'yelp_full_te' 'amazon_binary', 'framing_corpus', 'amazon_mixed'}
		self.dataset_name = dataset_name
		self.data_file = data_file
		self.processed_data_file = processed_data_file

		self.label_is_bool = False
		if self.dataset_name in CLASSIFICATION_SETTINGS:
			self.label_is_bool = True


		self.parse_args(**kwargs)
		self.process_dataset()
		self.preprocessing()

	def parse_args(self, **kwargs):
		self.min_year = int(kwargs.get('min_year', 2010))
		self.max_year = int(kwargs.get('max_year', 2016))
		self.subsample = int(kwargs.get('subsample', 20000))
		self.text_attr_key = kwargs.get('text_attr', 'reviewText')
		self.label_key = kwargs.get('label', 'overall')
		self.pretrained_theta = kwargs.get('pretrained_theta', None)
		self.framing_topic = kwargs.get('framing_topic', 'immigration')
		self.annotation_file = kwargs.get('annotation_file', '../dat/framing/codes.json')
		self.use_bigrams=bool(kwargs.get('use_bigrams', True))

	def load_data_from_raw(self):
		if self.dataset_name == 'peerread':
			docs, responses = load_peerread(self.data_file)
		elif self.dataset_name == 'amazon':
			docs, responses = load_amazon(self.data_file, self.subsample, self.text_attr_key, self.label_key)
		elif self.dataset_name == 'amazon_binary':
			docs, responses = load_amazon(self.data_file, self.subsample, self.text_attr_key, self.label_key, make_bool=True)
		elif self.dataset_name == 'semantic_scholar':
			docs, responses = load_semantic_scholar(self.data_file, self.min_year, self.max_year)
		elif self.dataset_name == 'yelp':
			docs, responses = load_yelp(self.data_file, subsample=self.subsample)
		elif self.dataset_name == 'yelp_full_tr' or 'yelp_full_te':
			docs, responses = load_yelp(self.data_file)
		elif self.dataset_name == 'framing_corpus':
			docs, responses = load_framing_corpus(self.data_file, self.framing_topic, self.annotation_file)
		elif self.dataset_name == 'amazon_mixed':
			docs, responses = load_mixed_corpus()
		return docs, responses

	def load_processed_data(self):
		arrays = np.load(self.processed_data_file, allow_pickle=True)
		labels = arrays['labels']
		counts = arrays['counts']
		vocab = arrays['vocab']
		docs = arrays['docs']

		return counts, labels, vocab, docs

	def get_vocab_size(self):
		return self.vocab.shape[0]

	def process_dataset(self):
		if os.path.exists(self.processed_data_file):
			counts, responses, vocab, docs = self.load_processed_data()
		else:
			docs, responses = self.load_data_from_raw()
			stop = stopwords.words('english')

			exclude1 = {'but'}

			stop1 = list(set(stop) - exclude1) 
			
			vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 1), stop_words=stop1, max_df=0.9, min_df=0.0007)
			counts = vectorizer.fit_transform(docs).toarray()
			vocab = vectorizer.get_feature_names()

			if self.use_bigrams:
				exclude2 = {'doesn','don', 'but', 'not', 'wasn', 'wouldn', 'couldn', 'didn', 'isn'} #{'not'}
				stop2 = list(set(stop) - exclude2) 
				bigram_vec = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop2, ngram_range=(2, 2), min_df=0.007)
				bigram_counts = bigram_vec.fit_transform(docs).toarray()
				counts = np.column_stack((counts, bigram_counts))
				bigrams = bigram_vec.get_feature_names()
				vocab = vocab+bigrams

			vocab = np.array(vocab)
			np.savez_compressed(self.processed_data_file, labels=responses, counts=counts, vocab=vocab, docs=docs)
			
		self.counts = counts
		self.vocab = vocab
		self.labels = responses
		self.docs = docs
		
	def preprocessing(self):
		term_total = self.counts.sum(axis=1)
		valid = (term_total > 1)
		self.labels = self.labels[valid]
		self.counts = self.counts[valid,:]
		self.docs = self.docs[valid]

		self.normalized_counts = self.counts / self.counts.sum(axis=1)[:,np.newaxis]

		if self.dataset_name == 'semantic_scholar':
			self.labels = np.log(self.labels + 1)

		if not self.label_is_bool:
			self.labels = (self.labels - self.labels.mean())/(self.labels.std())

	def assign_splits(self, tr_indices, te_indices):
		self.tr_counts = self.counts[tr_indices, :]
		self.tr_labels = self.labels[tr_indices]
		self.te_counts = self.counts[te_indices,:]
		self.te_labels = self.labels[te_indices]
		self.tr_docs = self.docs[tr_indices]
		self.te_docs = self.docs[te_indices]

		self.tr_normalized_counts = self.normalized_counts[tr_indices, :]
		self.te_normalized_counts = self.normalized_counts[te_indices, :]

		if self.pretrained_theta is not None:
			self.tr_pretrained_theta = self.pretrained_theta[tr_indices, :]
			self.te_pretrained_theta = self.pretrained_theta[te_indices,:]
		else:
			self.tr_pretrained_theta = None
			self.te_pretrained_theta = None

	def __getitem__(self, idx):
		datadict = {
				'normalized_bow':torch.tensor(self.tr_normalized_counts[idx,:], dtype=torch.float),
				'bow':torch.tensor(self.tr_counts[idx,:], dtype=torch.long),
				'label':torch.tensor(self.tr_labels[idx], dtype=torch.float)
			}
		if self.tr_pretrained_theta is not None:
			datadict.update({'pretrained_theta':torch.tensor(self.tr_pretrained_theta[idx,:], dtype=torch.float)})
		return datadict

	def __len__(self):
		return self.tr_counts.shape[0]

	def get_full_size(self):
		return self.counts.shape[0]


def main():
	proc_file = '../dat/proc/' + data + '_proc'
	dataset = TextResponseDataset(data, data_file, proc_file, framing_topic=framing_topic)
	dataset.process_dataset()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", action="store", default="amazon")
	parser.add_argument("--framing_topic", action='store', default='immigration')
	parser.add_argument("--data_file", action='store', default="../dat/reviews_Office_Products_5.json")

	args = parser.parse_args()
	data = args.data
	framing_topic = args.framing_topic
	data_file = args.data_file

	main()