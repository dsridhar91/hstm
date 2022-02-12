import os
import json
import numpy as np
import pandas as pd
import gzip
import argparse
from collections import Counter


def load_mixed_corpus():
	grocery_file = '../dat/reviews_Grocery_and_Gourmet_Food_5.json'
	office_file = '../dat/reviews_Office_Products_5.json'
	doc_groc, _ = load_amazon(grocery_file, 5000, 'reviewText', 'overall')
	doc_office, _ = load_amazon(office_file, 5000, 'reviewText', 'overall')
	responses_groc = np.ones(doc_groc.shape[0])
	responses_office = np.zeros(doc_office.shape[0])
	return np.hstack([doc_groc, doc_office]), np.hstack([responses_groc, responses_office])


def load_framing_corpus(data_dir, topic, annotation_code_file):
	data_file = os.path.join(data_dir, topic, topic + '_labeled.json')
	with open(data_file, 'r') as f:
		data = json.loads(f.read())

	with open(annotation_code_file, 'r') as f:
		annotation_dict = json.loads(f.read())

	docs = []
	responses = []

	for key, value in data.items():
		if 'tone' not in value['annotations']:
			continue
		else:
			tone_dict = value['annotations']['tone']
			if len(tone_dict.keys()) == 0:
				continue

			annotations = Counter()
			for annotator, annot_list in tone_dict.items():
				for annot_dict in annot_list:
					tone_code = annotation_dict[str(annot_dict['code'])]
					annotations.update([tone_code])
					
			majority_label = annotations.most_common()[0][0]
			if majority_label == 'Neutral':
				continue
			else:
				label = 1 if 'Pro' in majority_label else 0
				responses.append(label)
				text = value['text']
				docs.append(text)

	return np.array(docs), np.array(responses)

def load_yelp(data_file, subsample=None):
	df = pd.read_csv(data_file, names=['label', 'text'])
	df.loc[df.label==1, 'label'] = 0
	df.loc[df.label==2, 'label'] = 1
	if subsample is not None:
		indices = np.arange(df.shape[0])
		np.random.shuffle(indices)
		subsample_inds = indices[:subsample]
		df = df.iloc[subsample_inds, :]
	docs = df['text'].values
	responses = df['label'].values
	return docs, responses

def load_yelp_full(data_file):
	train = data_file + 'train.csv'
	test = data_file + 'test.csv'
	train_df = pd.read_csv(train, names=['label', 'text'])
	test_df = pd.read_csv(test, names=['label', 'text'])
	full_df = pd.concat([train_df, test_df])
	full_df.loc[full_df.label==1, 'label'] = 0
	full_df.loc[full_df.label==2, 'label'] = 1
	docs = full_df['text'].values
	responses = full_df['label'].values
	return docs, responses



def load_peerread(data_file):
	df = pd.read_csv(data_file)
	docs = df['abstract_text'].values
	responses = df['decision'].astype('int64').values
	return docs, responses

def load_semantic_scholar(data_file, min_year, max_year):
	papers = []
	with gzip.open(data_file, 'rb') as f:
		for line in f:
			paper = json.loads(line)
			papers.append(paper)

	papers = papers[0]

	docs = []
	responses = []
	for paper in papers:
		if 'year' in paper and 'inCitations' in paper and 'paperAbstract' in paper:
			year = int(paper['year'])
			if year >= min_year and year <= max_year:
				if len(paper['paperAbstract']) > 0:
					responses.append(len(paper['inCitations']))
					docs.append(paper['paperAbstract'])
	docs = np.array(docs)
	responses = np.array(responses)
	return docs, responses


def load_amazon(data_file, subsample, text_attr_key, label_key, make_bool=False):
	documents = []
	file_handle = open(data_file, 'r')
	for line in file_handle.readlines():
		doc = json.loads(line)
		documents.append(doc)
	
	review_indices = np.arange(len(documents))
	np.random.shuffle(review_indices)
	iter_indices = review_indices[:subsample]
	docs = []
	responses = []
	for idx in iter_indices:
		doc = documents[idx]
		if label_key in doc and text_attr_key in doc:
			docs.append(doc[text_attr_key])
			responses.append(doc[label_key])
	docs = np.array(docs)
	responses = np.array(responses)
	filtered = preprocess_ratings(responses)
	docs = docs[filtered]
	responses = responses[filtered]

	if make_bool:
		responses[responses <= 2.0] = 0
		responses[responses >= 4.0] = 1

	return docs, responses

def preprocess_ratings(ratings):
	valid_ratings = (ratings <= 2.0) | (ratings >= 4.0)
	return valid_ratings

def main(dataset, framing_topic):
	np.random.seed(seed)

	if dataset == 'amazon':
		if data_file = "":
			data_file = '../dat/reviews_Office_Products_5.json'
		doc,responses = load_amazon(data_file, 20000, 'reviewText', 'overall')

	elif dataset == 'amazon_binary':
		if data_file = "":
			data_file = '../dat/reviews_Grocery_and_Gourmet_Food_5.json'
		doc,responses = load_amazon(data_file, 20000, 'reviewText', 'overall', make_bool=True)

	elif dataset == 'yelp':
		if data_file = "":
			data_file = '../dat/yelp_review_polarity_csv/train.csv'
		doc, responses = load_yelp(data_file, 20000)

	elif dataset == 'yelp_full':
		if data_file = "":
			data_file = '../dat/yelp_review_polarity_csv/'
		doc, responses = load_yelp(data_file)

	elif dataset == 'peerread':
		if data_file = "":
			data_file = '../dat/peerread_abstracts.csv'
		doc, responses = load_peerread(data_file)

	elif dataset == 'framing_corpus':
		if data_file = "":
			data_file = '../dat/framing/' #+ framing_topic + '/'
		annotation_code_file = '../dat/framing/codes.json'
		doc, responses = load_framing_corpus(data_file, framing_topic, annotation_code_file)

	else:
		if data_file = "":
			data_file = '../dat/cs_papers.gz'
		doc, responses = load_semantic_scholar(data_file, 2010, 2016)

	if dataset == 'semantic_scholar':
		responses = np.log(responses + 1)

	if dataset == 'amazon' or dataset == 'semantic_scholar':
		responses = (responses - responses.mean()) / (responses.std())

	df = pd.DataFrame(np.column_stack((doc.T, responses.T)) , columns=['text', 'label'])
	os.makedirs('../dat/csv_proc', exist_ok=True)
	
	if dataset == 'framing_corpus':
		dataset = framing_topic
	
	df.to_csv('../dat/csv_proc/' + dataset + '.csv')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_file", action="store", default="")
	parser.add_argument("--data", action="store", default="amazon")
	parser.add_argument("--framing-topic", action='store', default='immigration')
	args = parser.parse_args()
	data_file = args.data_file
	dataset = args.data
	framing_topic = args.framing_topic
	seed = 12345

	main(dataset, framing_topic)