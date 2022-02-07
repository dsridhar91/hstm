from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
from sklearn import metrics
from sklearn.linear_model import Ridge
import sys
from scipy.special import expit, softmax
import itertools as it
from scipy.sparse import csr_matrix
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_coccurrence(counts):
	tf = csr_matrix(counts)
	cooccurence = tf.T.dot(tf)
	cooccurence = cooccurence.toarray()
	return cooccurence

def format_as_latex(list_of_topics):
	n_topics = len(list_of_topics)
	latex_str = ''
	for k in range(n_topics):
		latex_str += 'Topic ' + str(k) + '&' + '&'.join(list_of_topics[k])
		latex_str += '\\\\' + '\n'
	return latex_str

class Evaluator():
	def __init__(self, model, vocab, test_counts, test_labels, texts, model_name='') :
		self.model = model
		self.model_name = model_name
		self.vocab = vocab
		self.test_counts = test_counts
		self.test_labels = test_labels
		self.texts = texts

		self.cache_model_params()
		self.produce_theta()

	def cache_model_params(self):
		self.model.eval()
		with torch.no_grad():
			self.model.set_beta()
			self.topics = self.model.get_beta().cpu().detach().numpy()
			self.logit_topics = self.model.get_logit_beta().cpu().detach().numpy()

			if self.model_name == 'slda':
				self.weights = self.model.weights.weight.squeeze().cpu().detach().numpy()
			elif self.model_name == 'prodlda':
				self.base_rates = self.model.base_rates.squeeze().cpu().detach().numpy()
			else:
				self.gammas = self.model.gammas.t().cpu().detach().numpy()
				self.bow_weights = self.model.bow_weights.weight.squeeze().cpu().detach().numpy()
				self.base_rates = self.model.base_rates.squeeze().cpu().detach().numpy()
				self.topic_weights = self.model.topic_weights.weight.squeeze().cpu().detach().numpy()

	def produce_theta(self):
		normalized = self.test_counts/self.test_counts.sum(axis=-1)[:,np.newaxis]
		self.normalized_bow = torch.tensor(normalized, dtype=torch.float).to(device)

		self.model.eval()
		with torch.no_grad():
			theta, _ = self.model.get_theta(self.normalized_bow)
			self.theta = theta.cpu().detach().numpy()
		self.assigned_topics = self.theta.argsort(axis=1)

	def shuffle_gammas(self):
		shuffled_gammas = self.gammas.copy()
		np.random.shuffle(shuffled_gammas)
		return shuffled_gammas
		
	def visualize_topics(self, num_words=7, format_pretty=False):
		print("---"*60)
		topics_list = []
		num_topics = self.topics.shape[0]
		for k in range(num_topics):
			beta = self.logit_topics[k,:] #self.topics[k,:]
			top_words = (-beta).argsort()[:num_words]

			if not format_pretty:
				topic_words = [(self.vocab[t], beta[t]) for t in top_words]
				print('Topic {}: {}'.format(k, topic_words))
			else:
				topic_words = [self.vocab[t] for t in top_words]
				topics_list.append(topic_words)
				print('Topic {}: {}'.format(k, ' '.join(topic_words)))
		print("---"*60)
		latex_str = format_as_latex(topics_list)
		return latex_str

	def get_topics(self):
		num_topics = self.logit_topics.shape[0]
		topics = np.zeros((num_topics, self.vocab.shape[0]))
		for k in range(num_topics):
			topic = self.logit_topics[k,:]
			topics[k,:] = (-topic).argsort()
		return topics

	def visualize_word_weights(self, num_words=10):
		most_positive_words = self.bow_weights.argsort()[-num_words:]
		most_negative_words = self.bow_weights.argsort()[:num_words]

		print("---"*60)
		print('Overall pro: {}'.format(self.vocab[most_positive_words]))
		print('Overall anti: {}'.format(self.vocab[most_negative_words]))
		print("---"*60)

		latex_str = ''
		latex_str += ', '.join(self.vocab[most_positive_words])
		latex_str += '\\\\' + '\n'
		latex_str += ', '.join(self.vocab[most_negative_words])
		latex_str += '\\\\' + '\n'
		return latex_str

	def visualize_supervised_topics(self, num_words=7, normalize=True, pos_topics=True, format_pretty=False, compare_to_bow=False):
		num_topics = self.topics.shape[0]
		topics_list=[]
		bow = self.bow_weights

		print("---"*60)
		for k in range(num_topics):
			gamma = self.gammas[k,:]
			beta = self.logit_topics[k,:]

			if normalize:
					score = (gamma/beta)
			else:
				score = gamma
			if pos_topics:
				top_words = (-score).argsort()[:num_words]
			else:
				top_words = score.argsort()[:num_words]
			
			if not format_pretty:
				if not compare_to_bow:
					topic_words = [(self.vocab[t], score[t]) for t in top_words]
				else:
					topic_words = [(self.vocab[t], bow[t]) for t in top_words]
					
				print('Topic {}: {}'.format(k, topic_words))

			else:
				topic_words = [self.vocab[t] for t in top_words]
				topics_list.append(topic_words)
				print('Topic {}: {}'.format(k, ' '.join(topic_words)))
		print("---"*60)

		latex_str = format_as_latex(topics_list)
		return latex_str


	def get_latex_for_topics(self, num_words=7, normalize=True, num_topics=None):
		latex_str = ''

		if num_topics is None:
			num_topics = self.topics.shape[0]

		for k in range(num_topics):
			gamma = self.gammas[k,:]
			beta = self.logit_topics[k,:]

			if normalize:
					score = (gamma/beta)
			else:
				score = gamma

			pos_top_words = (-score).argsort()[:num_words]

			neg_top_words = score.argsort()[:num_words]
			neu_top_words = (-beta).argsort()[:num_words]

			latex_str += 'Pro:' + '&' + ', '.join([self.vocab[t] for t in pos_top_words])
			latex_str += '\\\\' + '\n'
			latex_str += 'Neutral:' +  '&' + ', '.join([self.vocab[t] for t in neu_top_words])
			latex_str += '\\\\' + '\n'
			latex_str += 'Anti:' + '&' + ', '.join([self.vocab[t] for t in neg_top_words])
			latex_str += '\\\\' + '\n'
		return latex_str


	def get_similar_words(self):
		word_embedding = self.theta.sum(axis=0) * self.logit_topics.T
		unnorm_sim = word_embedding.dot(word_embedding.T)
		sim = softmax(unnorm_sim, axis=1)
		return sim


	def get_matched_samples(self, n_examples=10, similarity='beta_only'):
		occurence = self.test_counts.copy()
		occurence[occurence > 1] = 1
		if similarity == 'beta_only':
			score = occurence.dot(self.logit_topics.T)
		else:
			score = occurence.dot((self.logit_topics * self.gammas).T)
			# score = occurence.dot(self.gammas.T)
		corr = np.corrcoef(score)
		
		indices = np.arange(self.test_counts.shape[0])
		np.random.shuffle(indices)
		count = 0
		i = 0
		while((count < n_examples) and (i < indices.shape[0])):
			idx = corr[i,:].argsort()[-2]
			sim = corr[i][idx]
			if sim >= 0.8:
				print("Text:", self.texts[i], '\n')	
				print("Matched text:", self.texts[idx], "Similarity:", sim , '\n\n')
				count+=1
			i+=1

	def show_text_features(self, index, n_words=10):
		document = self.test_counts[index,:]
		terms = document > 0
		counts = document[terms]
		freq = counts/counts.sum()
		valid_vocab = self.vocab[terms]

		gamma_beta = self.gammas[:,terms]*self.logit_topics[:,terms]
		gamma_scores = freq*(self.theta[index,:].dot(gamma_beta))
		bow_scores = self.bow_weights[terms]*freq

		bow_weights = np.abs(bow_scores).argsort()
		gamma_weights = np.abs(gamma_scores).argsort()

		top_bow = bow_weights[-n_words:]
		top_gamma = gamma_weights[-n_words:]

		print("Document:", self.texts[index])
			
		print("Words in this document with biggest BOW weights:", [(valid_vocab[b], round(bow_scores[b],3)) for b in top_bow])
		bow_tex = 'Words in doc w/ largest $\\omega$' + '&'.join(valid_vocab[top_bow])

		print("Words in this document with biggest gamma weights:", [(valid_vocab[b], round(gamma_scores[b],3)) for b in top_gamma])
		gamma_tex = 'Words in doc w/ largest $\\gamma$' + '&'.join(valid_vocab[top_gamma])


	def get_perplexity(self):
		bow = torch.tensor(self.test_counts, dtype=torch.float).to(device)
		self.model.eval()
		with torch.no_grad():
			theta, kl = self.model.get_theta(self.normalized_bow)
			log_prob = self.model.decode(theta)
			recon_loss = -(log_prob * bow).mean(1)
			loss = (recon_loss+ kl).mean()
			loss = loss.cpu().detach().numpy()
			average_perplexity = np.exp(loss)
		return average_perplexity

	def get_normalized_pmi_df(self, num_words=10, topics_to_use='neu'):
		num_topics = self.topics.shape[0]
		num_docs = self.test_counts.shape[0]
		per_topic_npmi = np.zeros(num_topics)
		
		counts = self.test_counts.copy()
		counts[counts>1] = 1

		cooccurence = get_coccurrence(counts)
		doc_count = counts.sum(axis=0)
		prob = doc_count/num_docs
		cooccurence_prob = cooccurence/num_docs

		for k in range(num_topics):
			npmi_total = 0
			if topics_to_use=='neu':
				beta = self.logit_topics[k,:] #self.topics[k,:]
			elif topics_to_use=='pos':
				beta = self.gammas[k,:]/self.logit_topics[k,:]
			else:
				beta = -self.gammas[k,:]/self.logit_topics[k,:]

			top_words = (-beta).argsort()[:num_words]
			n = 0 
			for (w1, w2) in it.combinations(top_words, 2):
				joint = cooccurence_prob[w1][w2]+1e-7
				p_w1 = prob[w1]+1e-7
				p_w2 = prob[w2]+1e-7
				numerator = np.log(joint/(p_w1*p_w2))
				denom = -np.log(joint)
				npmi_total += numerator/denom
				n+=1
			per_topic_npmi[k] = npmi_total #/ n
		return per_topic_npmi.mean()	

	def get_normalized_pmi(self, num_words=10):
		num_topics = self.topics.shape[0]
		num_docs = self.test_counts.shape[0]
		per_topic_npmi = np.zeros(num_topics)
		cooccurence = self.get_coccurrence_frequencies()
		count = self.test_counts.sum(axis=0)
		prob = count/count.sum()
		cooccurence_prob = cooccurence/cooccurence.sum()

		for k in range(num_topics):
			npmi_total = 0
			beta = self.logit_topics[k,:] #self.topics[k,:]
			top_words = (-beta).argsort()[:num_words]
			n = 0 
			for (w1, w2) in it.combinations(top_words, 2):
				joint = cooccurence_prob[w1][w2]+1e-6
				p_w1 = prob[w1]+1e-6
				p_w2 = prob[w2]+1e-6
				numerator = np.log(joint/(p_w1*p_w2))
				denom = -np.log(joint)
				npmi_total += numerator/denom
				n+=1
			per_topic_npmi[k] = npmi_total #/ n
		return per_topic_npmi.mean()

	def get_coccurrence_frequencies(self):
		tf = csr_matrix(self.test_counts)
		cooccurence = tf.T.dot(tf)
		cooccurence = cooccurence.toarray()
		return cooccurence

	def show_smoothing_params(self):
		self.model.eval()
		with torch.no_grad():
			self.smoothing_weights = self.model.smoothing.squeeze().detach().numpy()
			self.attention = self.model.attn.detach().numpy()
		n_topics = self.gammas.shape[0]
		for k in range(n_topics):
			beta = self.topics[k,:]
			idx = beta.argsort()[-1]
			most_similar = self.attention[idx,:].argsort()[-5:]
			print("Topic", k, "Word:", self.vocab[idx], "Most similar words:", self.vocab[most_similar])