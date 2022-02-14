import torch
import torch.nn.functional as F 
import numpy as np 
import math 

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_l1_loss(x, dim=0, C=0.01):
	l1_loss = nn.L1Loss(reduction='sum')
	size = x.size()[dim]
	target = torch.zeros(size, device=device)
	num_classes = x.size()[dim+1]

	loss = 0
	for i in range(num_classes):
		weights = x[:,i]
		loss += C*l1_loss(weights,target)
	return loss

class HeterogeneousSupervisedTopicModel(nn.Module):
	def __init__(self, num_topics, vocab_size, num_documents, t_hidden_size=300, enc_drop=0.0, theta_act='relu', label_is_bool=False, beta_init=None, C_weights=5e-4, C_topics=5e-6, response_model='hstm-all'):
		super(HeterogeneousSupervisedTopicModel, self).__init__()

		## define hyperparameters
		self.num_topics = num_topics
		self.vocab_size = vocab_size
		self.num_documents = num_documents
		self.t_hidden_size = t_hidden_size
		self.theta_act = self.get_activation(theta_act)
		self.enc_drop = enc_drop
		self.t_drop = nn.Dropout(enc_drop)
		self.C_topics=C_topics
		self.C_weights = C_weights
		self.C_base_rates = C_weights
		self.response_model = response_model

		if beta_init is not None:
			self.logit_betas = nn.Parameter(torch.tensor(beta_init, dtype=torch.float))
		else:
			self.logit_betas = nn.Parameter(torch.randn(vocab_size, num_topics))

		self.gammas = nn.Parameter(torch.randn(vocab_size, num_topics))
		self.base_rates = nn.Parameter(torch.randn(vocab_size, 1))
		self.bow_weights = nn.Linear(vocab_size, 1)
		self.topic_weights = nn.Linear(num_topics, 1)
	
		self.q_theta = nn.Sequential(
				nn.Linear(vocab_size, t_hidden_size), 
				self.theta_act,
				nn.BatchNorm1d(t_hidden_size),
				nn.Linear(t_hidden_size, t_hidden_size),
				self.theta_act,
				nn.BatchNorm1d(t_hidden_size)
			)
		self.mu_q_theta = nn.Linear(t_hidden_size, num_topics)
		self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics)

		self.is_bool = label_is_bool

		self.smoothing = nn.Parameter(torch.randn(vocab_size, 1))

	def get_activation(self, act):
		if act == 'tanh':
			act = nn.Tanh()
		elif act == 'relu':
			act = nn.ReLU()
		elif act == 'softplus':
			act = nn.Softplus()
		elif act == 'rrelu':
			act = nn.RReLU()
		elif act == 'leakyrelu':
			act = nn.LeakyReLU()
		elif act == 'elu':
			act = nn.ELU()
		elif act == 'selu':
			act = nn.SELU()
		elif act == 'glu':
			act = nn.GLU()
		else:
			print('Defaulting to tanh activations...')
			act = nn.Tanh()
		return act 

	def reparameterize(self, mu, logvar):
		"""Returns a sample from a Gaussian distribution via reparameterization.
		"""
		if self.training:
			std = torch.exp(0.5 * logvar) 
			eps = torch.randn_like(std)
			return eps.mul_(std).add_(mu)
		else:
			return mu

	def encode(self, bows):
		"""Returns paramters of the variational distribution for \theta.

		input: bows
				batch of bag-of-words...tensor of shape bsz x V
		output: mu_theta, log_sigma_theta
		"""
		q_theta = self.q_theta(bows)
		if self.enc_drop > 0:
			q_theta = self.t_drop(q_theta)
		mu_theta = self.mu_q_theta(q_theta)
		logsigma_theta = self.logsigma_q_theta(q_theta)
		kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
		return mu_theta, logsigma_theta, kl_theta

	def set_beta(self):
		self.betas = F.softmax(self.logit_betas, dim=0).transpose(1, 0) ## softmax over vocab dimension

	def get_beta(self):
		return self.betas

	def get_logit_beta(self):
		return self.logit_betas.t()

	def get_theta(self, normalized_bows):
		mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
		z = self.reparameterize(mu_theta, logsigma_theta)
		theta = F.softmax(z, dim=-1)
		return theta, kld_theta

	def decode(self, theta):
		logits = torch.mm(theta, self.logit_betas.t()) 
		logits += self.base_rates.squeeze(1)
		res = F.softmax(logits, dim=-1)
		preds = torch.log(res+1e-6)
		return preds

	def predict_labels(self, theta, bows):
		# gammas = self.apply_attention_smoothing(theta)
		gammas = self.gammas
		scaled_beta = self.logit_betas * gammas
		weights = torch.mm(theta, scaled_beta.t())

		if self.response_model == 'stm':
			expected_pred = self.topic_weights(theta).squeeze()
		elif self.response_model == 'stm+bow':
			expected_pred = self.topic_weights(theta).squeeze() + self.bow_weights(bows).squeeze()
		elif self.response_model == 'hstm':
			expected_pred = (bows * weights).sum(1)
		elif self.response_model == 'hstm+bow':
			expected_pred = (bows * weights).sum(1) + self.bow_weights(bows).squeeze()
		elif self.response_model == 'hstm+topics':
			expected_pred = (bows * weights).sum(1) + self.topic_weights(theta).squeeze()
		elif self.response_model == 'hstm-all' or self.response_model == 'hstm-all-2stage':
			expected_pred = (bows * weights).sum(1)\
							+ self.bow_weights(bows).squeeze() + self.topic_weights(theta).squeeze()
		elif self.response_model == 'hstm-nobeta':
			no_beta_weights = torch.mm(theta, self.gammas.t())
			expected_pred = (bows * no_beta_weights).sum(1)\
							+ self.bow_weights(bows).squeeze() + self.topic_weights(theta).squeeze()
		return expected_pred

		
	def forward(self, bows, normalized_bows, labels, theta=None, do_prediction=True, penalty_bow=True, penalty_gamma=True):
		if self.is_bool:
			loss = nn.BCEWithLogitsLoss()
		else:
			loss = nn.MSELoss()

		other_loss = torch.tensor([0.0], dtype=torch.float, device=device)

		if theta is None:
			theta, kld_theta = self.get_theta(normalized_bows)
			preds = self.decode(theta)
			recon_loss = -(preds * bows).sum(1).mean()
			other_loss = get_l1_loss(self.base_rates, C=self.C_weights)
		else:
			recon_loss = torch.tensor([0.0], dtype=torch.float, device=device)
			kld_theta = torch.tensor([0.0], dtype=torch.float, device=device)

		if do_prediction:
			expected_label_pred = self.predict_labels(theta, normalized_bows)
			other_loss += loss(expected_label_pred, labels)
			if penalty_gamma:
				other_loss += get_l1_loss(self.gammas, C=self.C_topics)
			if penalty_bow:
				other_loss += self.C_weights*(torch.norm(self.bow_weights.weight))

		return recon_loss, other_loss, kld_theta

