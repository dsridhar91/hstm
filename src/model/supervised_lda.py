import torch
import torch.nn.functional as F 
import numpy as np 
import math 

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_l1_loss(x, dim=0, C=0.01):
    l1_loss = nn.L1Loss(reduction='sum')
    size = x.size()[dim]
    target = torch.zeros(size)
    num_classes = x.size()[dim+1]

    loss = 0
    for i in range(num_classes):
        weights = x[:,i]
        loss += C*l1_loss(weights,target)
    return loss

class SupervisedLDA(nn.Module):
    def __init__(self, num_topics, vocab_size, num_documents, t_hidden_size=300, theta_act='relu', label_is_bool=False, beta_init=None, predict_with_z=False, predict_with_bow=False):
        super(SupervisedLDA, self).__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.num_documents = num_documents
        self.t_hidden_size = t_hidden_size
        self.theta_act = self.get_activation(theta_act)
        self.predict_with_z = predict_with_z
        self.predict_with_bow = predict_with_bow
        self.is_bool = label_is_bool
        
        if beta_init is not None:
            self.betas = nn.Parameter(torch.tensor(beta_init, dtype=torch.float))
        else:
            self.betas = nn.Parameter(torch.randn(vocab_size, num_topics))

        self.weights = nn.Linear(num_topics, 1)
        self.bow_weights = nn.Linear(vocab_size, 1)

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
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def set_beta(self):
        self.beta = F.softmax(self.betas, dim=0).transpose(1, 0) ## softmax over vocab dimension 

    def get_beta(self):
        return self.beta

    def get_logit_beta(self):
        return self.betas.t()

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1) 
        return theta, kld_theta

    def decode(self, theta):
        self.set_beta()
        beta = self.get_beta()
        self.prob = torch.mm(theta,beta)
        preds = torch.log(self.prob+1e-6)
        return preds

    def predict_labels(self, theta, bow):
        beta = self.get_beta()
        if self.predict_with_z:
            normalizer = self.prob.unsqueeze(1)
            expected_z = torch.einsum('ik,kj->ikj', [theta, beta]) / normalizer
            features = expected_z.mean(dim=-1)
        else:
            features = theta
        
        features = torch.log(features+1e-6)
        expected_pred = self.weights(features).squeeze() 
        
        if self.predict_with_bow:
            expected_pred += self.bow_weights(bow).squeeze()
        
        return expected_pred
        
    def forward(self, bows, normalized_bows, labels, theta=None, aggregate=True):
        ## get \theta
        theta, kld_theta = self.get_theta(normalized_bows)
        
        # if not self.is_beta_init:
        # self.set_beta()

        if self.is_bool:
            loss = nn.BCEWithLogitsLoss()
        else:
            loss = nn.MSELoss()

        preds = self.decode(theta)
        recon_loss = -(preds * bows).sum(1)
        expected_label_pred = self.predict_labels(theta, normalized_bows).squeeze()
        supervised_loss = loss(expected_label_pred, labels)

        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, supervised_loss, kld_theta

