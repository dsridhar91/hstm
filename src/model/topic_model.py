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

class TopicModel(nn.Module):
    def __init__(self, num_topics, vocab_size, num_documents, t_hidden_size=300, theta_act='relu', C=1e-4, beta_init=None):
        super(TopicModel, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.num_documents = num_documents
        self.t_hidden_size = t_hidden_size
        self.theta_act = self.get_activation(theta_act)
        self.C_weights = C
        
        if beta_init is not None:
            self.betas = nn.Parameter(torch.tensor(beta_init), dtype=torch.float, device=device)
        else:
            self.betas = nn.Parameter(torch.randn(vocab_size, num_topics))

        self.base_rates = nn.Parameter(torch.randn(vocab_size, 1))

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
        self.normalized_beta = F.softmax(self.betas, dim=0).transpose(1, 0)

    def get_beta(self):
        return self.normalized_beta

    def get_logit_beta(self):
        return self.betas.t()

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1) 
        return theta, kld_theta

    def decode(self, theta):
        beta = self.get_logit_beta()
        logits = torch.mm(theta,beta)
        logits += self.base_rates.squeeze(1)
        res = F.softmax(logits, dim=-1)
        preds = torch.log(res+1e-6)
        return preds

        
    def forward(self, bows, normalized_bows, theta=None):
        theta, kld_theta = self.get_theta(normalized_bows)
        preds = self.decode(theta)
        recon_loss = -(preds * bows).sum(1)
        penalty = get_l1_loss(self.base_rates, C=self.C_weights)
        recon_loss = recon_loss.mean()
        return recon_loss, penalty, kld_theta

