import torch
import torch.nn as nn
from .Model import Model

class RESCAL(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100, new=False):
		super(RESCAL, self).__init__(ent_tot, rel_tot)

		self.dim = dim
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.ent2_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_matrices = nn.Embedding(self.rel_tot, self.dim * self.dim)
		self.new = new

		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform_(self.ent2_embeddings.weight.data)
		nn.init.xavier_uniform_(self.rel_matrices.weight.data)
	
	def _calc(self, h, t, r):
		t = t.view(-1, self.dim, 1)
		r = r.view(-1, self.dim, self.dim)
		tr = torch.matmul(r, t)
		tr = tr.view(-1, self.dim)
		return -torch.sum(h * tr, -1)

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		if self.new:
			t = self.ent2_embeddings(batch_t)
		else:
			t = self.ent_embeddings(batch_t)
		r = self.rel_matrices(batch_r)
		score = self._calc(h ,t, r)
		return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		if self.new:
			t = self.ent2_embeddings(batch_t)
		else:
			t = self.ent_embeddings(batch_t)
		r = self.rel_matrices(batch_r)
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		score = -self.forward(data)
		return score.cpu().data.numpy()