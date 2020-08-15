import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
import numpy as np

class TransE(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None, new=False):
		super(TransE, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.new = new

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.rel2_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.ent2_embeddings = nn.Embedding(self.ent_tot, self.dim)

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel2_embeddings.weight.data)
			nn.init.xavier_uniform_(self.ent2_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.ent2_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel2_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False


	def _calc(self, hhh, ttt, rrr, mode, t2=None):
		if self.norm_flag:
			hhh = F.normalize(hhh, 2, -1)
			rrr = F.normalize(rrr, 2, -1)
			ttt = F.normalize(ttt, 2, -1)
			if t2 is not None:
				t2 = F.normalize(t2, 2, -1)
		if mode != 'normal':
			hhh = hhh.view(-1, rrr.shape[0], hhh.shape[-1])
			ttt = ttt.view(-1, rrr.shape[0], ttt.shape[-1])
			rrr = rrr.view(-1, rrr.shape[0], rrr.shape[-1])
			if t2 is not None:
				t2 = t2.view(-1, rrr.shape[0], t2.shape[-1])
		
		if mode == 'head_batch':
			score = hhh + (rrr - ttt)
			
		else:
			score = (hhh + rrr) - ttt
		# index = np.random.randint(0, 2000, 100).tolist()
		if t2 is not None:
			try:
				score += 0.9 * (ttt - t2)
				print("lol")
			except:
				pass
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		r = self.rel_embeddings(batch_r)
		h = self.ent_embeddings(batch_h)
		if self.new:
			t = self.ent2_embeddings(batch_t)
		else:
			t = self.ent_embeddings(batch_t)
		if self.new and len(batch_t) < 3000:
			score = self._calc(h, t, r, mode, self.ent_embeddings(batch_t))
		else:
			score = self._calc(h ,t, r, mode)

		if self.margin_flag:
			return self.margin - score
		else:
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
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2)) / 3
		return regul

	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()