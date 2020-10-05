import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class TransE(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None, weight1=.0, weight2=.0):
		super(TransE, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.weight1 = weight1
		self.weight2 = weight2

		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		self.hr_linear1 = nn.Linear(dim, dim)
		self.hr_linear2 = nn.Linear(dim, dim)
		self.rt_linear1 = nn.Linear(dim, dim)
		self.rt_linear2 = nn.Linear(dim, dim)

		nn.init.xavier_uniform_(self.hr_linear1.weight.data)
		nn.init.xavier_uniform_(self.rt_linear1.weight.data)
		nn.init.xavier_uniform_(self.hr_linear2.weight.data)
		nn.init.xavier_uniform_(self.rt_linear2.weight.data)

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
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
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False


	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	def _calc2(self, x, y):
		if self.norm_flag:
			x = F.normalize(x, 2, -1)
			y = F.normalize(y, 2, -1)
		score = (x - y)
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)

		h_hr = self.hr_linear1(h)
		r_hr = self.hr_linear2(r)
		t_rt = self.rt_linear1(t)
		r_rt = self.rt_linear2(r)

		score = self._calc(h ,t, r, mode)
		score1 = self._calc2(h_hr, r_hr)
		score2 = self._calc2(t_rt, r_rt)


		final_score = (1 - self.weight1 - self.weight2) * score + self.weight1 * score1 + self.weight2 * score2
		# try:
		# 	print("Score: {:.4f}, Score1: {:.4f}, Score2: {:.4f}, Final: {:.4f}".format(score, score1, score2, final_score))
		# except:
		# 	exit()
		# print(score, score1, score2, final_score)
		# exit()

		print(self.margin_flag)

		if self.margin_flag:
			return self.margin - final_score
		else:
			return final_score


	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()
