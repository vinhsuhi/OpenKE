import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import argparse

parser = argparse.ArgumentParser(description="transE")
parser.add_argument('--new', action='store_true')
parser.add_argument('--epochs', type=int, default=1000)
args = parser.parse_args()

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/WN18RR/", 
	batch_size = 2000,
	threads = 32,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 512, 
	p_norm = 1,
	norm_flag = False,
	margin = 6.0, new=args.new)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = SigmoidLoss(adv_temperature = 1),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 2e-5, use_gpu = True, opt_method = "adam")
trainer.run()
# transe.save_checkpoint('./checkpoint/transe_2.ckpt')

# test the model
# transe.load_checkpoint('./checkpoint/transe_2.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
