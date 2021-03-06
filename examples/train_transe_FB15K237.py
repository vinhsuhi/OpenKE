import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import argparse

parser = argparse.ArgumentParser(description="transE")
parser.add_argument('--weight1', type=float, default=.0)
parser.add_argument('--weight2', type=float, default=.0)
parser.add_argument('--epochs', type=int, default=1000)
# parser.add_argument('--model_name', type=str, default="FB15K23")
args = parser.parse_args()

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB15K237/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 128, 
	p_norm = 2, 
	norm_flag = True, weight1=args.weight1, weight2=args.weight2)


# define the loss function
model = NegativeSampling( 
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = args.epochs, alpha = 1.0, use_gpu = True)
trainer.run()
transe.save_checkpoint('./checkpoint/transe{}_{}.ckpt'.format(args.weight1, args.weight2))

# test the model
transe.load_checkpoint('./checkpoint/transe{}_{}.ckpt'.format(args.weight1, args.weight2))
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
