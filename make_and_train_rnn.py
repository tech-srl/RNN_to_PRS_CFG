from Helper_Functions import prepare_directory
from ContextFreeGrammar import get_n_samples
from RNNTokenPredictor import RNNTokenPredictor, train_rnn, save_rnn
import math
from time import process_time

def make_rnn_for_cfg(cfg,informal_name,train_size=10000,validation_size=1000,\
	num_layers=2,hidden_dim=10,input_dim=4,RNNClass="LSTM",\
	iterations_per_learning_rate=5,learning_rates=(0.01,0.004,0.001,0.0003),batch_size=50,\
	check_improvement_every=50,wee_sample_rate=10,sample_cutoff=50,subfolder="."):
	train_set = get_n_samples(cfg,train_size)
	validation_set = get_n_samples(cfg,validation_size)

	rnn = RNNTokenPredictor(cfg.terminals(),input_dim,hidden_dim,num_layers,\
		RNNClass,informal_name=informal_name,dropout=0.5)
	rnn_folder = "rnns/"+subfolder+"/"+informal_name+"/"+rnn.name # rnn.name is a timestamp..
	training_prints_filename = rnn_folder+"/training_prints.txt"
	prepare_directory(training_prints_filename,includes_filename=True)
	with open(training_prints_filename,"a") as f:
		print("training rnn with train set of size:",len(train_set),", average length:",\
			sum([len(s) for s in train_set])/len(train_set),file=f)
		print("validation set size:",len(validation_set),", average length:",\
			sum([len(s) for s in validation_set])/len(validation_set),file=f)
		print("first 10 samples in train set:\n","\n".join([str(s) for s in train_set[:10]]),file=f)
		str_validation = [str(s) for s in validation_set]
		str_train = [str(s) for s in train_set]
		validation_not_in_train = set(str_validation) - set(str_train)
		intersection = set(str_validation) - validation_not_in_train
		t_notin_v = len([True for s in str_train if not s in intersection])
		v_notin_t = len([True for s in str_validation if not s in intersection])
		print("num sequences in train set not in validation:",t_notin_v,"(",int(100*t_notin_v/len(train_set)),"% )",file=f)
		print("num sequences in validation set not in train:",v_notin_t,"(",int(100*v_notin_t/len(validation_set)),"% )",file=f)
		print("rnn has:",num_layers,"layers, hidden dim:",hidden_dim,"input dim:",input_dim)
		print("using batch size:",batch_size)
	start = process_time()
	rnn = train_rnn(rnn,train_set,validation_set,rnn_folder,
		iterations_per_learning_rate=iterations_per_learning_rate,learning_rates=learning_rates,
		batch_size=batch_size,check_improvement_every=check_improvement_every,
		step_size_for_prefs=100,step_size_for_progress_checks=200,
		progress_seqs_at_a_time=1000,wee_sample_rate=wee_sample_rate,
		print_sample_joiner="" if cfg.sample_as_strings else " ",
		sample_cutoff=sample_cutoff) 
	print("train rnn returned successfully",flush=True)
	print("this is the rnn:",rnn,flush=True)
	print("its name is:",rnn.informal_name,flush=True)
	with open(training_prints_filename,"a") as f:
		print("total time training:",process_time()-start,file=f)
	return rnn, rnn_folder
	# return 0




	



