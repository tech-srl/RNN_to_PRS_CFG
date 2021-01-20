from make_and_train_rnn import *
from find_rnn import *
from all_cfgs import cfgs
from TACAS_cfgs import TACAS_cfgs
import argparse
import ast
from Helper_Functions import prepare_directory
from Extraction import extract
from Helper_Functions import overwrite_file
from highlight_diffs import highlight_all_diffs


TACASpref = "TACAS."
allTACASs = TACASpref + "all"
language_options = list(cfgs.keys())+[TACASpref+l for l in TACAS_cfgs.keys()]+ [allTACASs]

def get_args():
	parser = argparse.ArgumentParser(description='Train/Load an RNN and extract from it')

	# language
	parser.add_argument('--lang',type=str,help="language to load/train",choices=language_options)
	parser.add_argument('--lang-params',type=ast.literal_eval,default=tuple())

	# rnn params
	parser.add_argument('--hidden-dim',type=int, default=10)
	parser.add_argument('--input-dim',type=int, default=4)
	parser.add_argument('--num-layers', type=int, default=2)
	parser.add_argument('--RNNClass', type=str, default='LSTM',choices=["LSTM","GRU"])

	# training params
	parser.add_argument('--train-set-size',type=lambda x:int(float(x)),default=10000)
	parser.add_argument('--validation-set-size',type=lambda x:int(float(x)),default=1000)
	parser.add_argument('--iterations-per-learning-rate',type=int,default=5)
	parser.add_argument('--learning-rates',type=ast.literal_eval,default=(0.01,0.004,0.001,0.0003))
	parser.add_argument('--batch-size',type=int,default=50)
	parser.add_argument('--check-validation-improvement-every',type=int,default=50)
	parser.add_argument('--wee-sample-rate',type=int,default=10)
	parser.add_argument('--wee-sample-cutoff',type=int,default=50)

	# reloading older rnn params
	parser.add_argument('--make-new',action='store_true',help="train a new rnn for this language, even if one with the given parameters already exists.\
		 (if not set, will take existing rnn if there is one)")
	parser.add_argument('--find-any-rnn-for-lang',action='store_true',help="ignore rnn parameters such as hidden dim, input dim, num layers, class -\
	 just return the last saved rnn for the requested language")

	# extraction params
	parser.add_argument('--transition-reject-threshold',type=float,default=0.02)
	parser.add_argument('--extraction-time-limit',type=int,default=50,help="extraction time limit in seconds")
	parser.add_argument('--initial-split-depth',type=int,default=10,help="number of splits to make for aggressive initial partitioning (after first counterexample)")
	parser.add_argument('--starting-samples',type=ast.literal_eval,default=None,help="ideally 2 samples, 1 positive and 1 negative, to start the extraction")
	parser.add_argument('--token-predictor-samples',type=int,default=-1,help="number of counterexample attempts to make by just sampling the rnn")
	parser.add_argument('--token-predictor-cutoff',type=int,default=-1,help="length of randomly sampled counterexample attempts")

	# location params
	parser.add_argument('--subfolder',type=str,default=".")

	args = parser.parse_args()
	return args

def check_TACASlang(args):
	if args.lang.startswith(TACASpref):
		TACASlang = args.lang[len(TACASpref):]
		lang_and_params = TACAS_cfgs[TACASlang]
		args.lang = lang_and_params[0]
		args.lang_params = lang_and_params[1]
		args.langpref = TACASlang + "_"
	else:
		args.langpref = ""
	return args

def get_cfg_rnn_and_folder(args):
	rnn_name = args.langpref + args.lang + "_" + str(args.lang_params)
	cfg = cfgs[args.lang](*args.lang_params)

	if not args.make_new:
		if args.find_any_rnn_for_lang:
			rnn,rnn_folder = find_rnn(rnn_name,get_location_too=True,subfolder=args.subfolder)
		else:
			rnn,rnn_folder = find_rnn(rnn_name,num_layers=args.num_layers,hidden_dim=args.hidden_dim,\
				input_dim=args.input_dim,RNNClass=args.RNNClass,get_location_too=True,subfolder=args.subfolder)
	else:
		rnn = None

	if None is rnn:
		rnn,rnn_folder = make_rnn_for_cfg(cfg,rnn_name,\
			train_size=args.train_set_size,validation_size=args.validation_set_size,\
			num_layers=args.num_layers,hidden_dim=args.hidden_dim,input_dim=args.input_dim,\
			RNNClass=args.RNNClass,iterations_per_learning_rate=args.iterations_per_learning_rate,\
			learning_rates=args.learning_rates,batch_size=args.batch_size,\
			check_improvement_every=args.check_validation_improvement_every,\
			wee_sample_rate=args.wee_sample_rate,sample_cutoff=args.wee_sample_cutoff,subfolder=args.subfolder)
		print("made new rnn ",end="")
		with open(rnn_folder+"/calling_args.txt","w") as f:
			d = vars(args)
			print("\n".join([(k+" : "+ str(d[k])) for k in d]),file=f)
	else:
		print("found rnn ",end="")
	print("with:",rnn.num_layers,"layers, hidden dim:",rnn.hidden_dim,", input dim:",rnn.input_dim)
	print("rnn informal name (language name):",rnn.informal_name)
	return cfg, rnn, rnn_folder


def run_and_save_extraction(args,rnn,rnn_folder,emptyseq):
	rnn.prep_for_extraction(args.transition_reject_threshold)
	print("extracting using transition reject threshold:",rnn.transition_reject_threshold)
	extraction_folder = rnn_folder + "/extraction_"+str(rnn.transition_reject_threshold)+"_"+str(args.initial_split_depth)+\
		"_"+str(args.extraction_time_limit)+"_"+str(args.starting_samples)
	output_dfa_imgs_folder =  extraction_folder+"/dfa_pngs"
	prints_filename = extraction_folder + "/prints.txt"
	prepare_directory(prints_filename)
	with open(prints_filename,"w") as prints_file:
		dfa, all_dfas = extract(rnn,output_dfa_imgs_folder,prints_file,time_limit = args.extraction_time_limit,\
			initial_split_depth = args.initial_split_depth,starting_examples=args.starting_samples,\
			token_predictor_samples=args.token_predictor_samples,token_predictor_cutoff=args.token_predictor_cutoff,
			emptyseq=emptyseq)
	overwrite_file(all_dfas,extraction_folder+"/all_dfas")
	highlight_all_diffs(all_dfas,extraction_folder+"/diff_pngs")


def one_lang_run(args):
	args = check_TACASlang(args)
	cfg, rnn, rnn_folder = get_cfg_rnn_and_folder(args)
	run_and_save_extraction(args,rnn,rnn_folder,"" if cfg.sample_as_strings else tuple())

def main_run():
	args = get_args()
	if args.lang == allTACASs:
		for l in TACAS_cfgs:
			args.lang = TACASpref+l
			one_lang_run(args)
	else:
		one_lang_run(args)

if __name__ == "__main__":
	main_run()