from RNNTokenPredictor import load_rnn
from LanguageModel import LanguageModel
from Helper_Functions import things_in_path, clean_val
import argparse
import math

def find_rnn(lang_name,num_layers=-1,hidden_dim=-1,input_dim=-1,RNNClass=-1,\
	just_get_last=True,testing=False,in_training_idc=False,as_lm=False,get_location_too=False,subfolder="."):
	if testing: # not sure this ever happens in this project but oh well
		main_rnns_path = "test_rnns/"
	else:
		main_rnns_path = "rnns/"+subfolder+"/"

	potential_rnns = things_in_path(main_rnns_path+lang_name)

	res = []
	locs = []
	for r_name in potential_rnns:
		location = main_rnns_path+lang_name +"/"+r_name
		rnn = load_rnn(location,quiet=in_training_idc)
		if (None is rnn) and in_training_idc:
			location += "/training_savepoints"
			rnn = load_rnn(location)
		if (not None is rnn) and\
		   (num_layers in [-1,rnn.num_layers]) and\
		   (hidden_dim in [-1,rnn.hidden_dim]) and\
		   (input_dim in [-1,rnn.input_dim]) and\
		   (RNNClass in [-1,rnn.RNNClass]):
			res.append(rnn)
			locs.append(location)
	if as_lm:
		res = [LanguageModel(r) for r in res]
			
	if just_get_last:
		res = res[-1] if res else None
		locs = locs[-1] if res else None
	return (res,locs) if get_location_too else res

def sample_training_rnn(lang_name,testing,cutoff=100,from_seq=None):
	print("=======sampling==========")
	a = find_rnn(lang_name,testing=testing,in_training_idc=True,as_lm=True)
	tloss = a.model.training_losses[-1]
	vloss = a.model.validation_losses[-1]									
	print("current train loss:",clean_val(tloss,4),", validation loss:",clean_val(vloss,4),".\
	 (e^losses: [",clean_val(pow(math.e,tloss),4),"], [",clean_val(pow(math.e,vloss),4),"]\n\n")
	print(''.join(a.sample(cutoff=cutoff,from_seq=from_seq)))
	return a