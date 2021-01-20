from time import process_time, time
import os
import shutil
import sys
from Helper_Functions import chronological_scatter, class_to_dict, load_from_file, overwrite_file, prepare_directory, do_timed, clean_val
from random import shuffle
import torch.nn as nn
import torch
import numpy as np
from copy import copy
import math
from LanguageModel import LanguageModel
import random
import itertools
import string

# TODO: only check validation every X batches else it'll get crazy



# for tensors, if trying to retrieve the actual value:
# t.data.tolist()     # if t is not an 'array' but a scalar (wrapped in a tensor..) then this will also return a scalar. otherwise, nested lists

class RNNModule(nn.Module):
	def __init__(self,input_dim,hidden_dim,num_layers,RNNClass,num_output_tokens,num_input_tokens,dropout):
		super(RNNModule,self).__init__()
		assert RNNClass in ["LSTM","GRU"], "RNNClass must be LSTM or GRU"
		self.initial_h = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim), requires_grad=True)	# 1 - for batch size 1
		if RNNClass == "LSTM":
			self.rnn = nn.LSTM(input_dim,hidden_dim,num_layers=num_layers,dropout=dropout)
			self.initial_c = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim), requires_grad=True)
			self.initial = (self.initial_h,self.initial_c)

		elif RNNClass == "GRU":
			self.rnn = nn.GRU(input_dim,hidden_dim,num_layers=num_layers,dropout=dropout)
			self.initial = self.initial_h

		self.embedding = nn.Embedding(num_input_tokens, input_dim)
		self.hidden2tag = nn.Linear(hidden_dim, num_output_tokens)

		self.RNNClass = RNNClass

	def states_equal(self,s1,s2):
		if self.RNNClass == "LSTM":
			return torch.equal(s1[0],s2[0]) and torch.equal(s1[1],s2[1])
		elif self.RNNClass == "GRU":
			return torch.equal(s1,s2)

	def tensor(self,val,kwargs=None):
		kwargs={} if None is kwargs else kwargs
		res = torch.tensor(val,**kwargs)
		if self.using_cuda:
			res = res.cuda()
		return res

	def contiguous_state(self,state):
		if self.RNNClass == "LSTM":
			return (state[0].contiguous(),state[1].contiguous())
		elif self.RNNClass == "GRU":
			return state.contiguous()

	def _expand_state_for_batches(self,state,num_batches): # state vecs should be num_layers X 1 X hidden_dim
		if self.RNNClass == "LSTM":
			res = (state[0].expand(-1,num_batches,-1), state[1].expand(-1,num_batches,-1)) # (h,c)
		elif self.RNNClass == "GRU":
			res = state.expand(-1,num_batches,-1)
		return self.contiguous_state(res)

	def forward(self,indices_seqs,state=None): #
		pass # make this one a full processing one that gives the probabilities of each token for each sequence (including of EOS)

	def detached_state(self,state):
		if self.RNNClass == "LSTM":
			res = (state[0].detach(),state[1].detach())
		elif self.RNNClass == "GRU":
			res = state.detach()
		return self.contiguous_state(res)

	def state_as_vec(self,state):
		def tensor_to_vec(t):
			return t.view(-1).tolist()
		if self.RNNClass == "LSTM":
			return tensor_to_vec(state[0]) + tensor_to_vec(state[1])
		elif self.RNNClass == "GRU":
			return tensor_to_vec(state)

	def state_from_vec(self,vec):
		def vec_to_tensor(v):
			return self.tensor(v).view(self.initial_h.size())
		if self.RNNClass == "LSTM":
			real_len = int(len(vec)/2)
			return (vec_to_tensor(vec[:real_len]), vec_to_tensor(vec[real_len:]))
		elif self.RNNClass == "GRU":
			return vec_to_tensor(vec)

	def state_batch_size(self,state):
		if None is state:
			return 0
		if self.RNNClass == "LSTM":
			return state[0].size(1)
		elif self.RNNClass == "GRU":
			return state.size(1)

	def split_state(self,state,first_k):
		if self.RNNClass == "LSTM":
			res = (state[0][:,:first_k,:],state[1][:,:first_k,:])
			res2 = (state[0][:,first_k:,:],state[1][:,first_k:,:])
		elif self.RNNClass == "GRU":
			res = state[:,:first_k,:]
			res2 = state[:,first_k:,:]
		return self.contiguous_state(res), self.contiguous_state(res2)


	def concat_states(self,states_tuple): # each state should be num_layers X batch_size X hidden_dim
		if self.RNNClass == "LSTM":
			res = (torch.cat(tuple(s[0] for s in states_tuple),dim=1), torch.cat(tuple(s[1] for s in states_tuple),dim=1))
		elif self.RNNClass == "GRU":
			res = torch.cat(states_tuple,dim=1)
		return self.contiguous_state(res)

	def permute_state(self,state,new_order_current_batch_indices):
		def reorder(vec):
			return vec[:,new_order_current_batch_indices,:]
		if self.RNNClass == "LSTM":
			res = (reorder(state[0]),reorder(state[1]))
		elif self.RNNClass == "GRU":
			res = reorder(state)
		return self.contiguous_state(res)

	def _check_seqs(self,indices_seqs,func_name,empty_seqs_ok=False):
		def issorted(a):
			return sorted(a,reverse=True)==a
		assert indices_seqs, func_name+" expects at least one seq"

		lengths = [len(s) for s in indices_seqs]
		assert (issorted(lengths)), func_name+" expect seqs sorted by descending order"
		if not empty_seqs_ok:
			assert lengths[-1]>0, func_name+" expects non-empty seqs, got:" + str(indices_seqs)
		return lengths

	def _prep_state_for_batch(self,state,batch_size):
		if None is state:
			state = self.initial
		if self.state_batch_size(state)==1: # i.e. received exactly one state, which all the seqs must go from
			state = self._expand_state_for_batches(state,batch_size)
		return state

	# def split_by_size(self,indices_seqs,lengths,states,step_size): # states are layers X batch X hidden_dim
	# 	under_stepsize_start = next((i for i,l in enumerate(lengths) if l<step_size),len(lengths))
	# 	zeros_start = next((i for i,l in enumerate(lengths) if l==0),len(lengths))

	# 	over_seqs = indices_seqs[:under_stepsize_start]
	# 	over_lengths = lengths[:under_stepsize_start]
	# 	over_states, under_and_zero_states = self.split_state(states,under_stepsize_start)
	# 	under_and_nonzero_seqs = indices_seqs[under_stepsize_start:zeros_start]
	# 	under_nonzero_states, zero_seq_states = self.split_state(under_and_zero_states,zeros_start- under_stepsize_start)
	# 	return over_seqs, over_lengths, over_states, under_and_nonzero_seqs, under_nonzero_states, zero_seq_states

	def split_sorted_by_size(self,step_size,lengths,forward_seqs,states,eval_seqs=None): # states are layers X batch X hidden_dim
		# assert sorted(lengths,reverse=True) == lengths
		def split_list(seqs):
			return seqs[:under_stepsize_start], seqs[under_stepsize_start:zeros_start], seqs[zeros_start:]
		def split_states(states):
			r1,r23 = self.split_state(states,under_stepsize_start)
			r2,r3 = self.split_state(r23,zeros_start- under_stepsize_start)
			return r1,r2,r3
		under_stepsize_start = next((i for i,l in enumerate(lengths) if l<step_size),len(lengths))
		zeros_start = next((i for i,l in enumerate(lengths) if l==0),len(lengths))

		eval_seqs = eval_seqs if not None is eval_seqs else []
		return split_list(lengths) ,split_list(forward_seqs), split_list(eval_seqs), split_states(states)


	def detached_negative_total_logs_on_seqs(self,indices_seqs,end_token_idx,step_size=100,already_sorted=False): 
		def first_logs(seq):
			return first_log_dist_for_all[seq[0]] if seq else first_log_dist_for_all[end_token_idx]
		def total_logs(eval_seqs,outputs): # outputs are batch X len X tokens
			output_after_linear = self.hidden2tag(outputs) 
			all_dists = nn.LogSoftmax(dim=2)(output_after_linear) # dim=1 - softmaxes across second dimension, i.e. across the output token weights for each batch.

			def seq_logs(seq,seq_dists):
				return [d[t] for t,d in zip(seq,seq_dists)]
			res = sum([sum(seq_logs(s,d)) for s,d in zip(eval_seqs,all_dists)]).item()
			# print("adding total logs:",res)
			return res
		if not indices_seqs: # none given
			return 0 # this is detached, its just an eval func anyway. its fine if called with empty
		if not already_sorted: # really trusting the customer here
			indices_seqs = sorted(indices_seqs,key=len,reverse=True)
		lengths = self._check_seqs(indices_seqs,"detached_negative_total_logs_on_sorted_seqs",empty_seqs_ok=True) # lol not really but still mildly less expensive to have an idea
		# first token loss for each of the seqs, from the initial state. (eos for empty seqs)
		first_log_dist_for_all = self.state_to_distribution(self.initial,actually_get_logs=True)[0] # (start all from initial state. function gives batch size X num_tokens, lose batch dim by going to 0)
		res = -sum([first_logs(s) for s in indices_seqs]).item()
		# empty seqs have been completely handled, remove any still here:
		num_zero_seqs = lengths.count(0)
		if num_zero_seqs > 0: # if its just 0, a[:-0] wont give a but rather []
			lengths = lengths[:-num_zero_seqs]# remember all reverse ordered by length at this point
			indices_seqs = indices_seqs[:-num_zero_seqs]


		# print("first res is:",res)
		# make shifted eval_seqs, with eos tacked on end
		# (remember: states_from_sorted_nonzero_seqs only gives the outputs AFTER the tokens, not the ones from before them)
		eval_indices_seqs = [s[1:]+[end_token_idx] for s in indices_seqs]
		states = self._prep_state_for_batch(None,len(indices_seqs)) # start from initial state

		while True:
			split_lengths, split_seqs, split_evals, split_states = self.split_sorted_by_size(\
				step_size,lengths,indices_seqs,states,eval_seqs=eval_indices_seqs) 
			# nothing to do with the zero-length seqs: they'll have zero evals left too 
			# (evals have been shifted and matched with input that creates their output)
			under_and_nonzero_seqs = split_seqs[1]
			if under_and_nonzero_seqs: # shorter. also, finished - need eos too
				new_under_outputs, _, _ = self.states_from_sorted_nonzero_seqs(under_and_nonzero_seqs,state=split_states[1],give_outputs_batch_first=True)
				# print("calling from under_and_nonzero_seqs",flush=True)
				# print("split seqs:",under_and_nonzero_seqs)
				# print("their evals:",split_evals[1])
				res -= total_logs(split_evals[1],new_under_outputs)
			# remaining
			over_seqs = split_seqs[0]
			over_evals = split_evals[0]
			if over_seqs:
				over_step = [s[:step_size] for s in over_seqs]
				new_over_outputs, new_over_states, _ = self.states_from_sorted_nonzero_seqs(over_step,state=split_states[0],give_outputs_batch_first=True)
				# print("calling from over_seqs",flush=True)
				over_evals_step = [s[:step_size] for s in over_evals]
				res -= total_logs(over_evals_step,new_over_outputs)
				# prep for next round
				states = self.detached_state(new_over_states) 
				indices_seqs = [s[step_size:] for s in over_seqs]
				eval_indices_seqs = [s[step_size:] for s in over_evals]
				lengths = [(l-step_size) for l in split_lengths[0]] # if one of them happens to be finished, it'll be sorted out by the zero_seq_states in the next round
			else:
				return res
			
		


	def detached_last_state_from_sorted_seqs(self,indices_seqs,step_size,state=None):
		def append_to_start(res,new_states):
			return new_states if None is res else  self.concat_states((new_states,res))
		lengths = self._check_seqs(indices_seqs,"detached_last_state_from_sorted_seqs",empty_seqs_ok=True)
		res = None
		states = self._prep_state_for_batch(state,len(indices_seqs))
		while True:
			split_lengths, split_seqs, _, split_states = self.split_sorted_by_size(step_size,lengths,indices_seqs,states) 
			# all split in order of over/between/zero
			zero_seq_states = split_states[2]
			if self.state_batch_size(zero_seq_states)>0: # shortest
				res = append_to_start(res,zero_seq_states)

			under_and_nonzero_seqs = split_seqs[1]
			if under_and_nonzero_seqs: # shorter
				_, new_under_states, _ = self.states_from_sorted_nonzero_seqs(under_and_nonzero_seqs,state=split_states[1])
				res = append_to_start(res,new_under_states)
			# remaining
			over_seqs = split_seqs[0]
			if over_seqs:
				over_step = [o[:step_size] for o in over_seqs]
				_, new_over_states, _ = self.states_from_sorted_nonzero_seqs(over_step,state=split_states[0])
				# prep for next round
				states = self.detached_state(new_over_states) 
				indices_seqs = [o[step_size:] for o in over_seqs]
				lengths = [(l-step_size) for l in split_lengths[0]]
			else:
				return res

	def states_from_sorted_nonzero_seqs(self,indices_seqs,state=None,give_outputs_batch_first=False): 
		# recursive application of the state transition function on NON-EMPTY inputs because pytorch is kinda bullshit, possibly on several sequences
		lengths = self._check_seqs(indices_seqs,"states_from_sorted_nonzero_seqs")
		max_len = max(lengths)
		
		padded_seqs = [s+([0]*(max_len-l)) for s,l in zip(indices_seqs,lengths)]
		embedded_padded_seqs = self._indices_seqs_to_embeddings_seqs(padded_seqs) #tensor - max_len X len(indices_seqs) (ie batch size) X input_dim
		packed_padded_seqs = nn.utils.rnn.pack_padded_sequence(embedded_padded_seqs, self.tensor(lengths), batch_first=False, enforce_sorted=True)
		
		state = self._prep_state_for_batch(state,len(indices_seqs))
		assert not None is state # pytorch is totally chill with this happening and will just use zeros b/c fuck logic i guess, so always always check this before calling rnn even if it shouldn't be None cause there might be some dumb bug
		packed_outputs, new_states = self.rnn(packed_padded_seqs,state)
		outputs, lengths = nn.utils.rnn.pad_packed_sequence(packed_outputs,batch_first=give_outputs_batch_first) 
		# new_states (i.e. hidden): (h if GRU, (h,c) if LSTM) values for each layer after each sequence. each vector in hidden is num_layers X batch_size X hidden_size
		# output: hidden values for top layer (h-)value for each token in each sequence. sequences_length X batch_size X hidden size, (or batch_size X sequences_length X hidden size if using batch_first=True)
		return outputs, new_states, lengths

	def _all_distributions_for_sorted_seqs(self,indices_seqs,state,actually_get_logs): 
		# returns list of lists of distributions: num sequences X each sequence's length X num output tokens
		# expects all sorted in descending length order for pytorch reasons (variable-length batching must be in descending length order)
		lengths = [len(s) for s in indices_seqs]
		assert sorted(lengths,reverse=True) == lengths, "all_distributions_for_seqs expects sequences sorted by descending order of length"
		## need to add initial distribution on initial state as well
		if not self.state_batch_size(state) == len(indices_seqs): # got a single state
			state = self._expand_state_for_batches(state,len(indices_seqs)) # expand so that get distribution for each one in first_dist_for_each 
			# (as opposed to one distribution, and then only the first* sequence gets the initial dist) (*this function is only called with sorted seqs,
			# so if called from somewhere else the initial distribution ends up being attached only to the longest sequence)
		first_dist_for_each = self.state_to_distribution(state,actually_get_logs=actually_get_logs) # tensor, size: batch_size X num_output_tokens

		non_zero_num = lengths.index(0) if lengths[-1]==0 else len(lengths) # index(item) returns the index of the first occurence of item, but raises a valueerror if it fails instead of something neater...
		res = []
		if non_zero_num>0:
			non_zero_outputs, _, non_zero_lengths = self.states_from_sorted_nonzero_seqs(indices_seqs[:non_zero_num],state=self.split_state(state,non_zero_num)[0],give_outputs_batch_first=True)
			outputs_after_linear = self.hidden2tag(non_zero_outputs) # tensor, size: batch size (len(indices_seqs)) X max_seq_len  X num_output_tokens
			s = nn.LogSoftmax(dim=2) if actually_get_logs else nn.Softmax(dim=2) 
			other_dists = s(outputs_after_linear)
			res += [torch.cat((first_dist_for_each[i:i+1],other_dists[i][:l])) for i,l in enumerate(non_zero_lengths)]
		if lengths[-1]==0:
			# print(first_dist_for_all,first_dist_for_all.size())
			empty = self.tensor([]).view(0,self.hidden2tag.out_features)
			# print(empty,empty.size(),flush=True)
			res += [torch.cat((first_dist_for_each[i:i+1],empty)) for i in range(non_zero_num,len(indices_seqs))]
		# return list of lists of distributions: num sequences X [[ each sequence's length+1 X num output tokens]]. just the bit in [[]] is a tensor
		return res			   


	def all_distributions_for_seqs(self,indices_seqs,state=None,actually_get_logs=False): 
		# returns list of lists of distributions: num sequences X each sequence's length X num output tokens
		if None is state:
			state = self.initial
		sorted_lengths_and_indices = sorted([(len(s),i) for i,s in enumerate(indices_seqs)],key=lambda x:x[0],reverse=True)
		new_order_current_batch_indices = [s[1] for s in sorted_lengths_and_indices]
		sorted_seqs = [indices_seqs[i] for i in new_order_current_batch_indices]
		original_index_to_sorted_index = {j:i for i,j in enumerate(new_order_current_batch_indices)}

		if self.state_batch_size(state) == len(indices_seqs): # got state for each seq, so need to permute it appropriately having reordered the seqs
			state = self.permute_state(state,new_order_current_batch_indices)

		sorted_all_dists = self._all_distributions_for_sorted_seqs(sorted_seqs,state,actually_get_logs) # this internal function returns list of lists of distributions: num sequences X each sequence's length X num output tokens
		
		# reorder results:
		original_order_all_dists = [sorted_all_dists[original_index_to_sorted_index[i]] for i in range(len(indices_seqs))]
		return original_order_all_dists # list of lists of distributions. each list of distributions is a tensor: num sequences X each sequence's length X num output tokens. in original sequence order

	def all_distributions_for_seq(self,indices_seq,state=None,actually_get_logs=False): 
	# returns len(indices_seq) + 1 distributions: the one from the start and then the one after every input
	# entire return value is a tensor, size: len(indices_seq)+1 X num_output_tokens
		return self.all_distributions_for_seqs([indices_seq],state=state,actually_get_logs=actually_get_logs)[0] # just one sequence, return that one


		# if None is state:
		# 	state = self.initial
		# h = state if self.RNNClass == "GRU" else state[0]
		# assert h.size()[1]==1, "in all_distributions_for_seq (expects one seq), got state with shape:"+str(h.size()) # assert got state for batch size 1
		# res = [self.state_to_distribution(state,actually_get_logs=actually_get_logs)] #  list with 1 element of size: 1 (batch size) X num_output_tokens
		# if indices_seq: # if not empty sequence, process its tokens
		# 	outputs,_,_ = self.states_from_sorted_nonzero_seqs([indices_seq],state)
		# 	# outputs dims: len(indices_seq) X batch_size (1) X hidden size
		# 	outputs_after_linear = self.hidden2tag(outputs) # now len(indices_seq) X batch_size(1) X num_output_tokens
		# 	s = nn.LogSoftmax(dim=2) if actually_get_logs else nn.Softmax(dim=2) 
		# 	other_dists = s(outputs_after_linear) # len(indices_seq) X 1 X num_output_tokens
			
		# 	res += list(other_dists)
		# return [dist.view(-1) for dist in res] # none of them need that extra batch dimension - flatten them out to 1-dim lists
		# # list of length len(indices_seq)+1, of tensors with 1 dim, with size: num_output_tokens

	def state_to_distribution(self,state,actually_get_logs=False): # also works on batches
		if self.RNNClass == "LSTM":
			state = state[0] # state = (h,c) ----> state = h
		output = state[-1] # last layer. gives vec with dim:  batch size X hidden size
		output_after_linear = self.hidden2tag(output) # batch_size X num_output_tokens
		s = nn.LogSoftmax(dim=1) if actually_get_logs else nn.Softmax(dim=1)
		return s(output_after_linear) # dim=1 - softmaxes across second dimension, i.e. across the output token weights for each batch.
		# returns batch_size X num_output_tokens

	def _indices_seqs_to_embeddings_seqs(self,indices_seqs):
		assert indices_seqs, "_indices_seqs_to_embeddings_seqs expects at least one sequence"
		assert len(set([len(seq) for seq in indices_seqs]))==1, "_indices_seqs_to_embeddings_seqs expects sequences of same length"
		batched_seqs = [list(tokens) for tokens in zip(*indices_seqs)] # eg for this bit to work (zip will all iterables to the length of the shortest among them)
		embedding_input = self.tensor(batched_seqs, {'dtype':torch.long})
		return self.embedding(embedding_input).view(len(indices_seqs[0]), len(indices_seqs), self.rnn.input_size)
		# sequence length (len(idxs) X batch size X input_dim 

	def state_to_lists(self,state):
		if self.RNNClass == "LSTM":
			return [state[0].data.tolist(),state[1].data.tolist()]
		if self.RNNClass == "GRU":
			return state.data.tolist()



class RNNTokenPredictor:
	def __init__(self,alphabet,input_dim,hidden_dim,num_layers,RNNClass,informal_name=None,end_token="<EOS>",dropout=0.5,source_dict=None):
		if not None is source_dict:
			self._build_from_dict(source_dict)


			return
		self.name = str(time()) # may get changed if being loaded
		self.informal_name = informal_name if not None is informal_name else RNNClass
		self.input_alphabet = alphabet # dont make this a list just because. the extraction notebooks later use the input alphabet and they'll make the wrong kind of wfas if you do this
		self.end_token = end_token
		assert not self.end_token in self.input_alphabet, "please give end token not in input alphabet"
		self.internal_alphabet = list(self.input_alphabet) + [self.end_token]#+[self.begin_token]
		self.int2char = self.internal_alphabet # its a list from def of internal_alphabet
		self.char2int = {c:i for i,c in enumerate(self.int2char)}
		self._prep_stats()
		
		# make the pytorch part
		self.rnn_module = RNNModule(input_dim,hidden_dim,num_layers,RNNClass,\
			len(self.internal_alphabet),len(self.input_alphabet),dropout)

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.RNNClass = RNNClass
		self.dropout = dropout
		self.eval()
		self.cpu(just_initiating=True) # start safe


	def prep_for_extraction(self,transition_reject_threshold):
		initial_vec = self.rnn_module.state_as_vec(self.initial_state())
		self.impossible_state = self.rnn_module.state_from_vec( [2.0]*len(initial_vec)) # h is always in -1,1, both GRUs and LSTMs have h
		self.transition_reject_threshold = transition_reject_threshold # all transitions are theoretically fine, but can explicitly define greater threshold for DFA extraction

	def neurons_vals_on_seq(self,seq,layers,indices,vectors):
		def get_neuron(s,layer,index,vector):
			assert vector in ["h","c"]
			if self.RNNClass == "LSTM": s = s[0] if vector == "h" else s[1] # h or c
			return s[layer,0,index].item() # layer X batch X hidden
		s = self.initial_state()
		res = {(l,i,v):[] for l,i,v in itertools.product(layers,indices,vectors)}
		for t in seq:
			s = self.next_state(s,t)
			for l,i,v in itertools.product(layers,indices,vectors):
				res[(l,i,v)].append(get_neuron(s,l,i,v))
		return res

	def cpu(self,just_initiating=False,just_saving=False):
		# print("going into CPU "+("from initiation" if just_initiating else "")+("(for saving)" if just_saving else ""),flush=True)
		self.rnn_module = self.rnn_module.cpu() # honestly no idea if this function changes the model in place and docs don't give a pissing clue
		self.rnn_module.using_cuda = False

	def cuda(self,returning_from_save=False):
		# print("going into cuda"+(" (returning from save)" if returning_from_save else ""),flush=True)
		self.rnn_module = self.rnn_module.cuda()
		self.rnn_module.using_cuda = True

	def train(self):
		self.rnn_module.train()

	def eval(self):
		self.rnn_module.eval()

	def training(self):
		return self.rnn_module.training

	def _to_dict(self):
		return class_to_dict(self,ignoring_list=["rnn_module"])

	def _build_from_dict(self,source_dict):
		[setattr(self,p,source_dict[p]) for p in source_dict]
				
	def _prep_stats(self):
		self.training_losses = []
		self.validation_losses = []
		self.max_param_values = []
		self.avg_gradients = []
		self.max_gradients = []
		self.total_train_time = 0
						
	def initial_state(self):
		return self.rnn_module.initial 

	def _classify_state(self,state):
		if self.rnn_module.states_equal(state,self.impossible_state):
			return False
		return self.state_char_prob(state,self.end_token) >= self.transition_reject_threshold

	def classify_word(self,word):
		s = self.initial_state()
		for c in word:
			if self.state_char_prob(s,c) < self.transition_reject_threshold:
				return False
			s = self.next_state(s,c)
		return self._classify_state(s)

	def get_first_RState(self):
		s = self.initial_state()
		return self.rnn_module.state_as_vec(s), self._classify_state(s)

	def get_next_RState(self,state,char):
		tensor_state = self.rnn_module.state_from_vec(state)

		if self.rnn_module.states_equal(tensor_state,self.impossible_state) or \
			(self.state_char_prob(tensor_state,char)<self.transition_reject_threshold):
			tensor_state = self.impossible_state
		else:
			tensor_state = self.next_state(tensor_state,char)
		return self.rnn_module.state_as_vec(tensor_state), self._classify_state(tensor_state)

	def state_from_sequence(self,seq,state=None):
		state = state if not None is state else self.initial_state()
		if not seq: # length 0:
			return state
		idxs = [self.char2int[w] for w in seq]
		_, new_states,_ = self.rnn_module.states_from_sorted_nonzero_seqs([idxs],state)
		return new_states # new_states: (h if GRU, (h,c) if LSTM) values for each layer after each sequence. 
		# each vector is num_layers X batch_size (1 in this case) X hidden_size
		
	def next_state(self,state,char):
		return self.state_from_sequence([char],state=state)

	def state_hash(self,state):
		return str(self.rnn_module.state_to_lists(state))
	
	def state_probs_dist(self,state):
		assert state.size()[1] == 1 if self.rnn_module.RNNClass=="GRU" else state[0].size(1)==1, "state_probs_dist called with state with batch size not 1 (state size: "+str(state.size())+")"
		dist = self.rnn_module.state_to_distribution(state) # batch_size X num_output_tokens
		return dist[0].tolist()
		
	def state_char_prob(self,state,char): 
		## assumes its a single state i.e. batch_size == 1
		return self.state_probs_dist(state)[self.char2int[char]]

	def weight(self,sequence,as_prefix=False,state=None,return_tensor=False):
		idxs = [self.char2int[w] for w in sequence]
		all_dists = self.rnn_module.all_distributions_for_seq(idxs,state=state,actually_get_logs=True)
		assert len(all_dists) == (len(idxs)+1) # first len(sequence) are states before each token, last one is there to guess EOS
		log_weights = [dist[i] for i,dist in zip(idxs,all_dists)] 
		if not as_prefix:
			log_weights.append(all_dists[-1][self.char2int[self.end_token]]) # log probability last state gave to EOS
		res = torch.exp(sum(log_weights)) if log_weights else self.rnn_module.tensor(1) # if no weights to multiply - i.e. empty sequence, as prefix: then just hardcode
		return res if return_tensor else res.item()

	def weights_for_sequences_with_same_prefix(self,prefix,suffixes):
		state = self.state_from_sequence(prefix)
		w = self.weight(prefix,as_prefix=True,return_tensor=True)
		all_idxs = [[self.char2int[t] for t in s] for s in suffixes]
		all_dists = self.rnn_module.all_distributions_for_seqs(all_idxs,state=state,actually_get_logs=True)

		return [(w*torch.exp(sum(self._log_transitions(idxs,dists)))).item() for idxs,dists in zip(all_idxs,all_dists)]

	def _log_transitions(self,idxs,log_dists,add_end_token=True):
		assert len(log_dists) == (len(idxs)+1), "|log_dists|: "+str(len(log_dists))+", indexes: "+str(idxs)+" (length: "+str(len(idxs))+")\n log_dists: "+str(log_dists) # states from before idxs started to after they finished
		res = [dist[i] for i,dist in zip(idxs,log_dists[:-1])]
		if add_end_token:
			res.append(log_dists[-1][self.char2int[self.end_token]])
		return res


	def detached_average_loss_on_group(self,sequences,step_size=100,seqs_at_a_time=100): # not useful for backprop! drops memory every now and then
		sequences = sorted(sequences,key=len,reverse=True)
		idxs = [[self.char2int[c] for c in s] for s in sequences] 
		loss = 0
		num_batches = int(len(sequences)/seqs_at_a_time)+1 # +1 for rounding and weird float errors or whatever
		for i in range(num_batches):
			loss += do_timed(process_time(),
				self.rnn_module.detached_negative_total_logs_on_seqs(\
					idxs[i*seqs_at_a_time:(i+1)*seqs_at_a_time],
					self.char2int[self.end_token],
					step_size=step_size,
					already_sorted=True),\
				"getting loss for group "+str(i+1)+" of "+str(num_batches))
		return loss/(sum([len(s) for s in sequences])+len(sequences))


	def average_loss_on_group(self,sequences): # average loss per prediction (including EOS after each sequence) of all the sequences
		all_idxs = [[self.char2int[t] for t in s] for s in sequences]
		all_dists = self.rnn_module.all_distributions_for_seqs(all_idxs,actually_get_logs=True)
		# loss on one word: -log(weight(word))
		return -sum([sum(self._log_transitions(idxs,dists,add_end_token=True)) for idxs,dists in zip(all_idxs,all_dists)])/\
						(sum([len(s) for s in sequences])+len(sequences)) # not adding .item() because will want to backprop this

	def naive_loss_on_suffixes(self,pref_suf_pairs):
		pref_suf_pairs = [p for p in pref_suf_pairs if len(p[1])>0]
		assert pref_suf_pairs, "no non-empty suffixes in loss (naive implementation): nothing to learn here"
		def loss_on_suffix(pref,suff):
			state = self.state_from_sequence(pref)
			ends_with_eos = (suff[-1]==self.end_token)
			suff = suff[:-1] if ends_with_eos else suff
			idxs = [self.char2int[t] for t in suff]
			dists = self.rnn_module.all_distributions_for_seq(idxs,state=state,actually_get_logs=True)
			return -sum(self._log_transitions(idxs,dists,add_end_token=ends_with_eos))
		return sum([loss_on_suffix(*p) for p in pref_suf_pairs])/\
				(sum([len(p[1]) for p in pref_suf_pairs])+len([p for p in pref_suf_pairs if p[1][:-1]==self.end_token]))
	

	def loss_on_suffixes(self,pref_suf_pairs,step_size_for_prefs,new_func = True): # suffixes have to include their own EOS's if they want to consider them because unclear if they are necessarily at ends of sequences
		pref_suf_pairs = [p for p in pref_suf_pairs if len(p[1])>0] # nothing to learn from empty suffixes
		assert pref_suf_pairs, "no non-empty suffixes in loss (fast implementation): nothing to learn here" # make sure still something left after that
		pref_suf_pairs = sorted(pref_suf_pairs,key=lambda x:len(x[0]),reverse=True) # sort by descending order of prefix length
		# no rearranging the pairs or things in general from here

		empty_prefs_start = next((i for i,p in enumerate(pref_suf_pairs) if len(p[0])==0),len(pref_suf_pairs))
		# start with just the empty states
		all_states = self.rnn_module._expand_state_for_batches(self.initial_state(),len(pref_suf_pairs)-empty_prefs_start) 
		if empty_prefs_start>0: # at least one nonempty pref
			nonempty_pref_indices_seqs = [[self.char2int[t] for t in p[0]] for p in pref_suf_pairs[:empty_prefs_start]]		
			with torch.no_grad(): # gonna be detaching these - doing truncated backprop - so no need to keep 'computation graph' during forward. (which gets huge..)
				if new_func:
					nonempty_pref_new_states = self.rnn_module.detached_last_state_from_sorted_seqs(nonempty_pref_indices_seqs,step_size_for_prefs)
				else:
					_, nonempty_pref_new_states, _ =  self.rnn_module.states_from_sorted_nonzero_seqs(nonempty_pref_indices_seqs) 
					nonempty_pref_new_states = self.rnn_module.detached_state(nonempty_pref_new_states) 
			# new_states (i.e. hidden): (h if GRU, (h,c) if LSTM) values for each layer after each sequence. each vector in hidden is num_layers X batch_size X hidden_size			
			
			# add these nonempty prefs, to the beginning because that's the order the pairs are all sorted in
			all_states = self.rnn_module.concat_states((nonempty_pref_new_states,all_states)) # num_layers X len(pref_suf_pairs) X hidden_size
		
		# remove the end_token from each suffix that has it, we want that for eval not for stepping
		eos_suff_indices = [i for i,p in enumerate(pref_suf_pairs) if p[1][-1]==self.end_token] 
		for i in eos_suff_indices:
			pref_suf_pairs[i] = (pref_suf_pairs[i][0],pref_suf_pairs[i][1][:-1]) 

		suff_indices_seqs = [[self.char2int[t] for t in p[1]] for p in pref_suf_pairs]
		all_dists = self.rnn_module.all_distributions_for_seqs(suff_indices_seqs,state=all_states,actually_get_logs=True)

		all_log_transitions = [self._log_transitions(idxs,dists,add_end_token=(index in eos_suff_indices)) for idxs,dists,index in\
														 zip(suff_indices_seqs,all_dists,list(range(len(pref_suf_pairs))))]
		return -sum(sum(l) for l in all_log_transitions)/(sum([len(s) for s in suff_indices_seqs])+len(eos_suff_indices))


	def max_param_value(self):
		max_val_per_tensor = [t.abs().max().item() for t in self.rnn_module.parameters()]
		return max(max_val_per_tensor)
		# trying to see what's up: what is the maximum weight inside the model?
		
	def max_and_avg_gradient_on_word(self,word):
		self.train() # needs to be in train to allow backward (at least does in cuda, cpu doesnt complain..)
		loss = self.average_loss_on_group([word])
		loss.backward()
		flattened_params_list = [t.grad.abs().detach().flatten() for t in self.rnn_module.parameters()]
		flattened_params_tensor = torch.cat(tuple(flattened_params_list))
		res = flattened_params_tensor.max().item(), flattened_params_tensor.mean().item()
		self.eval() # return to the norm
		return res

	def _update_stats_except_train_and_validation(self,representative_word):
		self.max_param_values.append(self.max_param_value())
		max_g,avg_g = self.max_and_avg_gradient_on_word(representative_word)
		self.avg_gradients.append(avg_g)
		self.max_gradients.append(max_g)

	def plot_stats_history(self,plots_path=None):
		filename = lambda x:plots_path+"/"+x+".png" if not None is plots_path else None
		chronological_scatter(self.training_losses,title="average training losses since initiation",filename=filename("training_loss"))
		chronological_scatter(self.validation_losses,vec2=self.training_losses,title="average validation vs classification losses since initiation",vec1label="validation",vec2label="train",filename=filename("validation_vs_training_loss"))
		chronological_scatter(self.max_param_values,title="max param values since initiation",filename=filename("max_param_values"))
		chronological_scatter(self.avg_gradients,title="average gradients on representative word since initiation",filename=filename("avg_gradients"))
		chronological_scatter(self.max_gradients,title="max gradient on representative word since initiation",filename=filename("max_gradients"))


def dict_file(full_rnn_folder):
	return full_rnn_folder + "/RNNTokenPredictor.dict"

def module_file(full_rnn_folder):
	return full_rnn_folder + "/RNNModule.pt"

def optimiser_file(full_rnn_folder):
	return full_rnn_folder + "/Optimiser.pt"

def save_rnn(full_rnn_folder,rnn,optimiser=None):
	was_cuda = rnn.rnn_module.using_cuda
	rnn.cpu(just_saving=True) # seems like a better idea if want to open on another computer later
	prepare_directory(full_rnn_folder,includes_filename=False)
	overwrite_file(rnn._to_dict(),dict_file(full_rnn_folder))
	torch.save(rnn.rnn_module.state_dict(),module_file(full_rnn_folder))
	if not None is optimiser:
		torch.save(optimiser.state_dict(),optimiser_file(full_rnn_folder))
	pass
	if was_cuda: # put it back!!
		rnn.cuda(returning_from_save=True)


def load_rnn(full_rnn_folder,quiet=False,with_optimiser=False,learning_rate=None):
	source_dict = load_from_file(dict_file(full_rnn_folder),quiet=quiet)
	if None is source_dict:
		return None
	res = RNNTokenPredictor(None,-1,-1,-1,None,source_dict=source_dict)
	res.rnn_module = RNNModule(res.input_dim,res.hidden_dim,res.num_layers,res.RNNClass,\
		len(res.internal_alphabet),len(res.input_alphabet),res.dropout)
	res.rnn_module.load_state_dict(torch.load(module_file(full_rnn_folder)))
	res.eval() # always load for eval, whenever training can set explicitly to train and then return to eval
	res.cuda() if torch.cuda.is_available() else res.cpu()
	if not with_optimiser:
		return res
	optimiser = torch.optim.Adam(res.rnn_module.parameters(),lr=learning_rate)
	optimiser.load_state_dict(torch.load(optimiser_file(full_rnn_folder)))
	return res, optimiser

class TrainingInfo:
	def __init__(self,rnn,validation_set,train_set,\
				 batches_before_validation_cutoff,training_prints_file,\
				 full_rnn_folder,start):
		self.training_prints_file = training_prints_file
		self.start = start
		# i.e. assumes no change in training or validation set
		self.best_validation_loss = np.inf if len(rnn.validation_losses)==0 else rnn.validation_losses[-1]
		self.best_train_loss = np.inf if len(rnn.training_losses)==0 else rnn.training_losses[-1]
		print("best validation loss is:",self.best_validation_loss,file=self.training_prints_file,flush=True)
		self.temp_rnn_folder = full_rnn_folder+"/training_savepoints" # when running several, may end up with overlap if just truncating this to int
		self.batches_since_validation_improvement = 0
		self.rnn = rnn
		self.trainer = None # for now
		self.save_rnn()
		self.validation_set = validation_set
		self.train_set = train_set
		self.batches_before_validation_cutoff = batches_before_validation_cutoff


	def validation_and_train_improving_overall(self,step_size_for_progress_checks,seqs_at_a_time,already_know_it_didnt_here=False):
		def check_improved_now():
			self.rnn.eval() # just in case it broke while in train
			with torch.no_grad(): # just evaling, don't care for backprop
			# cba with efficient implementation that wont cause out of memory on full linux data rn (though should later probably), so say hello to our old friend languagemodel
				print("just checking losses real slow",flush=True)
				start_time = process_time()
				print("train set total size:",sum([len(s) for s in self.train_set]),",validation set total size:",sum([len(s) for s in self.validation_set]),flush=True)
				# lm_rnn = LanguageModel(self.rnn)
				# new_validation_loss = lm_rnn.average_loss_on_group(self.validation_set)
				# new_train_loss = lm_rnn.average_loss_on_group(self.train_set)
				new_validation_loss = self.rnn.detached_average_loss_on_group(self.validation_set,step_size=step_size_for_progress_checks,seqs_at_a_time=seqs_at_a_time)
				new_train_loss = self.rnn.detached_average_loss_on_group(self.train_set,step_size=step_size_for_progress_checks,seqs_at_a_time=seqs_at_a_time)
				print("done checking losses, that took:",process_time()-start_time,flush=True)

			self.rnn.validation_losses.append(new_validation_loss)
			self.rnn.training_losses.append(new_train_loss)
			if new_validation_loss <= self.best_validation_loss and new_train_loss<=self.best_train_loss:
				self.best_validation_loss = new_validation_loss
				self.best_train_loss = new_train_loss
				self.batches_since_validation_improvement = 0
				self.save_rnn()
				print("☆☆ reached new best validation & train loss:",self.best_validation_loss,"☆☆ (e^loss=",pow(math.e,self.best_validation_loss),")",file=self.training_prints_file,flush=True)
				return True
			print("☹☹ validation and train not improved here ☹☹",file=self.training_prints_file,flush=True)
			return False

		if already_know_it_didnt_here or not check_improved_now(): # when "already_know_it_didnt_here" is True, it's just been reverted to an older version following an error, so don't want to recompute measures
			self.batches_since_validation_improvement += 1
			if self.batches_since_validation_improvement > self.batches_before_validation_cutoff:
				print("validation stopped improving (",self.batches_since_validation_improvement,\
					"iters without improvement), reverting to best by validation loss",file=self.training_prints_file,flush=True)
				return False
		print("time since start:",process_time()-self.start,file=self.training_prints_file,flush=True) # gotta flush else how will i know these things are getting ahead.. they just seem to dump when theyre done
		return True

	def save_rnn(self):
		save_rnn(self.temp_rnn_folder,self.rnn,optimiser=self.trainer)

	def reload_and_get_best_rnn(self,learning_rate=None,with_optimiser=True):
		if with_optimiser: # reloads optimiser as well
			self.rnn, self.trainer = load_rnn(self.temp_rnn_folder,with_optimiser=True,learning_rate=learning_rate)
		else:
			self.rnn = load_rnn(self.temp_rnn_folder,with_optimiser=False)
		return self.rnn

	def delete_metadata(self):
		shutil.rmtree(self.temp_rnn_folder)

def make_shuffled_batches(sequences,batch_size): # dont shuffle sequences in place, using them to keep track of some representative word for debugging stuff
	s2 = copy(sequences)
	shuffle(s2)
	if batch_size == np.inf:
		return [s2]
	res = [s2[i*batch_size:i*batch_size+batch_size] for i in range(int(len(s2)/batch_size)+1)] # last batch might be smaller
	return res if res[-1] else res[:-1] # drop empty last one if necessary


def _check_learning_rates_and_iterations(iterations_per_learning_rate,learning_rates):
	if isinstance(iterations_per_learning_rate,int): 
		if isinstance(learning_rates,float):
			learning_rates = [learning_rates]	
		iterations_per_learning_rate = [iterations_per_learning_rate]*len(learning_rates)
	if isinstance(learning_rates,float):
		assert len(iterations_per_learning_rate)==1, "if giving list of iterations per learning rate then must give matching list of learning rates"
		learning_rates = [learning_rates]
	assert len(iterations_per_learning_rate) == len(learning_rates), "give same number of learning rates and iterations per learning rate"
	return iterations_per_learning_rate, learning_rates

def train_specific_batch(rnn,batch,learning_rate,using_pairs,trainer,step_size_for_prefs):
	def total_batch_size():
		if not using_pairs:
			return sum([len(s) for s in batch])
		else:
			return sum([len(s[0])+len(s[1]) for s in batch])
	print("just training batch real quick, using pairs:",using_pairs,", total batch size:",total_batch_size(),flush=True)
	start = process_time()
	rnn.train()
	trainer.zero_grad()
	loss = rnn.average_loss_on_group(batch) if not using_pairs else rnn.loss_on_suffixes(batch,step_size_for_prefs)
	res = loss.item()
	loss.backward()
	trainer.step()
	rnn.eval()
	print("finished training batch, that took:",process_time()-start,flush=True)
	return res


def loss_str(loss,d=5):
	return str(clean_val(loss,d))+" (e^loss: "+str(clean_val(pow(math.e,loss),d)) +")"

def make_pref_seq_pairs(train_set,backprop_depth,skip_length,end_token): 
# ideally use backprop_depth = n*skip_length for some n, so that every token 
# ends up in training exactly the same number of times (n)
	assert backprop_depth >= skip_length, "backprop_depth should be greater than skip_length"
	pairs = []
	for s in train_set:
		full_s = list(s)+[end_token]
		pairs += [ (s[:max(0,i)],full_s[max(0,i):max(0,i+backprop_depth)]) for i in \
		range(-backprop_depth,len(s),skip_length)  ] 
		# each token gets in a suffix approximately int(backprop_depth/skip_length) times 
		# (can mess up a little at the edges, but shouldn't be more than off by one, and all should get in at least once)
	return [p for p in pairs if len(p[1])>0] # drop the ones with empty suffixes 

def train_rnn(rnn,train_set,validation_set,full_rnn_folder,iterations_per_learning_rate=500,learning_rates=None,\
	batch_size=100,backprop_and_skip_depths=None,
	check_improvement_every=1,step_size_for_prefs=100,step_size_for_progress_checks=200,\
	progress_seqs_at_a_time=100,wee_sample_rate=np.inf,print_sample_joiner="",sample_cutoff = 50):
	# iterations per learning rate: number of iterations to spend on each learning rate, eg [10,20,30]. learning rates: normally start at 0.001 and do like 0.7 for the decays?
	iterations_per_learning_rate, learning_rates = _check_learning_rates_and_iterations(iterations_per_learning_rate,learning_rates)
	# print("training rnn with check_improvement_every=",check_improvement_every)
	def finish():
		# print("adding",process_time()-start,"to rnn time")
		rnn = ti.reload_and_get_best_rnn(with_optimiser=False)
		save_rnn(full_rnn_folder,rnn)
		ti.delete_metadata()
		rnn.total_train_time += (process_time()-start) # add all the time we wasted in here 
		print("reached average training loss of:",rnn.training_losses[-1],file=training_prints_file,flush=True)
		print("and average validation loss of:",rnn.validation_losses[-1],file=training_prints_file,flush=True)
		print("overall time spent training, including those dropped to validation:",process_time()-start,file=training_prints_file,flush=True)
		rnn.plot_stats_history(plots_path=full_rnn_folder+"/training_plots")
		print("finished plotting stats history",flush=True)
		return rnn

	if torch.cuda.is_available():
		rnn.cuda()

	using_pairs = not None is backprop_and_skip_depths
	if using_pairs:
		pref_suf_pairs = make_pref_seq_pairs(train_set,backprop_and_skip_depths[0],backprop_and_skip_depths[1],rnn.end_token)

	prepare_directory(full_rnn_folder,includes_filename=False)
	check_improvement_counter = 0
	with open(full_rnn_folder+"/training_prints.txt","a") as training_prints_file:
		print("training rnn:",rnn.informal_name,file=training_prints_file,flush=True)
		start = process_time()
		# print("current rnn train time is:",rnn.total_train_time)

		ti = TrainingInfo(rnn,validation_set,train_set,0,training_prints_file,full_rnn_folder,start)
		representative_word = train_set[0]
		sample_counter = 0
		try:
			l_r_c = 0
			for learning_rate,iterations in zip(learning_rates,iterations_per_learning_rate):
				ti.trainer = torch.optim.Adam(rnn.rnn_module.parameters(),lr=learning_rate)
				l_r_c+=1
				ti.batches_since_validation_improvement = 0
				for i in range(iterations):
					break_iter = False
					batches = make_shuffled_batches((train_set if not using_pairs else pref_suf_pairs),batch_size)
					ti.batches_before_validation_cutoff = int(len(batches)/check_improvement_every) + 2
					batch_c=0
					for b in batches:
						batch_c += 1
						print("lr",l_r_c,"of",len(learning_rates),"(",clean_val(learning_rate,6),\
							"), iter.",i+1,"of",iterations,", batch",batch_c,"of",len(batches),file=training_prints_file,flush=True,end="")
						had_error = False
						try:
							batch_start = process_time()
							loss = train_specific_batch(rnn,b,learning_rate,using_pairs,ti.trainer,step_size_for_prefs)
							print(" finished, that had loss:",loss_str(loss,5),"and took:",clean_val(process_time()-batch_start),
								". (batch size:",len(b),"seqs, total len:",sum(len(s) for s in b),")",file=training_prints_file,flush=True)
							if sample_counter%wee_sample_rate==0:
								print("small sample (up to",sample_cutoff,"tokens):\n",
									print_sample_joiner.join(LanguageModel(rnn).sample(cutoff=sample_cutoff)),file=training_prints_file)
							sample_counter += 1
						except RuntimeError as e:
							print("\ntraining with learning rate",learning_rate,"hit error:",str(e),file=training_prints_file,flush=True)
							rnn = ti.reload_and_get_best_rnn(learning_rate=learning_rate) # something went wrong, get the best rnn back
							had_error = True
						rnn._update_stats_except_train_and_validation(representative_word) # weird partial updates here but w/e
						check_improvement_counter += 1
						if check_improvement_counter%check_improvement_every==1:
							if not ti.validation_and_train_improving_overall(step_size_for_progress_checks,\
								progress_seqs_at_a_time,already_know_it_didnt_here=had_error): 
								# move to next learning rate
								break_iter = True
								break
							rnn.plot_stats_history(plots_path=full_rnn_folder+"/training_plots")
					if break_iter:
						break
				
		except KeyboardInterrupt:
			print("stopped by user - losses may be different than those last recorded",file=training_prints_file,flush=True)
		ti.validation_and_train_improving_overall(step_size_for_progress_checks,\
								progress_seqs_at_a_time,already_know_it_didnt_here=had_error)
		rnn.plot_stats_history(plots_path=full_rnn_folder+"/training_plots")
		return finish()


# tests of all kinds
def diff_under_signal(r1,r2):
	return abs(r1-r2)/min(abs(r1),abs(r2)) < 1e-4

def weight_sanity_check(a,seq,as_prefix=False):
	print("=========\nweight sanity check on seq:",seq)
	direct_w = a.weight(seq,as_prefix=as_prefix)
	print("model's weight on whole sequence:",direct_w)	
	s = a.initial_state()
	transitions = []
	res = 1
	for t in seq:
		transitions.append(a.state_char_prob(s,t))
		res *= transitions[-1]
		s = a.next_state(s,t)
	if not as_prefix:
		transitions.append(a.state_char_prob(s,a.end_token))
		res*=transitions[-1]
	print("model's transitions on whole sequence:",transitions)
	print("product of transitions:",res)
	print("\t\t\tdiff: [[",res-direct_w,"]]")
	assert diff_under_signal(res,direct_w)

def states_sanity_check(a,seq):
	print("=========\nstates sanity check on seq:",seq)
	s1 = a.state_from_sequence(seq)
	s2 = a.initial_state()
	for t in seq:
		s2 = a.next_state(s2,t)
	assert a.rnn_module.states_equal(s1,s2)
	return s1,s2

def same_pref_sanity_check(a,pref,suffs):
	print("=========\nsame pref sanity check on pref:",pref,", suffs:",suffs)
	res1 = a.weights_for_sequences_with_same_prefix(pref,suffs)
	res2 = [a.weight(pref+s) for s in suffs]
	state = a.state_from_sequence(pref)
	w = a.weight(pref,as_prefix=True)
	res3 = [w*a.weight(s,state=state) for s in suffs]
	print("fast function          :",res1)
	print("separately             :",res2)
	print("separately, from middle:",res3)
	print("\t\t\tdiff1:		: [[",np.array(res1)-np.array(res2),"]]")
	print("\t\t\tdiff2:		: [[",np.array(res1)-np.array(res3),"]]")
	for r1,r2 in [(res1,res2),(res1,res3),(res2,res3)]:
		for a,b in zip(r1,r2):
			assert diff_under_signal(a,b)

def with_prefs_loss_sanity_check(a,pref_suf_pairs):
	print("=========\nwith prefs loss sanity check on pairs:",pref_suf_pairs)
	res1 = do_timed(process_time(),a.loss_on_suffixes(pref_suf_pairs,100),      "quick (hopefully) loss on suffs").item()
	res2 = do_timed(process_time(),a.naive_loss_on_suffixes(pref_suf_pairs),"naive loss on suffs            ").item()
	print("fast function    :",res1)
	print("naive function   :",res2)
	print("\t\t\tdiff:     [[",res1-res2,"]]")
	assert diff_under_signal(res1,res2)

def max_step_for_pytorch_loss_check(a,long_pref_suf_pairs,step_size):
	print("=========\nmax step for pytorch, check using loss========")
	print("pref_suf_pairs are:",long_pref_suf_pairs)
	res1 = do_timed(process_time(),a.loss_on_suffixes(long_pref_suf_pairs,step_size),      "limited step size loss on suffs").item()
	res2 = do_timed(process_time(),a.loss_on_suffixes(long_pref_suf_pairs,step_size,new_func=False),      "one step for pref loss on suffs").item()
	print("limited step     :",res1)
	print("full step        :",res2)
	print("\t\t\tdiff:     [[",res1-res2,"]]")
	assert diff_under_signal(res1,res2)

def average_loss_with_stepsize_check(a,long_seqs,step_size=100):
	print("=========\naverage loss on group, with max step in comp========")
	res1 = do_timed(process_time(),a.detached_average_loss_on_group(long_seqs,step_size),"limited step size average loss on group")
	res2 = do_timed(process_time(),a.average_loss_on_group(long_seqs).item(),"straightforward average loss on group")
	print("limited step:    :",res1)
	print("one go:          :",res2)
	print("\t\t\tdiff:      [[",res1-res2,"]]")
	assert diff_under_signal(res1,res2)



def run_checks(a): # assumes alphabet includes a-z
	weight_sanity_check(a,"")
	weight_sanity_check(a,"",as_prefix=True)
	weight_sanity_check(a,"asfdsfb")
	weight_sanity_check(a,"asdfsb",as_prefix=True)
	states_sanity_check(a,"")
	states_sanity_check(a,"dfgdg")
	same_pref_sanity_check(a,"",[""])
	same_pref_sanity_check(a,"",["sdfsdf"])
	same_pref_sanity_check(a,"ssf",[""])
	same_pref_sanity_check(a,"sfsfd",["sfsf"])
	same_pref_sanity_check(a,"",["","sdfs","as","ghdfad","s",""])
	same_pref_sanity_check(a,"dsfsf",["","sdfs","as","ghdfad","s",""])
	with_prefs_loss_sanity_check(a,[("","s")])
	with_prefs_loss_sanity_check(a,[("dfs","s")])
	with_prefs_loss_sanity_check(a,[("s","sd"),("sd","s")])
	with_prefs_loss_sanity_check(a,[("sd","s"),("s","sd")])
	with_prefs_loss_sanity_check(a,[("sd","sd"),("s","s")])
	with_prefs_loss_sanity_check(a,[("s","s"),("sd","sd")])
	with_prefs_loss_sanity_check(a,[("s","sdf"),("dfsfsfsf","df"),("sdf","d")])
	pairs = [("s","sdf"),("dfsfsfsf","df"),("sdf","d")]
	pairs_with_end = [(p[0],list(p[1])+[a.end_token]) for p in pairs]
	with_prefs_loss_sanity_check(a,pairs_with_end)
	with_prefs_loss_sanity_check(a,pairs+pairs_with_end) # check it handles mix of EOS and not EOS
	def string_of_len(X):
		return ''.join(random.choice(string.ascii_lowercase) for x in range(X))
	long_pref_suf_pairs = [(string_of_len(p),string_of_len(s)) for p,s in itertools.product([0,2,5,3],[1,5,3])]
	max_step_for_pytorch_loss_check(a,long_pref_suf_pairs,3)
	long_pref_suf_pairs = [(string_of_len(p),string_of_len(s)) for p,s in itertools.product([0,10,50,100,20],[10,50,20])]
	max_step_for_pytorch_loss_check(a,long_pref_suf_pairs,50)
	long_prefs = [string_of_len(l) for l in [1,3,0,6,9,3,2]]
	average_loss_with_stepsize_check(a,long_prefs,2)
	long_prefs = [string_of_len(l) for l in [10,50,0,100,200,330,250,214,30,100]]
	average_loss_with_stepsize_check(a,long_prefs,100)