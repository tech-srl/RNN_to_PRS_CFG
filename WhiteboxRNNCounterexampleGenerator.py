from copy import deepcopy
from time import process_time
from LanguageModel import LanguageModel

class WhiteboxRNNCounterexampleGenerator:
	def __init__(self,network,partitioning,starting_examples,prints_file,token_predictor_samples=-1,token_predictor_cutoff=-1):
		self.time_limit = None 
		self.whiteboxrnn = network
		self.partitioning = partitioning
		def quickfix(c):
			return tuple(c) if isinstance(c,list) else c
		self.starting_dict = {quickfix(cex):network.classify_word(cex) for cex in starting_examples}
		self.prints_file = prints_file
		self.token_predictor_samples = token_predictor_samples
		self.token_predictor_cutoff = token_predictor_cutoff
		return

	def set_time_limit(self,time_limit,start_time):
		self.time_limit = time_limit
		self.start_time = start_time

	def _get_counterexample_from(self,words):
		words = sorted(words,key=lambda x:len(x)) #prefer shortest possible counterexample
		# print("_get_counterexample_from got options:",words)
		for w in words:
			if not self.whiteboxrnn.classify_word(w) == self.proposed_dfa.classify_word(w):
				return w
		return None

	def _counterexample_from_classification_conflict(self,state_info):
		res = self._get_counterexample_from(state_info.paths)
		if None == res:
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			print("classification conflict didn't cause counterexample:")
			print("check your partitioning is consistent and transition function ")
			print("(from one continuous network state (R-State) to another) is correct ") 
			raise NoCounterexampleFromClassificationConflict()
		return res

	def _counterexample_from_cluster_conflict(self,old_info,new_info):
		q1 =old_info.dfa_state
		q2 = new_info.dfa_state
		prefixes = old_info.paths  + new_info.paths
		# print("old info paths:",old_info.paths)
		# print("new info paths:",new_info.paths)
		suffix = self.proposed_dfa.minimal_diverging_suffix(q1,q2)
		# print("got suffix:",suffix)
		return self._get_counterexample_from([p+suffix for p in prefixes])

	def _process_new_state_except_children(self,new_cluster,new_info):
		counterexample = None
		split = SplitInfo()

		old_info = self.cluster_information[new_cluster] if new_cluster in self.cluster_information else None
		full_info = old_info + new_info if not old_info == None else new_info

		if not new_info.accepting == (new_info.dfa_state in self.proposed_dfa.F):
			counterexample = self._counterexample_from_classification_conflict(new_info)
		elif not new_info.dfa_state == full_info.dfa_state:
			counterexample = self._counterexample_from_cluster_conflict(old_info,new_info)
			if counterexample == None:
				split = SplitInfo(agreeing_RStates=old_info.RStates, 
								  conflicted_RState=new_info.RStates[0]) # the one seen now, in new info
		else: #no conflicts, store state and continue processing it later
			self.cluster_information[new_cluster] = full_info

		return counterexample, split

	def _add_children_states(self,cluster):
		state_info = self.cluster_information[cluster]
		if not state_info.explored:
			# we explore a state only the first time we successfully visit and process it, and we store a state's
			# information in self.cluster_information only if we have successfully processed it.
			RState = state_info.RStates[0]
			state_info.explored = True
			for char,appendychar in zip(self.proposed_dfa.alphabet,self.proposed_dfa.appendy_alphabet):
				next_RState, pos = self.whiteboxrnn.get_next_RState(RState, char)
				path = state_info.paths[0] + appendychar
				# we only ever explore a state the first
				# time we find it, so, with the first path in its list of reaching paths
				next_dfa_state = self.proposed_dfa.delta[state_info.dfa_state][char]
				self.new_RStates.append(UnrollingInfo(next_dfa_state,path,next_RState,pos))

	def _process_top_pair(self):
		new_info = self.new_RStates.pop(0)
		self.new_RStates_backup = new_info #might want to unpop if we refine the partitioning and want to restart from here
		new_cluster = self.partitioning.get_partition(new_info.RStates[0])
		counterexample, split = self._process_new_state_except_children(new_cluster, new_info)
		if (counterexample == None) and (not split.has_info): #i.e. no conflicts
			self._add_children_states(new_cluster)
		return counterexample, split

	def _initialise_unrolling(self):
		self.cluster_information = {} 
		initial_RState, pos = self.whiteboxrnn.get_first_RState()
		self.new_RStates = [UnrollingInfo(self.proposed_dfa.q0,self.proposed_dfa.emptyseq,initial_RState,pos)]

	def _cex_from_starting_dict(self,dfa):
		for cex in self.starting_dict:
			if not dfa.classify_word(cex) == self.starting_dict[cex]:
				print("storing provided counterexample of length " + str(len(cex)),file=self.prints_file)
				return cex
		return None

	def _cex_from_token_predictor(self,dfa):
		print
		if self.token_predictor_samples <= 0:
			return
		lm = LanguageModel(self.whiteboxrnn)
		seen = set()
		for _ in range(self.token_predictor_samples):
			seq = lm.sample(cutoff=self.token_predictor_cutoff,from_seq=dfa.emptyseq) # an assumption we're working with simple char-level networks for now
			for i in range(len(seq)+1):
				s = seq[:i]
				if not s in seen:
					if not self.whiteboxrnn.classify_word(s) == dfa.classify_word(s):
						print("sampling rnn got counterexample of length "+str(len(s)),file=self.prints_file)
						return s
					seen.add(s)
		print("sampling rnn got no counterexamples",file=self.prints_file)
		return None

	def _out_of_time(self):
		if not None == self.time_limit:
			if (process_time() - self.start_time) > self.time_limit:
				return True
		return False

	def _split_was_clean(self,old_cluster,split):
		new_states_given_to_agreeing = list(set([self.partitioning.get_partition(vec) for vec in split.agreeing_RStates]))
		return self.partitioning.refinement_doesnt_hurt_other_clusters \
				and new_states_given_to_agreeing == [old_cluster] \
				and not self.partitioning.get_partition(split.conflicted_RState) == old_cluster

	def counterexample(self,dfa): 
		counterexample = self._cex_from_starting_dict(dfa)
		if not None == counterexample:
			return counterexample,counterexample_message(counterexample,self.whiteboxrnn)

		self.proposed_dfa = dfa
		while True: #main loop: restarts every time the partitioning is refined
			self._initialise_unrolling() # start BFS exploration of network abstraction with current partitioning
			while True: #inner loop: extracts according to the partitioning, comparing to the proposed dfa as it goes
				if self._out_of_time(): # note: putting this after all the next bits sometimes leaves the time limit unchecked for a very long time...
					return None, "lstar extraction not successful - ran out of time"
				if len(self.new_RStates) == 0: # seen everything there is to see here
					print("rnn state abstraction method found no counterexamples (found abstraction equivalent to this DFA)",file=self.prints_file)
					if not self.partitioning.had_initial_refine:
						print("however, have not yet done aggressive initial split.",file=self.prints_file)
						print("applying aggressive initial split between initial state and first visited not-initial state\n\
							   (i.e., state for sequence 'a' where 'a' is character that appears first in alphabet),\n\
							   for good measure. will then run equivalence query again.",file=self.prints_file)
						initial_RState, _ = self.whiteboxrnn.get_first_RState()
						first_other_RState, _ = self.whiteboxrnn.get_next_RState(initial_RState, self.proposed_dfa.alphabet[0])
						self.partitioning.refine([initial_RState],first_other_RState)
						break # restart unrolling. cleanest way out of here and v. cheap cause only in here b/c partition size is literally 1
					cex = self._cex_from_token_predictor(dfa)
					if None is cex:
						return None, "lstar successful: unrolling seems equivalent to proposed automaton"
					return cex,counterexample_message(cex,self.whiteboxrnn)
				counterexample, split = self._process_top_pair() # always returns a cex, or a split, or neither - but never both
				if not None == counterexample:
					return counterexample,counterexample_message(counterexample,self.whiteboxrnn)
				elif split.has_info:
					cluster_being_split = self.partitioning.get_partition(split.agreeing_RStates[0])
					self.partitioning.refine(split.agreeing_RStates,split.conflicted_RState)
					if self._split_was_clean(cluster_being_split,split):
						# the latest R-state got a new cluster of its own and absolutely nothing else changed,
						# so we can just reprocess this visitor and continue as if nothing happened
						self.new_RStates = [self.new_RStates_backup] + self.new_RStates
					else:
						print("split wasn't perfect: gotta start over",file=self.prints_file)
						break # clustering has changed, have to restart unrolling from the top


def counterexample_message(counterexample,rnn):
	return ("returning counterexample of length " + str(len(counterexample)) + ":\t\t" + str(counterexample) + 
		", this counterexample is " + ("accepted" if rnn.classify_word(counterexample)==True else "rejected") + 
		" by the given RNN.")

class SplitInfo: #todo: move this to quantisations and just give the whole thing over to the relevant function there, instead of unpacking it to 3 parameters here
	def __init__(self,agreeing_RStates=None,conflicted_RState=None):
		self.agreeing_RStates = agreeing_RStates
		self.conflicted_RState = conflicted_RState
		self.has_info = not (conflicted_RState == None)

class UnrollingInfo:
	def __init__(self,dfa_state,path,RState,accepting):
		self.explored = False
		self.dfa_state = dfa_state
		self.paths = [path]
		self.RStates = [RState]
		self.accepting = accepting

	def __add__(self, other):
		res = deepcopy(self)
		res.paths += other.paths
		res.RStates += other.RStates
		return res

class NoCounterexampleFromClassificationConflict(Exception):
	pass
