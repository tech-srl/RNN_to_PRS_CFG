from Helper_Functions import pick_index_from_distribution
import sys

class CFG:
	def __init__(self,nonterminals,initial_seq,name="CFG",sample_as_strings=True):
		self.nonterminals = nonterminals
		self.initial_seq = initial_seq
		self.rules = {NT:[] for NT in self.nonterminals}
		self.name = name
		self.sample_as_strings = sample_as_strings


	def print_nicely(self,print_weights_too=False,out_file=sys.stdout):
		print("cfg with name:",self.name,file=out_file)
		print("nonterminals:",list(self.nonterminals),file=out_file)
		print("initial seq (probably a single nonterminal):",self.initial_seq,file=out_file)
		print("all rules:",file=out_file)
		for n in self.rules:
			ders = self.rules[n]
			for r in ders:
				print(n,"    -->   ",r[0] if not print_weights_too else r,file=out_file)
		if self.sample_as_strings:
			print("final output is sampled as strings, i.e. if there is a derivation that looks like ['abc','ac'] then it is really just 'abcac' and contains 3 unique tokens.",file=out_file)		
		else:
			print("final output is not concatenated at end, i.e. if there is a derivation like ['p1','p2'] then it has length 2 and p1, p2 are treated as unique tokens",file=out_file)


	def add_rule(self,nonterminal,replacement,weight=1):
		self.rules[nonterminal].append((replacement,weight))

	def apply_rule(self,nonterminal,verbose):
		weights = [w for r,w in self.rules[nonterminal]]
		index = pick_index_from_distribution(weights)
		if verbose:
			print("applying rule:",nonterminal," --> ",self.rules[nonterminal][index][0])
		return self.rules[nonterminal][index][0] # [0]: give the rule, drop its weight

	def sample(self,max_expansions=1000,verbose=False):
		def next_nonterminal(r):
			return next((i for i,nt in enumerate(r) if nt in self.nonterminals),-1)
	
		res = self.initial_seq
		i = next_nonterminal(res)
		num_expansions = 0
		while (i > -1) and (num_expansions<max_expansions):
			res = res[:i] + self.apply_rule(res[i],verbose) + res[i+1:]
			i = next_nonterminal(res)
			num_expansions += 1
		if verbose:
			print("made",num_expansions,"expansions")
		if next_nonterminal(res) > -1:
			return None
		if self.sample_as_strings:
			return ''.join(res)
		return res

	def terminals(self):
		all_tokens = self.initial_seq
		for NT in self.rules: 
			for rule,weight in self.rules[NT]:
				all_tokens += rule			
		all_tokens = set(all_tokens) - set(self.nonterminals)
		if self.sample_as_strings:
			all_tokens = ''.join(all_tokens)
		return sorted(list(all_tokens)) 


def get_n_samples(cfg,n,max_expansions=1000,max_attempts=10,max_length=1000):
	res = []
	num_attempts = 0
	def drop_none(l):
		# print("samples contained",l.count(None),"Nones",flush=True)
		return [e for e in l if not None is e]
	def drop_overlong(l):
		len1 = len(l)
		res = [e for e in l if len(e)<=max_length]
		# print("samples contained",len1-len(res),"samples of length>",max_length," (now dropped)",flush=True)
		return res
	while (len(res) < n) and (num_attempts<max_attempts):
		res += drop_overlong(drop_none([cfg.sample(max_expansions=max_expansions) for _ in range(n)]))
		num_attempts+=1
		# print("attempt",num_attempts,". made",n,"samples, now have",len(res),"good samples overall",flush=True)

	res = res[:n] # in case got extras from later sample rounds
	assert len(res)==n, "could not sample enough complete sequences from given grammar"
	return res






