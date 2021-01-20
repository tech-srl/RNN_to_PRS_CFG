from Helper_Functions import prepare_directory

def get_spanned(small_dfa,big_dfa):
	dfa1, dfa2 = small_dfa, big_dfa
	assert sorted(dfa1.alphabet)==sorted(dfa2.alphabet) # at least be over the same thing

	alphabet = dfa1.alphabet

	sink1 = dfa1.find_sink_reject()
	
	# explore the subset dfa and the new dfa in parallel, 
	# except for transitions going to the sink reject. mark everything met in the new dfa
	queue1 = [(dfa1.q0,dfa2.q0)]
	marked2 = []
	seent1 = []
	while queue1:
		q1,q2 = queue1.pop()
		if (q1 == sink1) or (q1 in seent1):
			continue
		marked2.append(q2)
		seent1.append(q1) # otherwise infinite loop...
		for a in alphabet:
			queue1.append((dfa1.delta[q1][a],dfa2.delta[q2][a]))
	return marked2


def highlight_diff(small_dfa,big_dfa,filename):
	spanned = get_spanned(small_dfa,big_dfa)
	sink_big = big_dfa.find_sink_reject()
	not_spanned = [q for q in big_dfa.Q if ((not q in spanned) and (not q==sink_big))]
	big_dfa.draw_nicely(filename=filename,mark_dict={'green':not_spanned,'gray':get_connectors(big_dfa,not_spanned)})

def get_connectors(big_dfa,new_nodes):
	sink = big_dfa.find_sink_reject()
	def goes_into_new(q):
		return next((True for a in big_dfa.alphabet if big_dfa.delta[q][a] in new_nodes),False)
	return set( [q for q in big_dfa.Q if goes_into_new(q) ] + [big_dfa.delta[q][a] for a in big_dfa.alphabet for q in new_nodes])

def highlight_all_diffs(dfas,folder):
	prepare_directory(folder,includes_filename=False)
	if len(dfas)<= 1:
		return
	for i in range(len(dfas)-1):
		filename = folder + "/dfa_"+str(i+1)+"_delta_"+str(i)
		highlight_diff(dfas[i],dfas[i+1],filename)

