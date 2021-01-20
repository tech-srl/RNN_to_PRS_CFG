from ContextFreeGrammar import CFG
cfgs = {}



def dyck_i(num_parans=2,force_outer=False,
	empty_weight=0.35,double_weight=0.3,depth_weight=0.5): # with no REs inside
	# S -> SS | del_i  S led_i |  ""
	start_seq = ["D"] if force_outer else ["S"]

	res = CFG(["S","D"],start_seq,name="dyck_"+str(num_parans),sample_as_strings=False)
	paran_pairs = [("p"+str(i+1),"q"+str(i+1)) for i in range(num_parans)]
	
	[res.add_rule("D",[p1,"S",p2]) for p1,p2 in paran_pairs] 
	
	res.add_rule("S",[],weight=empty_weight)
	res.add_rule("S",["S","S"],weight=double_weight)
	[res.add_rule("S",[p1,"S",p2],weight=depth_weight/num_parans) for p1,p2 in paran_pairs]
	return res
cfgs["dyck_i"]=dyck_i

def REdyck(paranset_pairs=[(["a"],["b"])],neutral_strings=[],force_outer=False,
	empty_weight=0.35,double_weight=0.3,depth_weight=0.5,neutral_weight=0):
	start_seq = ["D"] if force_outer else ["S"]
	res = CFG(["S","D"],start_seq,name="REdyck_"+str(paranset_pairs))
	
	num_pairs = sum(len(ps1)*len(ps2) for ps1,ps2 in paranset_pairs)
	# print("making grammar with",num_pairs,"possible paran pairs")
	for ps1,ps2 in paranset_pairs:
		for p1 in ps1:
			for p2 in ps2:
				res.add_rule("D",[p1,"S",p2])
				res.add_rule("S",[p1,"S",p2],weight=depth_weight/num_pairs)

	for s in neutral_strings:
		res.add_rule("S",[s],weight=neutral_weight/len(neutral_strings))
	
	if not neutral_strings:
		empty_weight += neutral_weight
	res.add_rule("S",[],weight=empty_weight)
	res.add_rule("S",["S","S"],weight=double_weight)
	return res
cfgs["REdyck"]=REdyck
	

def cfg_xnyn(x_vals=["a"],y_vals=["b"],av_len=50): # average length.. ish
	# res = CFG(nonterminals,"initial-DFA","name of grammar")
	res = CFG("SXY","S",name="xnbn_"+str(x_vals)+"_"+str(y_vals))
	# res.add_rule("name-of-LHS-nonterminal","RHS", optional weight)
	res.add_rule("S","")
	res.add_rule("S","XSY",weight=(av_len/2))
	for x in x_vals:
		res.add_rule("X",x)
	for y in y_vals:
		res.add_rule("Y",y)
	return res
cfgs["xnyn"]=cfg_xnyn


def alternating_delimiter_2(av_len=50):
	res = CFG("SAB","S",name="alternating_delimiter_2")
	res.add_rule("S","A")
	res.add_rule("S","B")
	res.add_rule("A","(B)",weight=av_len/2)
	res.add_rule("A","")
	res.add_rule("B","{A}",weight=av_len/2)
	res.add_rule("B","")
	return res
cfgs["alternating_delimiter_2"]=alternating_delimiter_2