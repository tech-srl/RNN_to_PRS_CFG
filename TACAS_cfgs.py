from all_cfgs import cfgs

TACAS_cfgs = {}


TACAS_cfgs["LG1"]		= ("xnyn",(['a'],['b'],50))
TACAS_cfgs["LG2"]		= ("xnyn",(['a','b'],['c','d'],50))
TACAS_cfgs["LG3"]		= ("xnyn",(['ab','cd'],['ef','gh'],50))
TACAS_cfgs["LG4"]		= ("xnyn",(['ab'],['cd'],50))
TACAS_cfgs["LG5"]		= ("xnyn",(['abc'],['def'],50))
TACAS_cfgs["LG6"]		= ("xnyn",(['ab','c'],['de','f'],30))
TACAS_cfgs["LG7"]		= ("dyck_i",(2,True,0.35,0.3,0.5)) 
TACAS_cfgs["LG8"]		= ("dyck_i",(3,True,0.35,0.3,0.5))	
TACAS_cfgs["LG9"]		= ("dyck_i",(4,True,0.35,0.3,0.5)) 
TACAS_cfgs["LG10.2"]	= ("REdyck",([(["(a"],["b)"])],[],True,0.35,0.3,0.5,0))
TACAS_cfgs["LG10.4"]	= ("REdyck",([(["(abc"],["xyz)"])],[],True,0.35,0.3,0.5,0))
TACAS_cfgs["LG10.5"]	= ("REdyck",([(["(abcd"],["wxyz)"])],[],True,0.35,0.3,0.5,0))
TACAS_cfgs["LG11"]		= ("REdyck",([(["ab","c"],["de","f"])],[],True,0.35,0.3,0.5,0))
TACAS_cfgs["LG12"]		= ("alternating_delimiter_2",())
TACAS_cfgs["LG13"]		= ("REdyck",([(["("],[")"])],["a","b","c"],\
								True,0.05,0.25,0.4,0.35))
TACAS_cfgs["LG14"]		= ("REdyck",([(["("],[")"]),(["{"],["}"])],\
								["a","b","c"],\
								True,0.05,0.3,0.5,0.3))
TACAS_cfgs["LG15"]		= ("REdyck",([(["("],[")"])],["abc","d"],\
								True,0.05,0.3,0.45,0.35))



import sys
def print_TACAS_grammar(n,with_weights=False,out_file=sys.stdout):
	d = TACAS_cfgs[n]
	a = cfgs[d[0]](*d[1])
	a.print_nicely(print_weights_too=with_weights,out_file=out_file)

def print_all_TACAS_grammars(with_weights=False,f=sys.stdout):
	for n in TACAS_cfgs:
		print("=================\n=============",file=f)
		print("going to print grammar:",n,file=f)
		print_TACAS_grammar(n,with_weights=with_weights,out_file=f)