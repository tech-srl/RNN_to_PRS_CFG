from TACAS_cfgs import print_all_TACAS_grammars
from Helper_Functions import prepare_directory

import argparse 


parser = argparse.ArgumentParser(description='print all the TACAS grammars to a subdirectory of the rnns directory')

parser.add_argument('--subfolder',type=str,help="subfolder to save the grammars in",default=".")

args = parser.parse_args()

folder = "rnns/"+args.subfolder+"/"

prepare_directory(folder,includes_filename=False)

with open(folder+"grammars.txt","w") as f:
	print_all_TACAS_grammars(f=f)

with open(folder+"grammars_with_weights.txt","w") as f:
	print_all_TACAS_grammars(f=f,with_weights=True)