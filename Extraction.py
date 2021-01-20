from time import process_time
from ObservationTable import TableTimedOut
from DFA import DFA
from Teacher import Teacher
from Lstar import run_lstar
from Helper_Functions import prepare_directory

def extract(rnn,output_dfas_folder,prints_file,time_limit = 50,initial_split_depth = 10,starting_examples=None,
			token_predictor_samples=-1,token_predictor_cutoff=-1,emptyseq=""):
	print("provided counterexamples are:",starting_examples,file=prints_file)
	if token_predictor_samples>0:
		print("will make",token_predictor_samples,"sample attempts of length <=",token_predictor_cutoff,\
			"before accepting any equivalence query",file=prints_file)
	else:
		print("not using lm-based samples to seek counterexamples",file=prints_file)
		
	prepare_directory(output_dfas_folder,includes_filename=False)

	guided_teacher = Teacher(rnn,output_dfas_folder,prints_file,\
	 num_dims_initial_split=initial_split_depth,starting_examples=starting_examples,\
	 token_predictor_samples=token_predictor_samples,token_predictor_cutoff=token_predictor_cutoff)
	start = process_time()
	try:
	    run_lstar(guided_teacher,time_limit,prints_file,emptyseq)
	except KeyboardInterrupt: #you can press the stop button in the notebook to stop the extraction any time
	    print("lstar extraction terminated by user",file=prints_file)
	except TableTimedOut:
	    print("observation table timed out during refinement",file=prints_file)
	end = process_time()
	extraction_time = end-start

	dfa = guided_teacher.dfas[-1]
	dfa.draw_nicely(maximum=60,filename=output_dfas_folder+"/final")

	print("overall guided extraction time took: " + str(extraction_time),file=prints_file)

	print("generated counterexamples were: (format: (counterexample, counterexample generation time))",file=prints_file)
	print('\n'.join([str(a) for a in guided_teacher.counterexamples_with_times]),file=prints_file)
	return dfa, guided_teacher.dfas