# Training RNNs, and Extracting DFAs from them
The shell script train\_and\_extract\_all\_TACAS\_langs.sh will, for every one of the languages experimented on in the paper: 1. train an RNN on that language, in the subfolder rnns/[your subfolder as given in script]/[cfg description]/[timestamp] , and then 2. extract DFAs from that language using the L-star algorithm, saving the extracted DFAs in a further subfolder from the RNN's location. RNN, training, and extraction parameters (including the local threshold from which a sequence is considered 'rejected' by the RNN, as described in the paper) are set in the script.

It is also possible to train on a single language from the paper, or a subset of them, shown in the commented out loop at the bottom of the script. The paper's languages are defined in TACAS_cfgs.py, using base CFGs from all_cfgs.py. You can define new CFGs similarly to those in the paper, and experiment on those.

# PRS Inference

#### Program Description:
This program take as input a sequence of Deterministic Finite State Machines (DFAs)
It will infer, if it exists, a Pattern Rule Set (PRS) that describes the language that generates these DFAs.  It will
translate this PRS into a CFG.   The input DFAs can be generated from a Recurrent Neural Network and therefore this algorithm
can be used to reconstruct the hidden language learnt by the RNN.  The algorithm assumes there can and usually will be
"noise" in the DFAs, as the RNNs do not learn the language perfectly.   See paper for full details.


#### Program invocation:
`findRules.py [-d <path_name>] [-f <file_name>] [-w] [-v]`
`<path_name>` is the path to the directory containing the sequence of DFAs, relative to the current directory.
`<file_name>` is the name of the file containing the sequence of DFAs.
By default, `<path_name>` is the current directory (".") and `<file_name>` is the file "all_dfas.gz".
`<path_name>` can be changed using the parm `-d <path_name>`.   Then `<current_dir>/<path_name>` contains the file all\_dfas.gz
`<file_name>` can be changed using the parm `-f <file_name>`.  Then the sequence of DFAs is in the file `<file_name>`
The parm -v, if specified, gives verbose output to the terminal
The parm -w, if specified, causes the output file, specifying the discovered PRS, to be written to the directory
    `<path_name>`.  The name of this output file will begin with "discovered\_patterns\_and\_rules-" and include the date
    it was written and Threshold value.
The patterns filtered out will be written to the file ending with "Filtered" in the directory `<path_name>`

Configuration options are given in config.py.   By default, -w is set, -v is not, and PATTERN_THRESHOLD = 1.

NOTE: The terminal symbols of the output CFG actually refer to finite state machines.  These FSMs are also given in the
output.   It is trivial to replace these terminals by more CFG productions that describe the regular expression given
by the FSMs.


# TACAS Experiments Data
All extracted DFAs from the experiments in the TACAS paper are available in a zip at https://cs.technion.ac.il/~sgailw/tacas_2021.html.
Each sub-folder contains one experiment, giving the list of DFAs generated from the RNN for that language (all\_dfas.gz), the plots of the DFAs (dfa\_pngs), and the results
of running the PRS inference algorithm on those DFAs, given in a file with the name:
discovered\_patterns\_and\_rules-<date\_experiment\_ran>-Thresh=<Threshold\_value>
