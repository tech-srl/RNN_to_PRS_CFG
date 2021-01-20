import sys, getopt
from Helper_Functions import load_from_file
from patterns import Pattern, PatternRepo
from compare2dfas import find_junk_state, remove_equiv_transitions, create_pattern_from_DFA, mark_patterns_in_dfa
from config import verbose, write_output, PATTERN_THRESHOLD, RULE_THRESHOLD
from indRules import IndRule,IndRuleRepo
from prs_globals import PatternType, Shape
import datetime

# This is the main program.   A brief description of the algorithm is here.  See paper for full details.
# An Pattern Rule Set takes a DFA and expands it to include new paths (and thereby recognize more strings) by applying
# a rule.
# We assume three types of rules:  an initial rule, that creates the first DFA, or expands an existing DFA from the
# initial state with a cyclic pattern, a sequential rule, that inserts a new acylic pattern between a composite pattern,
# and a cyclic rule, that inserts a cyclic pattern between a composite pattern.
#
# This program takes a sequence of DFAs, where each DFA in the sequence is assumed to be the expansion of the prior DFA
# by applying a rule from a PRS.   It will tolerate "noise" in the sequence of DFAs.
# The algorithm works in 4 phases:
#
# Phase 1: Discover patterns.
# This phase iterates over the sequence of DFAs and compares each DFA to its successor DFA to extract patterns.   We
# find those states and transitions in the successor DFA that is not in its predecessor DFA, and these connected regions
# are candidates for patterns.   We consider only those patterns that pass a certain threshold  - that appear a minimal
# number of times in the sequence of DFAs.
#
# Phase 2:  Confirm pattern locations in each DFA
# Once we have discovered a set of patterns that are ubiquitous in the sequence of DFAs, we go back over the sequence
# of DFAs and mark exactly where each patterns appears in each DFA.
#
# Phase 3: Discover the (hidden) pattern rule set (PRS)
# Once we have the location of each pattern in each DFA, we once again compare each DFA to its successor DFA.   The
# successor DFA presumably expands on its predecessor DFA by inserting a new pattern instance into the prior DFA
# using one of the 3 types of rules described above.
#
# Phase 4: Discover intial rules by examining patterns initiating the at Start state of the last DFA

# initialize input parameters
if __name__ == '__main__':
    # <path_name> is the path to the directory containing the sequence of DFAs, relative to the current directory.
    # <file_name> is the name of the file containing the sequence of DFAs.
    # By default, <path_name> is the current directory (".") and <file_name> is the file "all_dfas.gz".
    # <path_name> can be changed using the parm -d <path_name>.   Then <current_dir>/<path_name> contains the file all_dfas.gz
    # <file_name> can be changed using the parm -f <file_name>.  Then the sequence of DFAs is in the file <file_name>
    # the parm -v, if specified, gives verbose output to the terminal
    # the parm -w, if specified, causes the output file, giving the discovered PRS, to be written to the directory
    # <path_name>.  The name of this output file will begin with "discovered_patterns_and_rules-" and include the date
    # it was written and Threshold value.
    # the patterns filtered out will be written to the file ending with "Filetered" in the directory <path_name>
    # Configuration options are given in config.py.   By default, -w is set, -v is not, and PATTERN_THRESHOLD = 1.
    try:
        # if char has following colon then parameter required.  otherwise no paramater
        opts, remainder = getopt.getopt(sys.argv[1:], "d:f:vw")
    except getopt.GetoptError:
        print('Usage: findRules.py [-d <path_name>] [-f <file_name>] [-w] [-v]')
        sys.exit(2)
    path_name_defined = False
    file_name_defined = False
    for opt,arg in opts:
        if opt == '-d':
            path_name = arg
            path_name_defined = True
        elif opt == '-v':
            verbose = True
        elif opt == '-w':
            write_output = True
        elif opt == '-f':
            file_name = arg
            file_name_defined = True
        else:
            print('Usage: findRules.py [-d <path_name>] [-f <file_name>] [-w] [-v]')
            sys.exit()
    if not path_name_defined:
        path_name = "."
    if not file_name_defined:
        file_name = "all_dfas.gz"
    if write_output:
        output_file = path_name + "/discovered_patterns_and_rules-" + str(datetime.date.today()) + "-Thresh=" + str(PATTERN_THRESHOLD)
        output_handle = open(output_file,"w+")
        output_filtered_file = path_name + "/discovered_patterns_and_rules-" + str(datetime.date.today()) + "Filtered"
        output_filtered_handle = open(output_filtered_file, "w+")
    fsms = load_from_file(path_name + "/" + file_name, quiet=False)
    num_fsms = len(fsms)
    print("Number of fsms = " + str(num_fsms))

    # DATA STRUCTURE INITIATION

    # 1. equiv_states is a dictionary of dictionaries.  It maps a DFA fi and state s1 to its equivalent state s2 in f(i+1).
    # For instance, if equiv_states[fi][s1] == s2, then s2 is the state in f(i+1) "equivalent" to state s1 in f1.
    # (where f(i+1) is the DFA successor to DFA fi).
    equiv_states = {}

    # 2. pat_repo is the repository of patterns found in this sequence of DFAs.
    # rule_repo is the repository of PRS rules found in this sequence of DFAs.
    pat_repo = PatternRepo()
    rule_repo = IndRuleRepo()

    # 3. patterns_in_dfa is a dictionary of dictionaries. It maps a DFA f and state st to a list of triples.
    # pats_in_dfa[f][st] has an entry (p_ID,position,p_state) in the list if state st in f is part of a
    # pattern with ID p_ID;  More specifically, a set of states, including st, matches pattern with ID p_ID, DFA state
    # st corresponds to the pattern state p_state, which has the given position, one of {"Start", "Middle","End"}, in
    # the pattern.
    patterns_in_dfa = {}

    # 4. new_patterns_in_dfa is a dictionary mapping f to a list [(st,ID)], where f is a DFA,  st is a state in f that
    # is the init state of a pattern found in f, this pattern instance is NOT in the predecessor DFA to f at st, and ID
    # is the ID of this pattern (as assigned in the PatternRepo).  Note that new_patterns_in_dfa[f] is a list of all such
    # new patterns in f
    new_patterns_in_dfa = {}

    # 5. junk_states is a dictionary that maps a DFA f to its junk state
    junk_states = {}

    # Phase 1: Find patterns: iterate over list of DFAs and compare each DFA to its successor DFA to extract patterns
    if verbose:
        print("Phase 1\n\n")
    i = 0
    alphabet = ""
    while i < num_fsms-1:
        f1 = fsms[i]
        f1alpha = f1.alphabet
        for c in f1alpha:
            if c not in alphabet:
                alphabet = alphabet + c
        equiv_states[f1] = {}
        if f1 not in junk_states:  # note we need to record its junk state for later even if DFA is only a single state
            junk_states[f1] = find_junk_state(f1)
        f2 = fsms[i+1]
        junk_states[f2] = find_junk_state(f2)
        # compare the two DFAs.   return a copy of the second one with all the equivalent transitions removed.
        # the returned parameter f2p is a DFA containing only the transitions in f2 not in f1.
        # the returned parameter pattern_heads is a list containing those states in f2p that are the initial states of a
        # new pattern in f2p.
        # this function will also update the data structure equiv_states
        f2p,patterns_heads= remove_equiv_transitions(f1,f2,junk_states[f1],junk_states[f2],equiv_states)
        for st in patterns_heads:
            # for each pattern head (pattern init state) in f2p, create a pattern and add it to the pattern repo.
            # i+1 = DFA number in which this pattern appears
            p = create_pattern_from_DFA(f2p,st,f1,equiv_states,i+1,pat_repo)
            p_ID = pat_repo.insert(p)
            if f2 in new_patterns_in_dfa.keys():  # f2 already has entry in the dictionary
                new_patterns_in_dfa[f2].append((st,p_ID))
            else:
                new_patterns_in_dfa[f2] = [(st,p_ID)]
        i = i + 1

    num_pats_before_filter = len(pat_repo.repo)
    print("Number of patterns found before filtering = " + str(num_pats_before_filter) + "\n")

    # filter out insignificant patterns
    pat_repo.filter(output_filtered_handle, remove_super_patterns=False)

    num_pats_after_filter = len(pat_repo.repo)
    print("Number of patterns after filtering = " + str(num_pats_after_filter) + "\n")

    # Phase 2: Mark where the patterns are found (instantiated) in each DFA
    if verbose:
        print("Phase 2\n\n")
    i = 0
    while i < num_fsms :  # find patterns in all DFAs including last one
        f = fsms[i]
        patterns_in_dfa[f] = {}
        junk_st=junk_states[f]
        mark_patterns_in_dfa(i, f, junk_states[f], pat_repo.repo, patterns_in_dfa)
        if verbose:
            print("\n DFA # " + str(i))
            for st in patterns_in_dfa[f]:
                print("DFA state " + str(st) + " is in the following patterns ")
                for pat_ID,pos,p_state in patterns_in_dfa[f][st]:
                    print("pattern ID = " + str(pat_ID) + ","+ "pattern pos = " + pos + "," + "pattern state = " + str(p_state) + "\n")
        i = i+1

    # Phase 3: Find the PRS rule r that expands a DFA #i to DFA #i+1 by applying r to state st in DFA #i
    # Note that new_patterns_in_dfa may refer to patterns that were subsequently filtered out.  Need to remove them
    if verbose:
        print("Phase 3")
    i = 0
    pattern_IDs = [p.pat_ID for p in pat_repo.repo]   # remaining (unfiltered) pattern IDs
    while i < num_fsms-1:
        f1 = fsms[i]
        f2 = fsms[i+1]
        for s1 in f1.Q:
            if s1 in equiv_states[f1]:
                s2 = equiv_states[f1][s1]
                # s1 in f1 and s2 in f2 are "equivalent" states
                if s1 in patterns_in_dfa[f1]:
                    # list_of_pats_s1 is a list of triples of the form (p_ID,position,p_state) representing a pattern
                    # p_ID that s1 plays the role of p_state in the pattern in the given position
                    list_of_pats_s1 = patterns_in_dfa[f1][s1]
                    if f2 in new_patterns_in_dfa:
                        list_of_new_pats_in_f2 = new_patterns_in_dfa[f2]
                        # new_patterns_f2 are a list of pairs (s2,pat_ID) that are new patterns in f2 where pat_ID is
                        # the pattern mumber of a remaining pattern
                        new_patterns_in_f2 = [x for x in list_of_new_pats_in_f2 if x[0] == s2 and x[1] in pattern_IDs]
                        if len(new_patterns_in_f2) > 0:
                            # we have found a PRS rule(s).  for each pattern p1 that s1 is a part of (list_of_pats_s1)
                            # then for each pattern p2 that begins at s2 we have a rule p1 => p1_LHS p2 p1-RHS.  s1 must
                            # be the join state of p1, and the rule will be of type 2 or type 3 depending on whether p2
                            # is acyclic or cyclic
                            for x in list_of_pats_s1:
                                pnum,pos,jstate = x
                                # jstate is the join state
                                # instead of using name jstate for pattern state, use the numeric state name (the join state
                                # is always given by its numeric number).
                                jstate_num = pat_repo.get_pattern(pnum).state_map[jstate]
                                for y in new_patterns_in_f2:
                                    if pos == "Middle":
                                        LHSpat = pat_repo.get_pattern(pnum)
                                        # setting the join state will also create two new patterns, the LHS and RHS
                                        # patterns comprising this pattern.   It will insert them into the pattern repo
                                        # if they are not already there.
                                        if LHSpat.set_join_state(jstate,i,pat_repo):
                                            # if set_join_state returns False then we have already discovered this pattern
                                            # with a different join state, so ignore it.
                                            # NOTE: note join_state expected to be state name after mapping
                                            # rule_repo.insert_rule(source-pat,target-pat,dfa-num,pat_repo,join-st)
                                            rule_repo.insert_rule(pnum, y[1], i, pat_repo, str(jstate_num))
                                        else:
                                            print("findRules: Found a different joint state for for LHS.pat = " + str(LHSpat.pat_ID) + "\n")
                                            print("Previous joint state = " + str(LHSpat.join_state) + "\n")
                                            print("Trying to set join state to " + str(jstate) + "\n")
                                            print("DFA i+1 = " + str(i+1) + "\n")
                                    else:
                                        print("\nFound pattern starting at pos " + pos + ".  Not added because pos not Middle")
        i = i+1

    if write_output:
        datetime_object = datetime.datetime.now()
        output_handle.write(str(datetime_object) + "\n")
        output_handle.write("Directory is " + str(path_name) + "\n\n")
        # output_handle.write("\'++\' indicates additional meta data\n")
        output_handle.write("Grammar found based upon examining " + str(num_fsms) + " DFAs\n")
        output_handle.write("Pattern Threshold = " + str(PATTERN_THRESHOLD) + "\n")
        # output_handle.write("Rule Threshold = " + str(RULE_THRESHOLD) + "\n") # rule threshold not used
        output_handle.write("Number of patterns found before filtering = " + str(num_pats_before_filter) + "\n")
        output_handle.write("Number of patterns after filtering based upon threshold = " + str(num_pats_after_filter) + "\n\n")
        output_handle.write("** PRS Alphabet = " + alphabet + " **\n")
        output_handle.write("\n\n** PRS patterns ** \n\n")
        # write all patterns
        for p in pat_repo.repo:
            p.write_pattern(output_handle)
            # only write votes for composite patterns??
            # if p.pattern_type == PatternType.Composite:
            output_handle.write("++ Pattern " + str(p.pat_ID) + " has " + str(pat_repo.votes[p.pat_ID]) + " votes\n\n")
        output_handle.write("\n** PRS (CF Grammar) Rules **\n\n")
        # write CFG rules
        rule_repo.write_rules(output_handle,pat_repo)
        # phase 4: find the initial rules (Start CFG rules)
        # try to find patterns beginning at initial state of last FSM.  if none exist, then try its predecessor.  Continue
        # until pattern found or no more FSMs to try
        found_start_rules = False
        j = num_fsms-1 # last fsm
        while not found_start_rules and j >= 0:
            final_dfa = fsms[j]
            if final_dfa.q0 not in patterns_in_dfa[final_dfa].keys():
                j=j-1
            else:
                found_start_rules = True
        if not found_start_rules:
            output_handle.write("Did not find any initial rules\n")
        else:
            start_patterns = patterns_in_dfa[final_dfa][final_dfa.q0]
            # start_patterns has an entry (p_ID,position,p_state) in the list if state q0 in final_dfa is part of a matching
            # pattern;  i.e., a set of states, including q0, matches pattern with ID p_ID.  In this match, DFA state st corresponds to
            # the pattern state p_state, which has the given position, one of {"Start", "Middle","End"}, in the pattern.
            # Hence each tuple represents a pattern at the Start node of the DFA.
            # We need to determine if this pattern is cylic or acylic.  in our current code, both an acyclic version of a pattern
            # matches a cyclic version (***** To DO.  Change this in Pattern code, and in method pattern.matches. ***)
            # For now we just list the pattern
            # output_handle.write("Init Rules:\n")
            did_cyc = False  # did not output yet any cyclic productions for Start symbol
            k = 1
            for pat_ID, pos, p_state in start_patterns:
                if pos == "Start":  # a cyclic pattern will have pos "Start" and "End" so only include if "Start"
                    pat = pat_repo.get_pattern(pat_ID)
                    # check if pattern is cyclic
                    if pat.shape == Shape.Cyclic:
                        if not did_cyc:  # if did not already output cyclic productions for Start symbol
                            output_handle.write("Rule ST-" + str(k) + ":  ")
                            k = k+1
                            output_handle.write("S --> SC\n")
                            output_handle.write("Rule ST-" + str(k) + ":  ")
                            k = k + 1
                            output_handle.write("SC --> SC SC\n")
                            did_cyc = True
                        #  output production for this cyclic pattern
                        output_handle.write("Rule ST-" + str(k) + ":  ")
                        k=k+1
                        output_handle.write("SC --> P" + str(pat_ID) + "\n")
                    else:  # pattern is serial
                        output_handle.write("Rule ST-" + str(k) + ":  ")
                        k = k + 1
                        output_handle.write("S --> P" + str(pat_ID) + "\n")
    output_handle.close()
    output_filtered_handle.close()


