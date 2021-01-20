import sys, getopt
import random
from create_PRSspec_from_input import createPRSspec
from DFA2 import DFA2
from prs_helper_functions import insertPatternDFA
from Helper_Functions import overwrite_file
from prs_globals import CompositeType, PatternType

DFA_MAX = 10

# createDFAsFromPRS takes a description of a Pattern rule set (PRS) and produces a sequence of DFAs
# constructed according the spec. The input spec format is given in create_PRSspec_from_input

#1 A patternType is "Base" or "Composite"
#2 A basePattern is a tuple (PatternType.Base,init,exit,transitions)
#3 A compositePattern is a tuple (PatternType.Composite,init,exit,transitions,jointState,firstPatNum,secondPatNum,CompositeType)
#    where firstPatNum (secondPatNum) refers to the first (second) pattern in the composite, Compositeype is Acyclic or Cyclic
# Patterns is a dictionary mapping compositePattern numbers to composition pattern.  Patterns[patnum] = the actual
#    base pattern or composite pattern with this number
#4 A startRule is a pattern number.
#5 startRules is list of the form: [patternNum].  Currently we assume it is a singleton, and patternNum refers to a composite pattern
#6 An insertRule is of the form: (LHSNum,firstPatNum,insertedPatNum,secondPatNum) where LHSnum refers to the the LHS
#  composite pattern, and firstPatNum,insertedPatNum,secondPatNum refer to the three patterns on the RHS of the rule
#7 insertRules is a dictionary mapping ruleNumbers (each rule is assigned a number) to an insert rule
#8 A PRSspec = (alpha, Patterns, startRules, insertRules)

if __name__ == '__main__':
    # -i <input_spec> is name of the PRS file.
    # -o <output_spec> is the name of the DFA spec file that gets generated.   By default, it will will <input_spec>"-DFASpec"
    # -d gives path of where <input_spec> is found and where the DFA spec is to be written.  By default it is current dir
    try:
        opts, remainder = getopt.getopt(sys.argv[1:], "i:o:d:n:v")
    except getopt.GetoptError:
        print("getopt error")
        sys.exit(2)
    input_file_defined = False
    output_file_defined = False
    dir_defined = False
    DFA_number_defined = False
    for opt,arg in opts:
        if opt == '-i':
            input_file = arg
            input_file_defined = True
        elif opt == '-o':
            output_file = arg
            ouput_file_defined = True
        elif opt == '-d':
            dir  = arg
            dir_defined = True
        elif opt == '-n':
            DFA_number  = arg
            DFA_number_defined = True
        elif opt == '-v':
            verbose = True
        else:
            print("Usage: create_dfas_from_spec.py [-i <input_file_name>] [-o <output_file_name>] [-d <path>]")
            sys.exit()
    if not input_file_defined:
        print("create_dfas_from_spec.py. <input_file_name> required")
    if not output_file_defined:
        output_file = input_file + "-DFASpec"
    if not dir_defined:
        dir = "."
    if not DFA_number_defined:
        DFA_number_defined = DFA_MAX
    fd = open(dir + "/" + input_file,"r")
    # Step 1: parse the input and create a PRSspec
    # prs = (alpha,patternSpecs, basePatternSpecs, compositePatternsSpecs, startRules, insertRules)
    prs = createPRSspec(fd)
    #prs = (alpha, Patterns, startRules, insertRules)
    fd.close()


    # *** following is testing code.  can be removed
    (alpha, Patterns, startRules, insertRules) = prs
    print("alpha = " + alpha)
    print("Patterns = ")
    for p in Patterns.keys():
        patt = Patterns[p]
        #if p[0] == create_PRSspec_from_input.PatternType.Base:
        ##    print("basePattern num " + str(p))
        #    print("init state, exit state =" + str(p[1]) + "," + str(p[2]))
        #    for (s,t,sym)
        print(p)
            # 2 A basePattern is a tuple (PatternType.Base,init,exit,transitions)
            # 3 A compositePattern is a tuple (PatternType.Composite,init,exit,transitions,jointState,firstPatNum,secondPatNum)
    print("startRules = ")
    for s in startRules:
        print(s)
    print("insertRules = ")
    for r in insertRules.keys():
        print(insertRules[r])
    # *** end testing code

    # 9. A DFA = (alpha, numStates, initState, finalStates, transitions)
    # where numStates is an <integer> that specifies the number of states, numbered from 1 until <integer>
    # 10. comp2rule maps composite patterns to rules.  comp2rule[patnum] = [r1,r2,...]  iff ri the composite pattern with
    # pattern number patnum is the LHS of rules r1,r2,...
    # 11. states2rules maps states in a DFA to rules that can be applied to that state.   state2rules(s) = [r1,r2,...] iff
    # the state s is the join state of a composite pattern instance p, and p is the LHS of rules r1,r2,... and these rules
    # have NOT been applied to state s yet.

    # Step: 2.  Create initial DFA
    dfas = []
    startPatnum = startRules[0]  # remove an init pattern num from startRules.  assume it is composite pattern
    tup = Patterns[startPatnum]
    # tup should be of form:  pt,init,exit,transitions,joinState,firstPatNum,secondPatNum,ct
    # where pt =
    if tup[0] == PatternType.Base:
        print("Error: initial pattern must be composite")
        sys.exit()
    else:
        pt, init, exit, transitions, joinState, firstPatNum, secondPatNum, ct = tup
    # find highestState number used in this pattern
    highestState = 0
    for (s,t,sym) in transitions:
        if int(s) > highestState:
            highestState = int(s)
        if int(t) > highestState:
            highestState = int(t)
    # create initial DFA
    DFAnum = 1
    DFAname = "DFA" + str(DFAnum)
    dfa = DFA2(alpha, init, exit, DFAname)
    for x in transitions:
        dfa.add_transition_and_states(x[0],x[1],x[2])
    # to use DFA.draw_nicely must have sink state and sink transitions
    dfa.add_sink_reject()
    dfas.append(dfa)
    dfa.draw_nicely(maximum=60, filename=dir + "/" + dfa.name)

    # Step 3.  Initialize comp2rules, state2rules
    comp2rules = {}
    for r in insertRules.keys():
        LHS = insertRules[r][0]
        if LHS not in comp2rules.keys():
            comp2rules[LHS] = [r]
        else:
            comp2rules[LHS].append(r)
    # exit1 = join state of composite pattern startPatnum of DFA1
    rules = comp2rules[startPatnum].copy()   # we will be updating rules so want a separate copy
    state2rules = {joinState:rules} # state2rules[joinState] = rules

    # step 4.  create the rest of the DFAs.
    while DFAnum < DFA_number_defined:
            DFAnum = DFAnum + 1
            DFAname = "DFA" + str(DFAnum)
            # need to pick a join state to update
            joinStates = list(state2rules.keys())
            if len(joinStates) == 0:
                print("create_dfas_from_PRS: Error! No more states to expand")
                sys.exit()
            temp = len(joinStates)
            rand = random.randint(1,len(joinStates))
            joinSt = joinStates[rand-1]  # since list starts from index 0
            # pick a rule to use to update the DFA at that joinState
            ruleLi = state2rules[joinSt]
            rand2 = random.randint(1,len(ruleLi))
            rule2apply = ruleLi[rand2-1]
            del ruleLi[rand2-1]  # since we will use this rule, cannot use it again
            if not ruleLi:  # if now empty
                del state2rules[joinSt]
            else:
                state2rules[joinSt] = ruleLi # update with list minus rule being used
            # find the pattern to be inserted.  It is the the middle pattern of the RHS of the rule.  I.e., insertPatNum
            ruleI = insertRules[rule2apply]
            patternNum = ruleI[2]
            (pt,initc,exitc,transitionsc,joinStatec,firstPatNum,secondPatNum,ct) = Patterns[patternNum]  # assume this is composite pattern
            dfaOld = dfa
            # insertPatterDFA takes params dfaOld, the existing dfa, DFAname, the name of the new DFA, patternToInsert,
            # a tuple (init,exit,trans) where is the init and exit states of the pattern and transitions in the pattern,
            # joinStatec, which is the join state of the composite pattern, and highest state, the highest numbered state
            # name in the old dfa.
            # insertPatterDFA returns d2, newjoinState, highest state where d2 is new dfa, newjoinState is the number of
            # the new join state of the pattern (joinStatec) after it is inserted into new DFA, and highestState whic
            # is the highest state number in new DFA
            patternToInsert = (initc,exitc,transitionsc)
            if ct == CompositeType.Cyclic:
                isCyclic = True
            else:
                isCyclic = False

            dfa, newjoin, highestState = insertPatternDFA(dfaOld,DFAname,patternToInsert,joinStatec,joinSt,isCyclic,highestState)
            # d1, dName, p, patternJoinSt, dfaJoinSt, lastUsedNum
            # update state2rules so that the joinSt of pattern patternNum inserted into the new DFA can also become
            # available for expansion
            rs = comp2rules[patternNum]
            state2rules[newjoin] = rs.copy()
            dfas.append(dfa)
            dfa.draw_nicely(maximum=60, filename=dir + "/" + dfa.name)
    overwrite_file(dfas,dir + "/all_dfas")












