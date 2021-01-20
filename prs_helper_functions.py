from DFA2 import DFA2
import prs_globals
import re

# conv_string_to_transitions takes a string str and finds the first three occurrences of "words" in str (ie., a word consists of chars
# digits, and underscore [a-zA-Z0-9_].
# it requires that the first two occurrences are only digits.   The last one can be any word.  It then returns the triplet
# consisting of these 3 words, the first two converted to integers and the last remaining as a char word.
def conv_string_to_transition(str):
    l = re.findall(r'\w+', str)
    if not l[0].isdigit():
        print("conv_to_triple error:  l[0] = " + l[0]  + "is not integer")
    if not l[1].isdigit():
        print("conv_to_triple error:  l[1] = " + l[1]  + "is not integer")
    return (int(l[0]),int(l[1]),l[2])


#   ******This function NO LONGER USED
# convertStates takes (i) exit1, name of exit state of first pattern, (ii) init2, name of init state of second pattern,
# (iii) lastUsedNum, the last number used as a state name in the first pattern, so states of the second pattern must start
# at lastUsedNum+1, and (iv) transitions
# of the second pattern.  It will return (a) the transitions in the second patterns with states renamed, such that the init state
# of the second pattern is renamed to the name of the exit state of the first pattern, and all the other states have new
# names (numbers) starting from firstAvailNumber.  (b) the new name of the exit state in the second pattern, and (c) the
# new highestState number used in second pattern
def convertStates(exit1,init2,exit2,lastUsedNum,transitions):
    newTransitions = []
    newNames = {}
    newNames[init2] = exit1
    for (s,t,sym) in transitions:
        if s in newNames.keys():
            s2 = newNames[s]
        else:
            lastUsedNum = lastUsedNum + 1
            s2 = lastUsedNum
            newNames[s] = s2
        if t in newNames.keys():
            t2 = newNames[t]
        else:
            lastUsedNum = lastUsedNum + 1
            t2 = lastUsedNum
            newNames[t] = t2
        newTransitions.append((s2,t2,sym))
    newExitName = newNames[exit2]
    return newExitName,newTransitions,lastUsedNum

# makeCompositePattern takes two (basic) patterns, and returns init,exit,transitions,jointState of the new composite pattern
# compositeType indicates if the compositePattern is to be Acyclic or Cyclic
def makeCompositePattern(init1,exit1,trans1,init2,exit2,trans2,compType):
    init = init1
    transitions = []
    highestState = 0
    for (s,t,sym) in trans1:
        if int(s) > highestState:
            highestState = int(s)
        if int(t) > highestState:
            highestState = int(t)
        transitions.append((s,t,sym))
    # now add second pattern transitions to transitions but rename the state names
    newNames = {}
    # set joint state to be the exit state of the first pattern
    joinSt = exit1
    # set the name of the init state of the second pattern to be the same as the exit state of the first pattern
    newNames[init2] = exit1
    if compType == prs_globals.CompositeType.Cyclic:
        newNames[exit2] = init
    for (s, t, sym) in trans2:
        if s in newNames.keys():
            s2 = newNames[s]
        else:
            highestState = highestState + 1
            s2 = highestState
            newNames[s] = s2
        if t in newNames.keys():
            t2 = newNames[t]
        else:
            highestState = highestState + 1
            t2 = highestState
            newNames[t] = t2
        transitions.append((s2, t2, sym))
    exit = newNames[exit2]
    return init,exit,transitions,joinSt


# insertPatterDFA takes params (1) d1, the existing dfa, (2) dName, the name of the new DFA, (3) p, a pattern to insert
# into d1.  p is a tuple (init,exit,trans) where init and exit are the init and exit states of the pattern p, and
# trans are the transitions of p, (4) patternJoinSt, which is the join state of the composite pattern p, (5) dfaJoinSt,
# the join state of d1 where p is to be inserted, (6) isCyclic, which is True if the pattern being inserted is a cycle
# pattern and False otherwise, and (7) lastUsedNum, the highest numbered state in d1
# insertPatterDFA returns (i) d2, the new dfa, (ii) newjoinState, the number of the new join state of the pattern (patternJoinSt)
# after it is inserted into new DFA, and (iii) lastUsedNum, the highest numbered state in d2
def insertPatternDFA(d1,dName,p,patternJoinSt, dfaJoinSt, isCyclic, lastUsedNum):
    # create d2.  ***** We assume that in d1, there is a single final state
    d2 = DFA2(d1.alphabet,d1.q0,d1.F[0],dName)
    # add transitions from d1 to d2.   ***** Note, this includes transitions to sink state
    for q in d1.Q:
        for sym in d1.alphabet:
            qprime = d1.delta[q][sym]
            d2.add_transition_and_states(q,qprime,sym)
    # add transitions from p to d2
    # newNames is a dictionary oldStateName --> newStateName
    newNames = {}
    newNames[p[0]] = dfaJoinSt  # the init state of the pattern is the same as the join state of d1
    for (s,t,sym) in p[2]:
        if s in newNames.keys():
            s2 = newNames[s]
        else:
            lastUsedNum = lastUsedNum + 1
            s2 = lastUsedNum
            newNames[s] = s2
        if t in newNames.keys():
            t2 = newNames[t]
        else:
            lastUsedNum = lastUsedNum + 1
            t2 = lastUsedNum
            newNames[t] = t2
        if s == patternJoinSt:
            newJoinState = s2
        if t == p[1]:  # t is the exit state of the pattern
            newExit = t2
        d2.add_transition_and_states(s2,t2,sym)
    if not isCyclic:  # add connections from exit state to successors of join state
        for symbol in d1.delta[dfaJoinSt]:
            trg = d1.delta[dfaJoinSt][symbol]
            if trg != dfaJoinSt:  # if it is a self loop, then do not connect back to dfaJoinSt itself!
                d2.add_transition_and_states(newExit,trg,symbol)
    # add missing junk transitions
    for q in d2.Q:
        no_trans = [c for c in d2.alphabet if not c in d2.delta[q].keys()]
        for ch in no_trans:
            d2.add_transition_and_states(q, "junk", ch)
    return d2, newJoinState, lastUsedNum

