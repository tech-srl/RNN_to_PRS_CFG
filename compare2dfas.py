import copy
from DFA import DFA
from patterns import Pattern, PatternRepo
from prs_globals import Shape
from config import *

# find junk state of DFA f.  A junk state is a sink state that is not final.  Assume that DFA is minimal and there
# is at most one junk state
# NOTE: can use DFA.find_sink_reject(self) instead of this function
def find_junk_state(f):
    # initialize unmarked to be all states - init state.   put init state onto stack
    for q in f.Q:
        # skip final states
        if q in f.F:
            continue
        is_junk = True
        for a in f.delta[q].keys():  # f.alphabet:
            if f.delta[q][a] != q:
                is_junk = False
                continue
        if is_junk:
            return q
    return None

# traverse f1 and f2.  return f2prime = f2 containing only those transitions that are not in f1.
# (Note: f2prime will still have all states as in f2).  also return the start state of each pattern found.
# also update dictionary equiv_states such that for each state s1 in f1 that has an equivalent state
# s2 in f2, equiv_states[f][s1] = s2
# Note: if f2 does not follow the PRS framework by adding a pattern to a join state but also deletes transitions from f1,
# then the equivalent state calculation can be misleading.   See comments below in procedure create_pattern_from_DFA
def remove_equiv_transitions(f1,f2,junk1,junk2,equiv_states):
    f2prime = copy.deepcopy(f2)
    # remove all transitions going to junk2 state of f2prime and remove junk2 state itself
    # some DFAs may not have junk state
    if junk2 is not None:
        for st in f2prime.Q:
            for a in f2prime.alphabet:
                # we assume that if there exists junk state, then transition is defined on every character
                trg = f2prime.delta[st][a]
                if trg == junk2:
                    del f2prime.delta[st][a]
        f2prime.Q.remove(junk2)
    # traverse f1 and f2prime in parallel and remove matching (equivalent) transitions from f2prime
    stack = []
    pattern_heads = []
    stack.append((f1.q0,f2prime.q0))
    visited = [f2prime.q0]
    while stack:
        q1,q2 = stack.pop()
        equiv_states[f1][q1] = q2
        for a in f2prime.alphabet:
            # we assume that, in f1, transitions on all characters are defined
            q1target = f1.delta[q1][a]
            if a not in f2prime.delta[q2].keys():   # this transition already deleted
                continue
            q2target = f2prime.delta[q2][a]
            if q1target != junk1:
                # q1target and q2target are "equivalent" states so delete transition (q2,a,q2target) in f2prime
                del f2prime.delta[q2][a]
                # if we have not explored q2target yet then add it to stack and record that we have put it
                # on stack by adding it to visited
                if not q2target in visited:
                    stack.append((q1target, q2target))
                    visited.append(q2target)
            else:
                # q1target goes to junk state and q2target does not so this is start of pattern
                if q2 not in pattern_heads:
                    pattern_heads.append(q2)
    return f2prime, pattern_heads

# create_pattern takes a DFA f which was returned from removed_equiv_transitions and a state, head, in f.
# f contains only transitions that were in f but not in DFA f_prev, where f_prev is the predecessor DFA of f.
# (f_prev is DFA i and f is DFA i+1).   CreatePattern will return a pattern for these new transitions starting at
# head.  it will not include the junk state of f in the pattern if it exists.
# other paramaters include eq_states (see equiv_states defn in findRules.py) that maps states in
# f_prev to equivalent states in f, and takes the DFA number, dfa_num, in which the pattern is discovered (i.e., i+1),
# and the pattern repository where the patterns are stored.
# 1. Create the pattern p to contain all the states and transitions reachable from head by traversing the DFA f starting
# at head.
# 2. During this traversal, record endpoints.  An endpoint is a state of f (p) that is a state in the pattern but also
# has an equivalent state in f_prev.   Hence, assuming the theoretical framework is followed, it will be a sink.   I.e.,
# All of its incoming edges are from new pattern states, and all of its outgoing edges have been deleted as they were to
# states that existed in the prior DFA.
# 3. Use algorithm given in paper to determine the exit state of the pattern.  As per the paper, there are 3 cases to
# consider.  See paper for explanation.
# 4. Due to noise in the DFAs genearated from the Neural Net, there are many assumptions that may not be satisfied;
# there may not exist any sink states (the exit state contains a loop) and/or there are not
# connecting transitions; there may not be equivalent states in the previous DFA, etc.
def create_pattern_from_DFA(f,head,f_prev,eq_states,dfa_num,pat_repo):
    # Create pattern p to have the states and transitions reachable from head.
    p = Pattern(head,dfa_num)
    p.set_pattern_shape(Shape.Acyclic)  # by default it is acyclic.  will be changed below if discovered to be otherwise
    todo = [head]
    endpoints = []
    # potential_endpoints = []
    while todo:
        src = todo.pop()
        alpha = f.delta[src].keys()
        if not alpha:   # then src is a sink and must be an endpoint
            endpoints.append(src)
        for a in alpha:
            trg = f.delta[src][a]
            if trg not in p.Q:  # have not yet seen state next
                p.add_state(trg)
                todo.append(trg)
            else:        # this pattern contains cycle
                if trg == head:  # cycle is back to src so type is "Cyclic", and init state = exit state
                    p.set_pattern_shape(Shape.Cyclic)
                    p.set_exit_state(trg)
            p.add_transition(src,trg,a)
            # if not endpoints:
                # equiv_states = [x for x in f_prev.Q if x in eq_states[f_prev] and eq_states[f_prev][x] == src]
                # if equiv_states:
                #     potential_endpoints.append(equiv_states[0])
    # If this was determined to be a Cyclic pattern we are done defining the pattern.  We have already found the exit
    # state (= head).  If this is an Acylic pattern, then we have to determine the exit state.
    if p.shape == Shape.Acyclic:
        # preds are the predecessor states to the endpoint states in p.
        # Use p.Q in expression below and not f.Q because there can be cases when a predecessor state in f does not exist
        # in p!
        preds= [q for q in p.Q if a in p.delta[q].keys() and p.delta[q][a] in endpoints]
        # f_prev_endpoints are the the states in the previous DFA corresponding the to the endpoints in this DFA
        f_prev_endpts = [x for x in f_prev.Q if x in eq_states[f_prev] and eq_states[f_prev][x] in endpoints]
        # We now deal with the "normal" case that follows the theoretical framework, so that pred and f_prev_endpts (and
        # endpoints) are all non-empty.
        case2 = False
        case3 = False
        if preds and f_prev_endpts:
            case2 = True
            # qx is the proposed exit state - the single predecessor to the endpoints (case 2 of the paper).
            qx = preds[0]
            # Now determine if case 2 of the paper is correct; every transition into an endpoint in f is a "connecting"
            # transition from the single state qx.  If this is the case, then
            # qx is the exit state, and the transitions emanating from it are not part of the pattern.
            for a in f.delta[qx].keys():
                if f.delta[qx][a] in endpoints:
                    # head_prev_list is a list containing the single state in the previous DFA that corresponds to the head
                    # state in this DFA
                    head_prev_list = [x for x in eq_states[f_prev] if head == eq_states[f_prev][x]]
                    if head_prev_list:
                        head_prev = head_prev_list[0]
                        # check that every transition on a symbol "a" that emanates from the proposed exit state qx to an
                        # endpoint is also a transition from head_prev to the corresponding endpoint state in the previous
                        # DFA.  If not, case2 is False.
                        if a not in f_prev.delta[head_prev] or not (f_prev.delta[head_prev][a] in f_prev_endpts):
                            case2 = False
                    # in the theoretical framework, head_prev_list is never empty- it is the join state.  However,
                    # in a noisy DFA it can happen that head_prev_list is empty.   Consider DFA1 that has two states,
                    # sa and sb.  There is a transition
                    # from sa to sb on "(" and back to "sa" on ")".   In DFA2, there are 4 states.  There is a transition
                    # from s1 to s2 on "(" and from s2 to s3 on ")".   There is a transition from s1 to s4 on "{" and
                    # from s4 to s3 on "}".   The algorithm first finds that sa is equivalent to s1, but later finds
                    # that sa is equivalent to s4.  This latter value overwrites the first, so it is recorded that sa is
                    # equivalent to s4.  This means that head = s1 in DFA1 has no equivalent state in DFA2, and hence
                    # head_prev_list is empty!
                    else:
                        case2 = False
            if case2:
                p.set_exit_state(qx)
                # syms = [a for a in f.delta[qx] and f.delta[qx][a] in endpoints]   this does not work!
                syms = []
                for c in f.delta[qx]:
                    trg = f.delta[qx][c]
                    if trg in endpoints:
                        syms.append(c)
                for d in syms:
                    if d in p.delta[qx]:  # this line should not be necessary
                        del p.delta[qx][d]
            # otherwise case 3 of the paper must be applicable; i.e., endpoints must be a singleton set.
            # set the exit state to that state.
            else:
                qx = endpoints[0]
                p.set_exit_state(qx)
                case3 = True
        # In the theoretical framework, we are done finding the exit state.  However, because of noise, it may be that
        # neither case2 nor case3 was satisfied.  E.g., might be that any of endpoints, preds, f_prev_endpoints, or
        # head_prev were empty or one of the conditions was not satisfied.  Hence we now need to use some heuristics to
        # determine the best choise for the exit state.
        if not case2 and not case3:
            if endpoints:
                qx = endpoints[0]
                p.set_exit_state(qx)
            else:
                # since there is no endpoint, there must not be a sink state.   But since the pattern in Acyclic, there
                # must be a cycle not involving the init state.   Traverse from the head state until a state v s.t.
                # there is a cycle starting and terminating at v.
                # TODO:  if there is a cycle (perhaps a self cycle) in the middle of the pattern, then that state will
                # be considered the exit state even though there are additional states to traverse.   I tried a couple
                # of other heuristics that did not work well.  Needs more work.  For L15 of paper, it does not find the
                # appropriate exit state.
                found = []
                queue = [p.q0]
                nocycle = True
                while queue and nocycle:
                    q = queue.pop()
                    for a in p.delta[q]:
                        trg = p.delta[q][a]
                        if trg in found:
                            # we have found cycle starting and ending at trg
                            if q == trg: # self loop at exit state
                                p.set_exit_state(trg)
                                nocycle = False
                            else: # q is state with back loop to trg
                                p.set_exit_state(q)
                                nocycle = False
                            break
                        else:
                            found.append(trg)
                            queue.append(trg)
                if nocycle:
                    print("compare2dfas.py: Error, nocycle = True\n")
    # Remove unreachable states in the pattern.
    wl = [p.q0]
    marked = [p.q0]
    while wl:
        st = wl.pop()
        for a in p.delta[st]:
            trg = p.delta[st][a]
            if trg not in marked:
                marked.append(trg)
                wl.append(trg)
    for v in p.Q:
        if v not in marked:
            p.Q.remove(v)
    # check if there is an exit loop.
    if p.shape == Shape.Acyclic:
        p.check_for_exit_loop(pat_repo)
    return p


# starts_pattern(f,st,junk_state,p,pats_in_dfa):
# this function will determine if state st in dfa f is the beginning of pattern p.
# if yes, then it will update the dictionary so that for each state s in dfa f that is part of the pattern,
# pats_in_dfa[f][s] has an entry (p_ID,position,p_state), where p_ID is the ID of the matching pattern, p_state
# is the state of the pattern that matches s, and position is one of {"Start", "Middle","End"}
# (see patterns_in_dfa dictionary description).
# junk_state is the "junk" state of f.
# NOTE: we previously used a more loose definition that would allow sub-patterns of a pattern in the DFA to match, as
# long as it was "similar" enough.   Need to revisit if this is beneficial for noisy DFAs.
def starts_pattern(f,st,junk_state,p,pats_in_dfa):
    if st == junk_state:
        return False
    # p.pat_matches_dfa will return -1 if there is not match of the pattern to DFA f starting at state st.
    # otherwise it returns a set of matching pairs, (p-state,f-state), where p-state is the state of p and f-state is the
    # state of f.
    matches = p.pat_matches_dfa(f,st,junk_state)
    if matches == -1:
        return False
    # if reach here then st matches beginning of pattern
    # need to insert (p.pat_ID,"Start",p.qo) into pats_in_dfa[f][st] if not already there
    if st in pats_in_dfa[f]:  # list pats_in_dfa[f][st] already exists
        if (p.pat_ID, "Start", p.q0) not in pats_in_dfa[f][st]:
            pats_in_dfa[f][st].append((p.pat_ID, "Start", p.q0))
    else:   # list pats_in_dfa[f][st] does not yet exists
        pats_in_dfa[f][st] = [(p.pat_ID, "Start", p.q0)]
    for r,q in matches:
        if r == p.q0:  # already added above so skip this case
            continue
        if r == p.exit:
            pos = "End"
        else:
            pos = "Middle"
        if q in pats_in_dfa[f]:    # list pats_in_dfa[f][q] already exists
            if (p.pat_ID, pos, r) not in pats_in_dfa[f][q]:
                pats_in_dfa[f][q].append((p.pat_ID, pos, r))
        else:   # list pats_in_dfa[f][r] does not yet exists
            pats_in_dfa[f][q] = [(p.pat_ID,pos,r)]
    return True

# mark_patterns_in_dfa(i,f, junk, pat_list, dfa_state_to_pat):
# i is the sequence number for DFA f
# f is DFA, junk is the junk state of the DFA, pat_list is a list of patterns, and dfa_state_to_pat is a dictionary of
# dictionaries. It maps a DFA f and state st to a list of triples.
# pats_in_dfa[f][st] has an entry (p_ID,position,p_state) in the list if state st in f is part of a matching pattern;
#
# for each state st in DFA f, for each pattern p in pat_list, this function will determine if st is the beginning of
# the pattern p;  i.e., p is "embedded" in f starting at the state st.  If so, it will
# update the dictionary dfa_state_to_pat to record all the states of f that participate in this pattern, and for each
# such state, whether it is the start of the pattern, a state in the middle of the pattern, or a final state of the
# pattern.   The exact nature of dfa_state_to_pat dictionary is described by patterns_in_dfa in "findRules.py"
def mark_patterns_in_dfa(i, f, junk, pat_list,dfa_state_to_pat):
    for pat in pat_list:
        for q in f.Q:
            if q != junk:
                res = starts_pattern(f, q, junk, pat, dfa_state_to_pat)
                # adding the following line because of the following:
                # when a pattern pat is discovered in DFA # x, it records x as the first DFA it appears in.
                # however, it may be that a DFA preceding x already contain pat but was not recorded as the first
                # DFA this pattern appears in because the pattern was actually contained
                # in the very first DFA.  (I.e., a pattern is only discovered when it extends a previous DFA).
                # Hence in this phase when we look at all DFAs and mark patterns, if we find a pattern in DFA # w,
                # if w < x, we update and record w as the first DFA this pattern appears in
                if res:  # if pat is found in this DFA
                    if pat.DFA_discovered > i:
                        pat.DFA_discovered = i







