import itertools
from config import PATTERN_THRESHOLD, TRANSITION_MINIMUM, verbose
from prs_globals import PatternType, Shape

# For definitions of patterns see paper.   We briefly describe some definitions here.
# A pattern is a restricted incomplete DFA.  (Incomplete means it does not record transitions to the junk state - it
# "meaningful" transitions).    A pattern has a unique init state, and only has one final state, called its exit state.
# if there is a transition entering the init state, then that transition originates from the exit state.   In such a
# case the pattern is called Cyclic.  Otherwise, the pattern is called Acyclic.   The final state of an Acyclic pattern
# must be a sink.   If the DFA is noisy and  a pattern has a exist state containing a cycle, then we treat it as two
# patterns, where the second pattern is Cyclic and is grafted onto the exit state.
#
# On initialization the parm DFA_num is the number of the DFA in which this pattern is first discovered.
# State names supplied will reflect the state names in the DFA where the pattern was first identified.   We want these
# state names to reflect ordered integers starting at 1.  Hence we maintain a state_map, which maps the original name of
# the state with the name (number) we give it.
# ***** TBD: Would be better to assign state names on creation and only use those.
class Pattern:
    def __init__(self,init_state,DFA_num,patType = PatternType.Base,exit_state=None):
        self.alphabet = []
        self.Q = []
        self.delta = {}
        self.state_num = 1
        self.state_map = {}
        self.add_state(init_state)
        self.q0 = init_state
        # self.next_state_ID = 1
        if exit_state != None:
            self.add_state(exit_state)
        self.exit = exit_state  # the Exit (final) state of the pattern
        # pat_ID == 0 is a placeholder.   Real ID will be set by creator
        self.pat_ID = 0
        self.num_transitions = 0
        # Shape is one of Cyclic or Acyclic
        self.shape = None
        # self.pattern_type is either PatternType.Base or PatternType.Composite
        self.pattern_type = patType
        # self.join_state == "" if self.pattern_type = PatternType.Base.   Otherwise, if
        # self.pattern_type == PatternType.Composite, it is the name of the join state of the pattern
        # ******* NOTE: The join_state is the name of the state in state_map, not the name in self.Q
        # here we use "" to mean undefined
        self.join_state = ""
        # if pattern_type == PatternType.Composite, then lhs, rhs are the pattern IDs of the left-hand-side and
        # right-hand-side patterns that make up this pattern.  Otherwise they are None
        self.lhs = None
        self.rhs = None
        self.DFA_discovered = DFA_num
        # exit_loop is True iff pattern shape = Acyclic and there exists at least one cycle beginning and ending at the
        # exit state, and each state on any such cycle is only reachable from the initial state by going through the exit
        # state.  Note:  This is not a legal pattern (since exit state is not a sink), but can arise due to noise and is
        # treated specially:  we remove the cycle beginning and ending at the exit state, and create a new pattern from
        # those transitions.   If this is the case, then self.exit_pattern is the ID of that pattern.  
        self.exit_loop = False
        # if self.exit loop is True, then self.exit_pattern is the ID of the pattern corresponding to that exit_loop
        self.exit_pattern = None

    def add_state(self,q):
        if q in self.Q:
            print("pattern.add_state: DFA state already exists\n")
            return()
        self.Q.append(q)
        self.delta[q] = {}
        self.state_map[q] = self.state_num
        self.state_num = self.state_num +1

    def add_transition(self,src,trg,c):
        if src not in self.Q or trg not in self.Q:
            print("DFA: no such state")
            exit()
        if c not in self.alphabet:
            self.alphabet.append(c)
        if c in self.delta[src].keys():
            print("ERROR: transition from state " + src + " on symbol " + c + " is already defined\n")
        self.delta[src][c] = trg
        self.num_transitions = self.num_transitions + 1

    # check if there is an exit loop (see definition above)
    # if there is, create a new pattern that derives this exit loop
    # remove the transitions that form the new pattern from this pattern.
    # Also, if this pattern is part of a composite pattern, remove these transitions from its parent as well.
    # NOTE: the code to remove the transitions from the parent is dependent on the state names of the parent (composite)
    # pattern and this child pattern being the same!
    def check_for_exit_loop(self,pat_repo,parent=None):
        # exit loop is only relevant for Acyclic patterns
        if self.shape== Shape.Cyclic:
            self.exit_loop = False
            return False
        reachable = []
        queue = [self.exit]
        trans = []
        while queue:
            q = queue.pop()
            for a in self.delta[q]:
                trg = self.delta[q][a]
                # initially self.exit is removed from queue but it was not in reachable.   So it will be added to reachable
                # and to queue.  This means that self.exit will be removed from the queue twice.  To avoid adding
                # (self.exit,trg,a) to trans twice, we need the following conditional.
                if (q,trg,a) not in trans:
                    trans.append((q,trg,a))
                if trg not in reachable:
                    reachable.append(trg)
                    queue.append(trg)
        # reachable are those states reachable from the exit state.
        # trans are the all the transitions traversed from exit state.
        if not reachable:
            self.exit_loop = False
            return False
        # otherwise, there are states reachable from the exit state.  Note that exit state itself is usually in reachable.
        # check that the states in reachable are only reachable from the initial state via exit state.
        queue = [self.q0]
        visited = [self.q0]
        while queue:
            q = queue.pop()
            for a in self.delta[q]:
                trg = self.delta[q][a]
                # skip exit state
                if trg == self.exit:
                    continue
                if trg not in visited:
                    # if state is in reachable, then it can be reached from initial state without going through exit state
                    if trg in reachable:
                        self.exit_loop = False
                        return False
        # if get to this point, there is no path from initial state to a state in reachable without going through the
        # exit state
        self.exit_loop = True
        # create new exit loop pattern.  at the same time remove the states and transitions of this pattern from this
        # pattern (except for exit state which is in both patterns)
        ep = Pattern(self.exit,self.DFA_discovered,PatternType.Base)
        for x in reachable:
            ep.add_state(x)
            if x != self.exit:
                self.Q.remove(x)
        for (src,trg,c) in trans:
            ep.add_transition(src,trg,c)
            del self.delta[src][c]
            if parent != None:
                del parent.delta[src][c]
        # Todo: remove any states that have become unreachable from both this and the parent patterns.
        ep.exit = self.exit
        ep.shape = "Cyclic"
        self.exit_pattern = pat_repo.insert(ep)
        return True


    def set_exit_state(self,q):
       if q not in self.Q:
           print("patterns.py: Error! exit state " + str(q) + " does not exist in pattern states. Adding this state.")
           self.add_state(q)
           self.exit = q
       else:
           self.exit = q

    # when we discover that pattern p has a join state q, we set the join state attribute of p, and we create two new
    # patterns, the "LHS" and the "RHS" of the join state.  we add these into the pattern repository
    # dfa_num is the number of the DFA when this join state is discovered.
    def set_join_state(self,join,dfa_num,pat_repo):
        join_num = self.state_map[join]
        if self.pattern_type == PatternType.Composite:
            if self.join_state == join_num:
                return True   # this pattern already set to Composite type with this join state
            else:
                return False  # this pattern has a different join state which is not legal
        # otherwise discovering that this is a composite pattern for the first time
        self.pattern_type = PatternType.Composite
        self.join_state = join_num
        # get lhs pattern
        sts,trans = self.get_subpattern(self.q0,join)
        lhs = Pattern(self.q0, dfa_num)
        lhs.set_pattern_shape(Shape.Acyclic)
        # we want the state numbers to be in the order they are in the composite parent
        for i in range(1, self.state_num):
            xlist = [q for q in sts if self.state_map[q] == i]
            if xlist:
                if xlist[0] != lhs.q0:
                    lhs.add_state(xlist[0])
        for src,trg,c in trans:
            lhs.add_transition(src,trg,c)
        lhs.set_exit_state(join)
        lhs.check_for_exit_loop(pat_repo,self)
        num = pat_repo.insert(lhs)  # this call returns pat_ID to lhs pattern
        self.lhs = num
        # get rhs pattern
        sts, trans = self.get_subpattern(join, self.exit)
        rhs = Pattern(join, dfa_num)
        rhs.set_pattern_shape(Shape.Acyclic)
        # we want the state numbers to be in the order they are in the composite parent
        for i in range(1,self.state_num):
            xlist = [q for q in sts if self.state_map[q] == i]
            if xlist:
                if xlist[0] != rhs.q0:
                    rhs.add_state(xlist[0])
        for src, trg, c in trans:
            rhs.add_transition(src, trg, c)
        rhs.set_exit_state(self.exit)
        rhs.check_for_exit_loop(pat_repo,self)
        num = pat_repo.insert(rhs)  # this call returns pat_ID to rhs pattern
        self.rhs = num
        return True


    def  set_exit_loop(self):
        self.exit_loop = True

    def set_type_to_Composite(self,join_state):
        # ******* We assume that join_state is the name of the state in state_map, not the name in self.Q!
        self.pattern_type = PatternType.Composite
        # self.join_state = self.state_map[join_state]
        if self.join_state != "" and self.join_state != join_state:
            print("set_type_to_composite: pattern " + str(self.pat_ID)+": trying to set join state to " + str(join_state))
            print("join state already set to " + str(self.join_state))
        else:
            self.join_state = join_state

    # matches(p) returns the ID of this pattern if it matches the pattern p modulo different state names
    # otherwise it returns -1.
    # NOTE:  this will declare that the patterns match as long as this pattern is "embedded" in p; i.e., even if p has
    # more transitions (and states) than this pattern.  However, the match must begin with the init state of this
    # pattern.
    # returns the pattern ID of this pattern (self.pat_ID) if there is a match.  Otherwise returns -1
    def matches(self,p):
        # states = copy(self.Q)
        equiv = [(self.q0,p.q0)]
        stack = [(self.q0,p.q0)]
        while stack:
            st1,st2 = stack.pop()
            for a in self.alphabet:
                if a not in self.delta[st1].keys():
                    continue # transition on this char not defined for st1
                trg1 = self.delta[st1][a]
                if a not in p.delta[st2].keys():
                    return -1  # exists transition in self but not in p
                trg2 = p.delta[st2][a]
                if (trg1,trg2) in equiv:
                    continue
                else:
                    # If one of trg1 or trg2 matches another state return false
                    if [(r1,r2) for (r1,r2) in equiv if r1 == trg1 or r2 == trg2]:
                        return -1
                    else: # new equivalent states
                        equiv.append((trg1,trg2))
                        stack.append((trg1,trg2))
        return self.pat_ID

    # matches_from_state(self, p, qprime) is exactly the same as the method matches, except the match begins from
    # state qprime of p.  Hence matches_from_state(p, p.q0) is the same as matches(p)
    # returns the pattern ID of this pattern (self.pat_ID) if there is a match.  Otherwise returns -1
    def matches_fm_state(self,p, qprime):
        # states = copy(self.Q)
        equiv = [(self.q0,qprime)]
        stack = [(self.q0,qprime)]
        while stack:
            st1,st2 = stack.pop()
            for a in self.alphabet:
                if a not in self.delta[st1].keys():
                    continue # transition on this char not defined for st1
                trg1 = self.delta[st1][a]
                if a not in p.delta[st2].keys():
                    return -1  # exists transition in self but not in p
                trg2 = p.delta[st2][a]
                if (trg1,trg2) in equiv:
                    continue
                else:
                    # If one of trg1 or trg2 matches another state return false
                    if [(r1,r2) for (r1,r2) in equiv if r1 == trg1 or r2 == trg2]:
                        return -1
                    else: # new equivalent states
                        equiv.append((trg1,trg2))
                        stack.append((trg1,trg2))
        return self.pat_ID

    # pat_matches_dfa(self, f, qprime) is similar to the method matches_fm_state, except the match is not to
    # another pattern but to a DFA f, starting at state qprime of p.  It takes an additional parameter, the junk state
    # of f.   Instead of returning the pattern ID if there is a match, it returns equiv, the equivalent states in the
    # pattern and those in f of the form (p-state,f-state) where p-state is a pattern state and f-state is a dfa state.
    # Otherwise returns -1
    def pat_matches_dfa(self,f,qprime,junk):
        # states = copy(self.Q)
        equiv = [(self.q0,qprime)]
        stack = [(self.q0,qprime)]
        while stack:
            st1,st2 = stack.pop()
            for a in self.alphabet:
                if a not in self.delta[st1].keys():
                    continue # transition on this char not defined for st1
                trg1 = self.delta[st1][a]
                trg2 = f.delta[st2][a]
                if trg2 == junk:
                    return -1  # exists transition in self but not in f
                if (trg1,trg2) in equiv:
                    continue
                else:
                    # If one of trg1 or trg2 matches another state return false
                    # unless trg1 = the init state of the pattern.  in this case
                    # it is a cyclic pattern, and it may be that the pattern embedded
                    # in f matches p but it is not cyclic.  do not add it to stack, as already visited
                    if trg1 == self.q0:
                        equiv.append((trg1, trg2))
                    elif [(r1,r2) for (r1,r2) in equiv if r1 == trg1 or r2 == trg2]:
                        if trg1 != self.q0:
                            return -1
                    else: # new equivalent states
                        equiv.append((trg1,trg2))
                        stack.append((trg1,trg2))
        return equiv

    # is_subpattern(self,p) returns true if this pattern is a subpattern of the pattern p.
    # in this context subpattern means the following: there is some state q in p such that if we consider q as
    # the start state of p, then this pattern matches p exactly (starting at q)
    def is_subpattern(self,p):
        for q in p.Q:
            if self.matches_fm_state(p,q) != -1:
                return True
        return False

    # get_subpattern takes a src and trg state of the pattern.  It assumes that all paths starting at the src state
    # pass through the target state.   It will return all the states and transitions on any path from the src to the
    # target state.  It is intended to used when the src state is the init state and the trg state is the join state
    # of the pattern, or the src state is the join state and the trg state is the final state of the pattern.
    # if the join state has a self loop, we want that loop to be in the LHS pattern not RHS pattern.  I.e.,
    # let q0 (f) be init (exit) state, j be join state, and (j,j,c) be a transition.  Then (j,j,c) is returned with the
    # transitions in the call get_subpattern(q0,j) but not in get_subpattern(j,f).
    def get_subpattern(self,s,t):
        states = [s,t]
        trans = []
        queue = [s]
        while len(queue) > 0:
            src = queue.pop(0)
            for c in list(self.delta[src].keys()):
                trg = self.delta[src][c]
                if src !=s or trg !=s:   # don't put cycle from source state to source state into trans
                    trans.append((src, trg, c))
                if trg not in states:  # not going past state t and have not already put this state onto queue
                    states.append(trg)
                    queue.append(trg)
            # put (t,t,c) -- self loops on target state, into trans
        for c in list(self.delta[t].keys()):
            trg = self.delta[t][c]
            if trg == t:
                trans.append(((t,t,c)))
        return states,trans


    def write_pattern(self,h):  # h is file handle of file already opened
        h.write("Pattern P" + str(self.pat_ID) + ". ")
        if self.pattern_type == PatternType.Base:
            h.write("Base Pattern Type.\n")
        else: # PatternType.Composite
            h.write("Composite Pattern Type. LHS pattern is P" + str(self.lhs) + ".  RHS pattern is P" + str(self.rhs) + "\n")
            h.write("Join state = " + str(self.join_state) + "\n")
        if self.shape == Shape.Acyclic:
            h.write("Pattern is acyclic.\n")
        else:
            h.write("Pattern is cyclic\n")
        h.write("Derives regular expressions given by the following DFA:\n")
        h.write("Initial state = " + str(self.state_map[self.q0]) + ", Exit state = " + str(self.state_map[self.exit]) + "\n")
        h.write("States: {")
        h.write(str(self.state_map[self.q0]))
        for q in self.Q:
            if q != self.q0:
                h.write("," + str(self.state_map[q]))
        h.write("}\nTransitions:\n")
        for src in self.Q:
            for c in self.delta[src].keys():
                trg = self.delta[src][c]
                h.write("(" + str(self.state_map[src]) + "," + str(self.state_map[trg]) + "," + c + ")\n")

    def set_pattern_shape(self,ty):
        if ty in Shape:
            self.shape = ty
        else:
            print("set_pattern_shape: " + str(ty) + " is not a recognized pattern shape")


# Pattern_repo class is a repository for patterns.
class PatternRepo:
    def __init__(self):
        self.nextID = 1
        self.repo = []
        # votes is a dictionary mapping pattern IDs to number of votes
        self.votes = {}

    # A pattern p matches a pattern p' if they are the same modulo state names.
    # when a pattern p is inserted into the repo, the following happens:
    # if p matches a pattern rp in the repo, rp.pat_ID is returned (repo remains the same).  the number of votes for rp
    # increases by one.
    # if p is not in the repo, then it is inserted into the the repo and assigned a unique repo pattern ID and assigned
    # one vote
    def insert(self,p):
        if self.repo == []:
            p.pat_ID = self.nextID
            self.nextID = self.nextID + 1
            self.repo.append(p)
            self.votes[p.pat_ID] = 1 # has one vote since just created
            return p.pat_ID
        for rp in self.repo:
            num = rp.matches(p)
            if  num == -1: # match not found
                continue
            else:  # otherwise p is in repo
                self.votes[num] = self.votes[num] + 1 # increment number of votes since found another pattern instance
                return num   # return ID of matching pattern
        # did not find match
        p.pat_ID = self.nextID
        self.nextID = self.nextID + 1
        self.repo.append(p)
        self.votes[p.pat_ID] = 1  # has one vote since just created
        return p.pat_ID

    # the method get_pattern(p_ID) returns the pattern with that ID from the repo if it exists
    # otherwise returns None
    def get_pattern(self,p_ID):
        for pat in self.repo:
            if pat.pat_ID == p_ID:
                return pat
        # else did not find this pattern
        return None

    def remove_pattern(self,p_ID):
        index = -1
        for i in range(len(self.repo)):
            if self.repo[i].pat_ID == p_ID:
                index = i
                print("index = " + str(index))
                break
        if index == -1:
            print("No such pattern ID in repo")
        self.repo.pop(index)
        print("removed pattern at index " + str(index) + "from repo")



    # the method filter removes from repo patterns that are unlikely to be real patterns.  these are patterns that
    # have a single transition or have low number of votes (can set threshold to be any number desired).
    # Optionally, if remove_super_patterns == True  it will also remove every pattern p such that there exists
    # pattern p' and p' is a sub-pattern of p

    def filter(self,h, remove_super_patterns=False): # h is the file handle to write filetered out patterns
        new_repo = [p for p in self.repo if p.num_transitions >= TRANSITION_MINIMUM and (self.votes[p.pat_ID] > PATTERN_THRESHOLD)]
        filtered = [p for p in self.repo if p not in new_repo] # filtered are those patterns filtered out
        self.repo = new_repo
        if remove_super_patterns:
            product = list(itertools.product(self.repo,self.repo))
            while product:
                a,b = product.pop()
                if a.pat_ID != b.pat_ID and a in self.repo and b in self.repo and a.is_subpattern(b):
                    # self.repo.remove(b)
                    self.remove_pattern(b.pat_ID)
        if filtered:
            h.write("The following patterns were filtered out as they did not have at least " + str(PATTERN_THRESHOLD) + " votes\n")
            for p in filtered:
                p.write_pattern(h)
                h.write("\n")
        else:
            h.write("No patterns filtered out")

    # DFA_of_first_pattern will return the DFA x such that a pattern first_pattern in the repo was discovered in DFA x and
    # no other pattern in the repo was discovered in a DFA numbered less than x
    # it will also return first_pattern
    def DFA_of_first_pattern(self):
        first_DFA = 999999
        first_pattern = None
        for p in self.repo:
            if p.DFA_discovered < first_DFA:
                first_pattern = p
                first_DFA = p.DFA_discovered
        return first_pattern,first_DFA

    def write_patterns(self,h):
        # first write Base patterns, then Composite patterns
        for p in self.repo:
            if p.pattern_type == PatternType.Base:
                p.write_pattern(h)
        for p in self.repo:
            if p.pattern_type == PatternType.Composite:
                p.write_pattern(h)

