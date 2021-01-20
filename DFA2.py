# Build sequence of DFAs for different specific languages

from DFA import DFA


class DFA2(DFA):
    def __init__(self,alpha,init_state,final_state,dfa_name):
        self.alphabet = alpha
        self.Q = []
        self.delta ={}
        self.q0 = init_state
        self.add_state(init_state)
        self.F = [final_state]
        self.add_state(final_state)
        self.name = dfa_name
        self.appendy_alphabet = ""
        self.emptyseq = []

    def add_state(self,q):
        if q not in self.Q:
            self.Q.append(q)
            self.delta[q] = {}

    def add_transition_and_states(self,src,trg,c):
        if src not in self.Q:
            self.add_state(src)
        if trg not in self.Q:
            self.add_state(trg)
        if c not in self.alphabet:
            print("adding char to alphabet")
            self.alphabet = self.alphabet + c
        if c in self.delta.keys():
            print("transition from state " + src + " on symbol " + c + " is already defined\n")
        self.delta[src][c] = trg

    def add_sink_reject(self):
    # create a sink_reject state junk
    # for every state s that does not have a transition on a letter c in the alphabet to a regular state,
    # create a transition (s,junk,c)
        self.add_state("junk")
        for q in self.Q:
            no_trans = [c for c in self.alphabet if not c in self.delta[q].keys()]
            for ch in no_trans:
                self.add_transition_and_states(q,"junk",ch)