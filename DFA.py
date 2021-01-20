import graphviz as gv
import functools
from copy import deepcopy, copy
import itertools
import Lstar
from random import randint, shuffle
import random
import string
import os

digraph = functools.partial(gv.Digraph, format='png')
graph = functools.partial(gv.Graph, format='png')

separator = "_"

class DFA:
    def __init__(self,obs_table):
        self.alphabet = obs_table.A #alphabet
        self.appendy_alphabet = obs_table.AwithoutEmpty
        self.Q = [s for s in obs_table.S if s==obs_table.minimum_matching_row(s)] #avoid duplicate states
        self.q0 = obs_table.initial_row()
        self.F = [s for s in self.Q if obs_table.T[s]== 1]
        self._make_transition_function(obs_table)
        self.emptyseq = obs_table.emptyseq

    def _make_transition_function(self,obs_table):
        self.delta = {}
        for s in self.Q:
            self.delta[s] = {}
            for a,append_a in zip(self.alphabet,self.appendy_alphabet):
                self.delta[s][a] = obs_table.minimum_matching_row(s+append_a)

    def classify_word(self,word):
        #assumes word is string with only letters in alphabet
        q = self.q0
        for a in word:
            q = self.delta[q][a]
        return q in self.F

    def find_sink_reject(self):
        for q in self.Q:
            if q in self.F:
                continue
            # if list(set([self.delta[q][a] for a in self.delta[q]]))==[q]:
            if list(set([self.delta[q][a] for a in self.alphabet])) == [q]:
                return q
        return None


    def draw_nicely(self,force=False,maximum=60,filename='img/automaton',
            display=False,hide_sink_reject=True,mark_dict=None): #todo: if two edges are identical except for letter, merge them and note both the letters
        if (not force) and len(self.Q) > maximum:
            return
        if None is mark_dict: mark_dict = {}
        #suspicion: graphviz may be upset by certain sequences, avoid them in nodes
        def state_color(state):
            # return 'green' if state in self.F else 'black'
            for d in mark_dict:
                if state in mark_dict[d]: return d
            return 'black'

        def edge_color(edge):
            two_states = [number_to_state[int(e)] for e in edge]
            for d in mark_dict:
                if len(set(two_states)-set(mark_dict[d])) < 2: # ie one of them is in the mark dict
                    return d # note that the color will be the first mark dict that hits them
            return 'black'



        def shape(state,is_first=False):
            if state == self.q0:
                return 'doubleoctagon' if state in self.F else 'octagon'
            return 'doublecircle' if state in self.F else 'circle'

        state_to_number = {False:0} #false is never a label but gets us started
        number_to_state = [False]
        def state_to_numberstr(state):
            if None is state: return None
            if not state in state_to_number:
                state_to_number[state] = len(number_to_state)
                number_to_state.append(state)
            return str(state_to_number[state])

        def group_edges(sink_reject):
            edges_dict = {}
            for state in self.Q:
                for a in self.alphabet:
                    edge_tuple = (state_to_numberstr(state),state_to_numberstr(self.delta[state][a]))
                    if state_to_numberstr(sink_reject) in edge_tuple:
                        continue
                    # print(str(edge_tuple)+"    "+a)
                    if not edge_tuple in edges_dict:
                        edges_dict[edge_tuple] = a
                    else:
                        edges_dict[edge_tuple] += separator+a
                    # print(str(edge_tuple)+"  =   "+str(edges_dict[edge_tuple]))
            for et in edges_dict:
                edges_dict[et] = clean_line(edges_dict[et], string.ascii_lowercase)
                edges_dict[et] = clean_line(edges_dict[et], string.ascii_uppercase)
                edges_dict[et] = clean_line(edges_dict[et], "0123456789")
                edges_dict[et] = edges_dict[et].replace(separator,",")
            return edges_dict

        g = digraph()
        g.node(state_to_numberstr(self.q0),color=state_color(self.q0),shape=shape(self.q0),label='start')     
        sink_reject = self.find_sink_reject() if hide_sink_reject else None
        states = list(set(self.Q)-{self.q0,sink_reject})

        [g.node(state_to_numberstr(s),color=state_color(s),shape=shape(s),label=str(i+2)) for i,s in enumerate(states)]

        edges_dict = group_edges(sink_reject)
        [g.edge(*e,label=edges_dict[e],color=edge_color(e)) for e in edges_dict]
        img_filename = g.render(filename=filename) # adds a .png or something
        # dmy if display:
            # dmy display(Image(filename=img_filename))
        os.remove(filename)


    def minimal_diverging_suffix(self,state1,state2): #gets series of letters showing the two states are different,
        # i.e., from which one state reaches accepting state and the other reaches rejecting state
        # assumes of course that the states are in the automaton and actually not equivalent
        res = None
        # just use BFS til you reach an accepting state
        # after experiments: attempting to use symmetric difference on copies with s1,s2 as the starting state, or even
        # just make and minimise copies of this automaton starting from s1 and s2 before starting the BFS,
        # is slower than this basic BFS, so don't
        seen_states = set()
        new_states = {(self.emptyseq,(state1,state2))}
        while len(new_states) > 0:
            prefix,state_pair = new_states.pop()
            s1,s2 = state_pair
            if len([q for q in [s1,s2] if q in self.F])== 1: # intersection of self.F and [s1,s2] is exactly one state,
                # meaning s1 and s2 are classified differently
                res = prefix
                break
            seen_states.add(state_pair)
            for a,append_a in zip(self.alphabet,self.appendy_alphabet):
                next_state_pair = (self.delta[s1][a],self.delta[s2][a])
                next_tuple = (prefix+append_a,next_state_pair)
                if not next_tuple in new_states and not next_state_pair in seen_states:
                    new_states.add(next_tuple)
        return res

def clean_line(line,group): # function for neatening up from something like 012acd3b into something like 0-3,a-d
    line = line.split(separator)
    line = sorted(line) + ["END"]
    in_sequence= False
    last_a = ""
    clean = line[0]
    if line[0] in group:
        in_sequence = True
        first_a = line[0]
        last_a = line[0]
    for a in line[1:]:
        if in_sequence:
            if a in group and (ord(a)-ord(last_a))==1: #continue sequence
                last_a = a
            else: #break sequence
                #finish sequence that was
                if (ord(last_a)-ord(first_a))>1:
                    clean += ("-" + last_a)
                elif not last_a == first_a:
                    clean += (separator + last_a)
                #else: last_a==first_a -- nothing to add
                in_sequence = False
                #check if there is a new one
                if a in group:
                    first_a = a
                    last_a = a
                    in_sequence = True
                if not a=="END":
                    clean += (separator + a)
        else:
            if a in group: #start sequence
                first_a = a
                last_a = a
                in_sequence = True
            if not a=="END":
                clean += (separator+a)
    return clean
