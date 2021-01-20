from Quantisations import SVMDecisionTreeQuantisation
from WhiteboxRNNCounterexampleGenerator import WhiteboxRNNCounterexampleGenerator
from time import process_time

class Teacher:
    def __init__(self, network, output_dfas_folder, prints_file, num_dims_initial_split=10,starting_examples=None,\
            token_predictor_samples=-1,token_predictor_cutoff=-1,human_only_use_starters=False):
        if None == starting_examples:
            starting_examples = []
        self.recorded_words = {} # observation table uses this as its T (according to angluin paper terminology)
        self.discretiser = SVMDecisionTreeQuantisation(num_dims_initial_split)
        self.counterexample_generator = WhiteboxRNNCounterexampleGenerator(network,self.discretiser,
            starting_examples,prints_file,
            token_predictor_samples=token_predictor_samples,token_predictor_cutoff=token_predictor_cutoff)
        self.dfas = []
        self.counterexamples_with_times = []
        self.current_ce_count = 0
        self.network = network
        self.alphabet = network.input_alphabet #this is more for intuitive use by lstar (it doesn't need to know there's a network involved)
        self.prints_file = prints_file
        self.output_dfas_folder = output_dfas_folder
        self.output_dfas_count = 0

    def update_words(self,words):
        seen = set(self.recorded_words.keys())
        words = set(words) - seen #need this to avoid answering same thing twice, which may happen a lot now with optimistic querying...
        self.recorded_words.update({w:self.network.classify_word(w) for w in words})

    def classify_word(self, w):
        return self.network.classify_word(w)

    def equivalence_query(self, dfa):
        self.dfas.append(dfa)
        start = process_time()
        print("starting equivalence query for DFA of size " + str(len(dfa.Q)),\
            "(DFA #",self.output_dfas_count,")",file=self.prints_file)
        dfa.draw_nicely(maximum=60,filename=self.output_dfas_folder+"/"+str(self.output_dfas_count))
        self.output_dfas_count += 1
        counterexample,message = self.counterexample_generator.counterexample(dfa)
        counterexample_time = process_time() - start
        print(message,file=self.prints_file)
        print("equivalence checking took: " + str(counterexample_time),file=self.prints_file)
        if not counterexample == None:
            self.counterexamples_with_times.append((counterexample,counterexample_time))
            return counterexample
        return None