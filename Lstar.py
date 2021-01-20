from ObservationTable import ObservationTable
import DFA
from time import process_time

def run_lstar(teacher,time_limit,prints_file,emptyseq):
    table = ObservationTable(teacher.alphabet,teacher,prints_file,emptyseq=emptyseq)
    start = process_time()
    original_start = start
    teacher.counterexample_generator.set_time_limit(time_limit,start)
    table.set_time_limit(time_limit,start)

    while True:
        while True:
            while table.find_and_handle_inconsistency():
                pass
            if table.find_and_close_row():
                continue
            else:
                break
        dfa = DFA.DFA(obs_table=table)
        print("obs table refinement took " + str(int(1000*(process_time()-start))/1000.0) ,file=prints_file,flush=True)
        print("overall time since extraction init:",process_time()-original_start,file=prints_file,flush=True)
        counterexample = teacher.equivalence_query(dfa)
        if counterexample == None:
            break
        start = process_time()
        table.add_counterexample(counterexample,teacher.classify_word(counterexample))
    return dfa