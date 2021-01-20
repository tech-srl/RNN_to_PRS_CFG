from prs_globals import PatternType, Shape

class IndRule:
    # source_ID and target_ID are the pattern IDs of the source and target patterns.
    # join_st is the join state of the pattern if composite, otherwise it is "".  DFA_num is the number of the source
    # DFA in which this rule is discovered.
    def __init__(self, source_ID, target_ID, join_st, DFA_num, pat_repo):
        self.rule_ID = 0
        # the source pattern is the pattern p on the LHS of the rule.  Its value is p.pat_ID
        self.source_pat = source_ID
        # the target pattern is the new pattern p' on the RHS of the rule.  Its value is p'.pat_ID
        self.target_pat = target_ID
        tp = pat_repo.get_pattern(self.target_pat)
        # if the source (LHS) pattern appears in more than one rule, and in these rules the target (RHS) patterns are
        # of shape  Cyclic, then they will "share" multiple productions.  Example: if  X -> X-L Y X-R and
        # X -> X-L Z X-R., where Y and Z are cyclic patterns.    Then we need to create productions
        # X ::= X-L X-C X-R, X-C ::= X-C X-C, X-C ::= Y and X-C ::= Z.  In order not to repeat these productions
        # multiple times, we group all the rules with the same source pattern but different Cyclic target patterns together
        # in self.similar_rules.
        self.similar_rules = []
        if tp.shape == Shape.Cyclic:
            self.cyclic = True
        else:
            self.cyclic = False
        # if source pattern is nested in another pattern, then the source pattern has a "join" state
        self.join = join_st
        # DFA_number == i means that DFA i+1 extends DFA i using this rule, and it is the first time in the sequence
        # this rule is used
        self.DFA_number = DFA_num

    def add_similar_rule(self, r_ID):
        if r_ID not in self.similar_rules:
            self.similar_rules.append(r_ID)

    # matches(r) compares this rule with rule r.   If this rule matches rule r, it returns True, self_rule_ID.
    # if this rule matches r except that the join_state's do not match, then it returns False, self_rule_ID.
    # otherwise (it does not match) it returns False,0.
    # however if they have the same source pattern and their target pattern are both Cyclic, then update their
    # attribute similar_rules
    def matches(self,r):
        if self.source_pat == r.source_pat:
            if self.target_pat == r.target_pat and self.join == r.join:  # rules completely match
                return True, self.rule_ID
            elif self.join == r.join: # source patterns and join states match but their targets patterns do not match
                if self.cyclic and r.cyclic:  # both target patterns are Cyclic
                    # we need to add each pattern's list of other Cyclic target patterns the other pattern's ID
                    return False, 99999
            else: # source patterns and target patterns match but join states do not match
                return False, self.rule_ID
        return False, 0   # source patterns do not match


    def write_rule(self, h, pat_repo, indrulerepo, already_written): # h is handle of file already opened for write
        rule_repo = indrulerepo.repo
        # two types of rules, those with target pattern acyclic (rule of type 2) and those with target pattern cyclic (rule of type 3)
        if self.rule_ID in already_written:  # if this rule already written, return empty list (no rules written on this invocation)
            return []
        sp = pat_repo.get_pattern(self.source_pat)
        # Assume LHS pattern sp is CompositeType
        tp = pat_repo.get_pattern(self.target_pat)
        if tp.shape != Shape.Cyclic:
            # RHS pattern tp is acyclic
            h.write("Rule " + str(self.rule_ID) + "A: ")
            h.write(" P" + str(self.source_pat) + "--> P" + str(sp.lhs) + " P" + str(self.target_pat) + \
                " P" + str(sp.rhs) + "\n")
            # h.write("Rule " + str(self.rule_ID) + "B: ")
            # h.write(" P" + str(self.source_pat) + "--> P" + str(sp.lhs) + " P" + str(sp.rhs) + "\n")
            h.write("++ Rule " + str(self.rule_ID) + " first appears when this rule is used to extend DFA " + \
                    str(self.DFA_number) + " to DFA " + str(self.DFA_number +1) + "\n")
            h.write("++ Rule " + str(self.rule_ID) + " has " + str(indrulerepo.votes[self.rule_ID]) + " votes \n\n")
            return [self.rule_ID] # return that just this rule was written in this invocation
        else:  # RHS pattern tp is cyclic
            h.write("Rule " + str(self.rule_ID) + "A: ")
            h.write(" P" + str(self.source_pat) + "-->" + "P" + str(sp.lhs) + " P"+str(self.source_pat)+"C " +\
                    "P" + str(sp.rhs) + "\n")
            h.write("Rule " + str(self.rule_ID) + "B: ")
            h.write(" P" + str(self.source_pat) + "C --> P" + str(self.source_pat) + "C  P" + \
                    str(self.source_pat) + "C\n")
            h.write("Rule " + str(self.rule_ID) + "C: ")
            h.write(" P" + str(self.source_pat) + "C --> P" + str(self.target_pat) + "\n")
            h.write("++ Rule " + str(self.rule_ID) + " first appears when this rule is used to extend DFA " + \
                    str(self.DFA_number) + " to DFA " + str(self.DFA_number +1) + "\n")
            h.write("++ Rule " + str(self.rule_ID) + " has " + str(indrulerepo.votes[self.rule_ID]) + " votes \n\n")
            # write other rules with the same source pattern
            additional_rules = [r for r in rule_repo if r.rule_ID in self.similar_rules]
            new_writes = []
            for r in additional_rules:
                # since this rule has the same source pattern, it shares may of the productions above.  Just add the
                # one missing production
                if r not in already_written:
                    h.write("Rule " + str(r.rule_ID) + "C: ")
                    h.write(" P" + str(self.source_pat) + "C --> P" + str(r.target_pat) + "\n")
                    h.write("++ Rule " + str(r.rule_ID) + " first appears when this rule is used to extend DFA " + \
                            str(r.DFA_number) + " to DFA " + str(r.DFA_number + 1) + "\n")
                    h.write("++ Rule " + str(r.rule_ID) + " has " + str(indrulerepo.votes[r.rule_ID]) + " votes \n\n")
                    new_writes.append(r.rule_ID)
            # Let p be a composite cyclic pattern, with a cycle at its join state.  This means when we separate the
            # composite pattern p into two different patterns, p1' and p2, p1' will have a cycle in its exit state.
            # (When we have a cycle at the join state of a cyclic pattern, then we always associate the cycle with the
            # LHS pattern).   We call this cycle the exit_loop of the pattern.
            # But we do not allow patterns at exit states.  Hence we view p as actually being composed of p1 and p2,
            # where p1 is p1' with the exit_loop cycle removed.   We view the cycle as being generated by the
            # rule p --> p1 p3 p2, where p3 is the pattern expressing that exit_loop.  (Since p is cyclic, this is
            # actually p -> p1 pC p2 and pC -> p3).
            # We record the pattern ID of the exit loop in p1.exit_pattern.
            lhs_pat = pat_repo.get_pattern(sp.lhs)
            if lhs_pat.exit_loop:  # the LHS pattern of this cycle rule contains an exit_loop, so write the additional rule.
                h.write("Rule " + str(self.rule_ID) + "E: ")
                h.write(" P" + str(self.source_pat) + "C --> P" + str(lhs_pat.exit_pattern) + "\n")
            return new_writes

# IndRule_repo class is a repository for rules.
class IndRuleRepo:
    def __init__(self):
        self.nextID = 1
        self.repo = []
        # votes is a dictionary mapping rule IDs to number of votes
        self.votes = {}
        # related_rules is a list of lists.   rule_ID1 and rule_ID2 are in the same list (e.g., equivalence class)
        # if they match except for their join states
        self.related_rules = []

    def update_related_rules(self,existing_ID, new_ID):
        for i in range(len(self.related_rules)):
            if existing_ID in self.related_rules[i]:
                self.related_rules[i].append(new_ID)
                break

    # insert_rule function updates IndRuleRepo.   If the source and target patterns and the rule type and join state
    # matches an existing rule, then it is already in repo so just increment its vote.
    # if it is not in the repo already, then add it and initialize its vote to 1.   If it "almost" matches an existing
    # rule - except the join state is different - then still add it, but also include it in the equivalence class of
    # related_rules.  if the rules do not match because they have different CYCLIC target (RHS) patterns, then it will
    # create a new rule, but it will update the multipe_target_patterns list attribute of each rule to reference the
    # other target pattern.
    #
    # insert_rule(source_ID, target_ID, DFA_number, pat_repo, join_st) where source_ID, target_ID are the
    # patterns in the original and extended DFAs (DFA i and DFA i+1).
    # DFA_number is the number of the source DFA.   join_st is only applicable if the pattern is composite, otherwise
    # it is "".
    def insert_rule(self,source_ID, target_ID, DFA_number, pat_repo, join_st = 'none'):
        ru = IndRule(source_ID, target_ID, join_st,DFA_number,pat_repo)
        similar_rules = []
        if len(self.repo)  == 0:
            ru.rule_ID = self.nextID
            self.nextID = self.nextID + 1
            self.repo.append(ru)
            self.votes[ru.rule_ID] = 1  # has one vote since just created
            return ru.rule_ID
        semi_match_ID = 0
        for r in self.repo:
            does_match,num = r.matches(ru)
            if does_match: # perfect match
                # increment number of votes for this rule since found another matching rule instance
                self.votes[num] = self.votes[num] + 1
                return num
            else: # otherwise does not match
                if num == 99999:
                    similar_rules.append(r)
                elif num != 0:  # then matches except for join_state so remember the rule it semi-matches
                    semi_match_ID = num
        # did not find match
        ru.rule_ID = self.nextID
        self.nextID = self.nextID + 1
        for rr in similar_rules:
            ru.add_similar_rule(rr.rule_ID)
            rr.add_similar_rule(ru.rule_ID)
        self.repo.append(ru)
        self.votes[ru.rule_ID] = 1  # has one vote since just created
        if semi_match_ID != 0:  # then ru matches rule # semi_match_ID except for split state
            self.update_related_rules(semi_match_ID,ru.rule_ID)
        return ru.rule_ID

    def write_rules(self,h,pat_repo): # h is handle of file already opened for write
        rules_already_written = []
        for ru in self.repo:
            newly_written = ru.write_rule(h, pat_repo, self, rules_already_written)
            if newly_written:
                rules_already_written.extend(newly_written)


    def first_rule(self):
        first_DFA = 999999
        first_rule = None
        for ru in self.repo:
            if ru.DFA_number < first_DFA:
                first_rule = ru
                first_DFA = ru.DFA_number
        return first_rule,first_DFA

