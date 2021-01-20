from enum import Enum
import re
import sys
from prs_helper_functions import conv_string_to_transition, makeCompositePattern
import prs_globals

# createPRSspec takes a description of a Pattern rule set (PRS) and produces a PRSspec.
# <PRS spec input format> ::=
    # Alpha: <string>    /* <string> are the characters of the alphabet */
    # <pattern_spec>*   /* 0 or more pattern specs */
    # <rule_spec>+ /* 1 or more rule specs */
    # NumToGenerate: <integer>  /* number of DFAs to generate from the spec */
    # EndPRS
# <pattern_spec> ::= <base_pattern> | <composite_patter>
# <base_pattern> ::=
    # BasePattern: <patternNum>  /* number of this pattern.  Each pattern must have a unique number */
    # InitState: <name>
    # ExitState: <name>
    # Transitions: [ {(<src_state>,<trg_state>,<character>) {,}}* ]   /* I.e., list of triplets, (src,trg,c) separated by commas. */
        # <src_state> = <trg_state> = <integer>. <character> is in Alpha.
        # If multiple lines, each line will be same format (leading keyword "Transitions")
    # END_BasePattern
# <composite_pattern> ::=
    # CompositePattern: patternNum,pattern1num,pattern2num  /* pattern1num,pattern2num are the first and second patterns in composite pattern */
# <rule_spec> ::=
    # StartRule:  <pattern-num>  [,<pattern_num>]*  # ******** initially assume a single pattern
    # InsertRule: <LHSpatternNum> -> <RHSpattern1> , <RHSpattern3>, <RHSpattern2>
# END_PRS

#1 A patternType is "Base" or "Composite"
#2 A basePattern is a tuple (PatternType.Base,init,exit,transitions)
#3 A compositePattern is a tuple (PatternType.Composite,init,exit,transitions,jointState,firstPatNum,secondPatNum,CompositeType)
#    where firstPatNum (secondPatNum) refers to the first (second) pattern in the composite, Compositeype is Acyclic or Cyclic
# Patterns is a dictionary mapping compositePattern numbers to composition pattern.  Patterns[patnum] = the actual
#    basePattern or composite pattern with this number
#4 A startRule is a pattern number.
#5 startRules is list of the form: [patternNum].  Currently we assume it is a singleton, and patternNum refers to a composite pattern
#6 An insertRule is of the form: (LHSNum,firstPatNum,insertedPatNum,secondPatNum) where LHSnum refers to the the LHS
#  composite pattern, and firstPatNum,insertedPatNum,secondPatNum refer to the three patterns on the RHS of the rule
#7 insertRules is a dictionary mapping ruleNumbers (each rule is assigned a number) to an insert rule


def parse_line(line):
    global ruleNum
    # assume keyword is one of following
    # Alpha:, BasePattern:, InitState:, ExitState:, Transitions:, CompositePattern:, StartRule:, InsertRule:
    ind = line.find("#")
    if ind != -1:  # there is comment
        line = line[:ind]   #strip comment
    line = line.strip()   # removes leading and trailing whitespace
    if line == "" or line == "\n": # nothing else on line
        return ("Comment", line)
    ind = line.find(":")  # find first :
    if ind == -1:
        if line.find("END_PRS") != -1 or line.find("End_PRS") != -1:
            return ("End",0)
        elif line.find("END_BASEPATTERN") != -1 or line.find("END_BasePattern") != -1 or line.find("End_BASEPATTERN") != -1 or line.find("End_BasePattern") != -1:
            return ("END_BASEPATTERN", 0)
        elif line.find("END_COMPOSITEPATTERN") != -1 or line.find("END_CompositePattern") != -1 or line.find("End_COMPOSITEPATTERN") != -1 or line.find("End_CompositePattern") != -1:
            return ("END_COMPOSITEPATTERN", 0)
        else:
            return ("Error",line)
    r = line[:ind] # r is string from start of line until ":"
    r = r.rstrip() # strip trailing whitespace
    rest = line[ind+1:]  # rest is string in line after ":"
    rest = rest.strip() # strip leading and trailing whitespace
    rest = rest.rstrip('\r\n')  # strip trailing return or newline
    #if r == "PRS_name":
    #    name = rest # re.findall('\d+', rest) # return integer in line
    #    return ("name",name)
    if (r ==  "Alpha"):
        # rest is alphabet
        return ("Alpha", rest)
    elif (r == "BasePattern"):
        # rest is patternNum
        return("BasePattern",rest)
    elif (r ==  "InitState"):
        # rest is patternNum
        return ("Init", rest)
    # elif (r ==  "Num_states"):
    #    num = rest # re.findall('\d+', rest) # return integer in line
    #    return ("States", num)
    elif r == "ExitState":
        # rest is patternNum
        return ("Exit",rest)
    elif r == "Transitions":
        string_triples = re.findall("\\(\d+,\d+,\w\\)",rest)
        triple_strings = list(map(conv_string_to_transition, string_triples))
        return("Transitions",triple_strings)
    # elif r == "END_BASEPATTERN" or r == "END_BasePattern" or r == "End_BASEPATTERN" or r == "End_BasePattern": # this case handled above
    #    return ("END_BASEPATTERN",0)
    elif (r ==  "CompositePattern"):
        # search the rest of the line for 3 patternNums separated by commas and optional spaces
        # each one is a group (since enclosed in parens)
        # match = re.search(r'(\w+)\s*\,\s*(\w+)\s*\,\s*(\w+)s*',rest)
        # return ("CompositePattern", (match.group(1),match.group(2),match.group(3)))
        # rest is patternNum
        return("CompositePattern",rest)
    elif (r== "LeftPattern") or (r == "leftPattern") or (r == "FirstPattern") or (r == "firstPattern"):
        return ("LeftPattern",rest)
    elif (r == "RightPattern") or (r == "rightPattern") or (r == "SecondPattern") or (r == "secondPattern"):
        return ("RightPattern", rest)
    elif (r == "Acyclic") or (r == "acyclic"):
        return ("Acyclic", 0)
    elif (r == "Cyclic") or (r == "cyclic"):
        return ("Cyclic", 0)
    elif (r ==  "StartRule"):
        # rest is patternNum
        return ("StartRule", rest)
    elif (r ==  "InsertRule"):
        # find LHS pattern num which comes before "->"
        ind2 = line.find("-")
        if ind2 != -1:  # found "-"
            LHSnum = line[ind+1:ind2] # LHSpatNum is from char after ":" until "-" char
            LHSnum.strip() # strip leading and trailing whitespace
            if line[ind2+1] == ">":
                line = line[ind+2] # line should now be of form <RHSpattern1> , <RHSpattern3>, <RHSpattern2>
                # search the rest of the line for 3 patternNums separated by commas and optional spaces
                # each one is a group (since enclosed in parens)
                match = re.search(r'(\w+)s*\,s*(\w+)s*\,s*(\w+)s*',rest)
                composite = (LHSnum,match.group(1),match.group(2),match.group(3))
                return ("InsertRule",composite)
            else:
                return("Error",line)
        else:
            return ("Error", line)
    else:
        print("create_PRSspec_from_input:parse_line: Format Error.  Did not correctly specify spec.")
        return ("Error",line)

# After line "BasePattern <integer>" parse following lines and return the base pattern
def parseBasePattern(fd):
    suppliedInit = False
    suppliedExit = False
    suppliedTrans = False
    done = False
    while not done:
        line = fd.readline()
        k, val = parse_line(line)
        if k == "Init":
            initState = val  # re.findall('\d+', rest) # return integer in line
            suppliedInit = True
        elif k == "Exit":
            exitState = int(val)
            suppliedExit = True
        elif k == "Transitions":
            if not suppliedTrans:
                trans = val
                suppliedTrans = True
            else:  # already found a line of transitions and this is an additional line
                trans = trans + val
        elif k == "END_BASEPATTERN":
            if suppliedInit and suppliedExit and suppliedTrans:
                done = True
            else:
                print("create_PRSspec_from_input:parseBasePattern: Found END_BASEATTERN before completed specification.")
                sys.exit()
        else:
            print("create_PRSspec_from_input:parseBasePattern: Format Error.  Did not correctly specify base pattern.")
            sys.exit()
    return (initState,exitState,trans)

# After line "CompositePattern <integer>" parse following lines and return the composite pattern
# CompositePatter is of form:
# CompositePattern: <patNum>
# LeftPattern: <patNum1>
# RightPattern: <patNum2>
# <cyclic>: which is either the keyword "Cyclic" or "Acyclic#
# End_CompositePattern
def parseCompositePattern(fd):
    suppliedFirst = False
    suppliedSecond = False
    suppliedCyclic = False
    done = False
    while not done:
        line = fd.readline()
        k, val = parse_line(line)
        if k == "LeftPattern":
            firstStateNum = val
            suppliedFirst = True
        elif k == "RightPattern":
            secondStateNum = val
            suppliedSecond = True
        elif k == "Cyclic":
            CycStatus = prs_globals.CompositeType.Cyclic
            suppliedCyclic = True
        elif k == "Acyclic":
            CycStatus = prs_globals.CompositeType.Acyclic
            suppliedCyclic = True
        elif k == "END_COMPOSITEPATTERN":
            if suppliedFirst and suppliedSecond and suppliedCyclic:
                done = True
            else:
                print("create_PRSspec_from_input:parseBasePattern: Found END_COMPOSITEPATTERN before completed specification.")
                sys.exit()
        else:
            print("create_PRSspec_from_input:parseBasePattern: Format Error.  Did not correctly specify composite pattern.")
            sys.exit()
    return (firstStateNum,secondStateNum,CycStatus)


def createPRSspec(fd):
    alpha = ""
    Patterns = {}
    startRules = []
    insertRules = {} # maps rulenNum to rule
    ruleNum = 1
    line = fd.readline()
    while line == "\n" or line == "":
        line = fd.readline()
    if line == " ":
        print("line one space")
    if line == "":
        print("line 0 space")
    while line:
    # while not end of file and have not reached end of PRS description
        k, val = parse_line(line)
        if k == "Alpha":
            alpha = val
        elif k == "BasePattern":
            patNum = int(val)
            # need to parse rest of base pattern spec
            initState,exitState,trans = parseBasePattern(fd)
            Patterns[patNum] = ((prs_globals.PatternType.Base,int(initState),int(exitState),trans))
        elif k == "CompositePattern":
            # val = (patNum,firstPatternNum,secondPatternNum)
            patNum = int(val)
            # need to parse rest of composite pattern spec
            firstPatNum,secondPatNum,cyclicType = parseCompositePattern(fd)
            (patType1,init1,exit1,trans1) = Patterns[int(firstPatNum)]
            (patType2,init2,exit2,trans2) = Patterns[int(secondPatNum)]
            # makeCompositePattern returns n init,exit,transitions,joinSt,highestState
            init,exit,transitions,joinSt = makeCompositePattern(init1, exit1, trans1, init2, exit2, trans2,cyclicType)
            # (PatternType.Composite,init,exit,transitions,jointState,firstPatNum,secondPatNum,compositeType)
            Patterns[int(patNum)] = (prs_globals.PatternType.Composite,init,exit,transitions,joinSt,int(firstPatNum),int(secondPatNum),cyclicType)
        elif k == "StartRule":
            # add patternNum of the RHS of this rule into startRules
            startRules.append(int(val))
        elif k  ==  "InsertRule" :
            # val = (LHSnum, match.group(1), match.group(2), match.group(3))
            # add patternNum of the RHS of this rule into startRules
            # ***** Note we are aking the index for insertRules a string to be consistent
            insertRules[ruleNum] = (int(val[0]),int(val[1]),int(val[2]),int(val[3]))
            ruleNum = ruleNum + 1
        elif k == "Comment":
            # skip this line, read next, and continue
            pass
        elif k == "End":
            # reached end of PRS description ("END_DFA")
            pass
        elif k == "Error":
            print("create_PRSspec_from_input:createPRSspec: Specification incorrect.\n")
            print("line = " + val)
            sys.exit()
        else:  # never reach this as all other cases are covered by "Error"
            print("create_PRSspec_from_input:createPRSspec: Specification incorrect.\n")
            print("k = ")
            print(k)
            print("val = ")
            print(val)
            sys.exit()
        line = fd.readline()
    return (alpha, Patterns, startRules, insertRules)