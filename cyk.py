# cyk.py: parse a CNF PCFG
# CYK Algorithm Implemented by Ogi Moore
# Edited by Stephen Wu 
# Original by Kyle Gorman

from __future__ import division

from itertools import product
import numpy as np
from pprint import PrettyPrinter
from pcfg import PCFG
from bitweight import BitWeight
import os

PCFG_SOURCE = "PCFG.pkl.gz"
TOKEN_SOURCE = "tst-100.tok"


class Chart(object):
    """
    A CKY chart instance---a column tuple of row tuples of dict cells.
    """

    # initialize pretty printer (shared across all class instances)
    pformat = PrettyPrinter(width=75).pformat

    def __init__(self, size):
        if size <= 0:
            raise ValueError("Invalid size: {}.".format(size))
        self.chart = np.empty((size, size+1), dtype=dict)

    def __str__(self, width=20):
        rep = ''
        for i, row in enumerate(self.chart):
            for j, cell in enumerate(row):
                if cell:
                    try:
                        cell_nonterms = ",".join(cell.keys())
                    except AttributeError:
                        cell_nonterms = cell

                    if len(cell_nonterms) > width:
                        cell_nonterms = "{}..".format(cell_nonterms[:(width-4)])
                    rep += "{}".format(cell_nonterms).center(width)
                else:
                    rep += "{},{}".format(i, j).center(width)
            rep += "\n"
        return rep

    def __getitem__(self, idx):
        return self.chart[idx]

    def __setitem__(self, idx, val):
        self.chart[idx] = val


# Functions Created to Assist with the back-propagation
def backtrace_left(loc_o, backtrace_chart, root_val):
    loc = backtrace_chart[loc_o][root_val][0][1:]
    val = backtrace_chart[loc_o][root_val][0][0]
    return loc, val


def backtrace_right(loc_o, backtrace_chart, root_val):
    loc = backtrace_chart[loc_o][root_val][1][1:]
    val = backtrace_chart[loc_o][root_val][1][0]
    return loc, val


def CKY_chart(tokens, pcfg):
    """
    Create a CKY chart for the token string `tokens` given a PCFG `pcfg`.
    """
    f = open('workfile.txt', 'a')
    L = len(tokens)
    chart = Chart(L)
    backtrace_chart = Chart(L)
    top_results_to_return = 5

    # Filling in the main diagonal and applying the pre-Terminal Rules
    for index, token in enumerate(tokens):
        # place tokens along diagonal
        chart[index][index] = token
        # place pre-terminal rules along first off-diagonal
        chart[index][index+1] = pcfg.preterminal_rules(token).copy()

    for j in range(1, L+1):
        for i in range(j-2, -1, -1):
            for k in range(i+1, j):
                # Ignore the values in the matrix that were set by the pre-terminal rules
                if i == j or i == j-1:
                    continue
                # Handle matrix cells that have not been accessed before
                if type(chart[i][j]) is not dict:
                    chart[i][j] = {}
                    backtrace_chart[i][j] = {}

                for r in product(chart[i][k].keys(), chart[k][j].keys()):
                    if pcfg.nonterminal_rules(r):
                        non_terminal_rules = pcfg.nonterminal_rules(r).copy()
                        backtrace = {}
                        for rule in non_terminal_rules.keys():
                            # Calculating the Conditional Probabilities
                            non_terminal_rules_updated = {}
                            non_terminal_rules_updated[rule] = \
                                non_terminal_rules[rule] * \
                                chart[i][k][r[0]] * \
                                chart[k][j][r[1]]
                            backtrace[rule] = ((r[0], i, k), (r[1], k, j))
                            if not (rule in chart[i][j].keys() and
                                    non_terminal_rules_updated[rule] <=
                                    chart[i][j][rule]):
                                # Adding the possible trees here
                                chart[i][j].update(non_terminal_rules_updated)
                                backtrace_chart[i, j].update(backtrace)
                    # If there are no rules for the combination above, ignore and move on
                    else:
                        pass
    # Applying the Start Rules
    for key in chart[0][L].keys():
        if pcfg.start_rules(key):
            chart[0][L][key] *= pcfg.start_rules(key)
        # No need to keep start entries that don't meet the start rules
        # else:
        #     del chart[0][L][key]

    # Best 5 Top Parts of Speech Tags
    best_outcomes = sorted(chart[i][j], key=chart[i][j].get,
                           reverse=True)[0:top_results_to_return]

    best_probability = [None] * top_results_to_return
    for index, result in enumerate(best_outcomes):
        resulting_chart = Chart(L)
        best_probability[index] = chart[0, L].get(result)
        # take the appropriate entry
        resulting_chart[0, L] = result.split('^')[0].split('|')[0]

        # Perform Back Propagation (this took forever to implement!)
        possibilities = [((0, L), result)]
        # Ensuring that the back-propagation doesn't run away on me, implementing an L^3 cap
        for counter in range(L**3):
            loc = possibilities[-1][0]
            val = possibilities[-1][1]

            loc_r, val_r = backtrace_left(loc, backtrace_chart, val)
            resulting_chart[loc_r] = val_r.split('^')[0].split('|')[0]

            loc_c, val_c = backtrace_right(loc, backtrace_chart, val)
            resulting_chart[loc_c] = val_c.split('^')[0].split('|')[0]

            # Checking to make sure we are not at the pre-terminals
            if loc_r[0] != loc_r[1] - 1:
                possibilities.append((loc_r, val_r))
            if loc_c[0] != loc_c[1] - 1:
                possibilities.append((loc_c, val_c))

            # Adding dashes and pipes to better visualize the tree
            if loc[1] - loc_r[1] > 1:
                resulting_chart[loc[0], loc_r[1]+1:loc[1]] = '-' * 16
            if loc_c[0] - loc[0] > 1:
                resulting_chart[loc[0]+1:loc_c[0], loc[1]] = '|'

            # Removing the possibilities we just implemented
            possibilities.remove((loc, val))

            # If no more possibilities, no more back-propagation!
            if not possibilities:
                break

        # Cleaning up the formatting on the bottom half of the matrix
        for index, token in enumerate(tokens):
            resulting_chart[index, index] = token

        for row in range(1, L):
            for col in range(0, row):
                resulting_chart[row][col] = ' '

        for row in range(0, L):
            for col in range(row+2, L+1):
                if not resulting_chart[row, col]:
                    resulting_chart[row, col] = ' '
        f.write(str(resulting_chart))
        f.write('\n')
    f.close()
    return (chart, best_probability[0])

if __name__ == "__main__":
    try:
        os.remove('workfile.txt')
    except OSError:
        pass
    sentences_to_parse = 10
    pcfg = PCFG.load(PCFG_SOURCE)
    with open(TOKEN_SOURCE, "r") as source:
        for index, line in enumerate(source):
            if index == sentences_to_parse:
                break
            tokens = line.split()
            (_, best_probability) = CKY_chart(tokens, pcfg)
            print("{:.4f}: {}".format(best_probability.bw, " ".join(tokens)))
