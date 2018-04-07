# -*- coding: utf-8 -*-

# 1, 1, 0.4 - crAIg
# 2, 2, 0.4 - marI/O
C1 = 1.5
C3 = 0.4

# 2.5 - crAIg
# 1.0 - marI/O
THRESHOLD = 1.7

def disjoints(genes1, genes2):
    s1 = set()
    for gene in genes1:
        s1.add(gene.innovation())

    s2 = set()
    for gene in genes2:
        s2.add(gene.innovation())

    n = max(len(genes1), len(genes2))
    if n < 10:
        n = 1

    return len(s1 ^ s2) / n

def weights(genes1, genes2):
    s1 = {}
    for gene in genes1:
        s1[gene.innovation()] = gene.w()

    sum = 0
    n = 0
    for gene in genes2:
        if gene.innovation() in s1:
            sum += abs(gene.w() - s1[gene.innovation()])
            n += 1

    if n == 0:
        n = 1

    return sum / n

def distance(genes1, genes2):
    # we set C1 = C2, so no need to calculate excess separately
    D = disjoints(genes1, genes2)
    W = weights(genes1, genes2)

    return (C1 * D) + (C3 * W)

def is_same_species(network1, network2):
    d = distance(network1.genes(), network2.genes())
    #print(d)
    return d < THRESHOLD

from .gene import *
