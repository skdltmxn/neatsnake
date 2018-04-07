# -*- coding: utf-8 -*-

import numpy as np

class Gene:
    def __init__(self, into, out, w, enable, innovation):
        self._into = into
        self._out = out
        self._w = w
        self._enable = enable
        self._innovation = innovation

    @staticmethod
    def create(into, out):
        return Gene(into, out, np.random.rand() * 2 - 1, True, Neat.new_innovation())

    @staticmethod
    def copy(src, new=True):
        if new:
            gene = Gene(src._into, src._out, src._w, src._enable, Neat.new_innovation())
        else:
            gene = Gene(src._into, src._out, src._w, src._enable, src.innovation())
            
        return gene

    def w(self):
        return self._w

    def set_w(self, w):
        self._w = w

    def into(self):
        return self._into

    def set_into(self, into):
        self._into = into

    def out(self):
        return self._out

    def set_out(self, out):
        self._out = out

    def enabled(self):
        return self._enable

    def set_enable(self, enable):
        self._enable = enable

    def innovation(self):
        return self._innovation

    def link(self):
        return (self._into, self._out)

# class Genome:
#     def __init__(self):
#         self.genes = []
#         self.network = {}
#
#     def add(self, gene):
#         self.genes.append(gene)
#
#     def evaluate(self):
#         pass

from .neat import *
