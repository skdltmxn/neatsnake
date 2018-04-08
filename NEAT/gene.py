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
    def from_json(obj):
        return Gene(
            obj['into'],
            obj['out'],
            obj['w'],
            obj['enable'],
            obj['innovation']
        )

    def to_json(self):
        return {
            'into': self._into,
            'out': self._out,
            'w': self._w,
            'enable': self._enable,
            'innovation': self._innovation,
        }

    @staticmethod
    def create(into, out):
        return Gene(into, out, np.random.rand() * 4 - 2, True, Neat.new_innovation())

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

from .neat import *
