# -*- coding: utf-8 -*-

from .gene import *

class Pool:
    def __init__(self, population):
        self.population = population
        self.species = []
        self.generation = 0
        self.innovation = 0

    def new_innovation(self):
        self.innovation += 1
        return self.innovation
