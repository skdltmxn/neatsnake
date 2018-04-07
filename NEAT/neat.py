# -*- coding: utf-8 -*-

import math

class Neat:
    innovation = 0

    def __init__(self, population=50, input_size=400, output_size=4):
        #self.pool = Pool(population)
        self._population = population
        self._species = []
        self._current_species = 0
        self._current_network = 0
        self._generation = 1
        self._input_size = input_size
        self._output_size = output_size
        self._total_adjust_fitness = 0

        for _ in range(self._population):
            network = Network.create_random(input_size, output_size)
            network.mutate()
            self.add_species(network)

    def add_species(self, network):
        for species in self._species:
            existing_network = species.fetch_random_network()
            if existing_network is not None and is_same_species(network, existing_network):
                species.add_network(network)
                return

        # could not find one, create new
        species = Species()
        species.add_network(network)
        self._species.append(species)

    def _remove_stale_species(self):
        for species in self._species:
            mfn = species.max_fitness_now()
            if species.max_fitness() < mfn:
                species.set_max_fitness(mfn)
            else:
                if species.add_stale_count() >= 15:
                    # too many disappointments... remove it
                    self._species.remove(species)

    def next_generation(self):
        self._remove_stale_species()

        for species in self._species:
            # remove lowest performing members
            species.remove_lower(species.num_networks() // 2)

            # calculate adjust fitness
            self._total_adjust_fitness += species.calculate_adjust_fitness() + 10

        children = []
        for species in self._species:
            n_children = math.floor(species.adjust_fitness() / self._total_adjust_fitness * self._population)
            for _ in range(n_children):
                children.append(species.make_child())

            species.remove_lower(1)

        for _ in range(self._population - len(children) - len(self._species)):
            species = self._species[np.random.randint(len(self._species))]
            children.append(species.make_child())

        for child in children:
            self.add_species(child)

        self._current_network = 0
        self._generation += 1

    def next(self):
        self._current_network += 1
        species = self._species[self._current_species]

        if self._current_network == species.num_networks():
            self._current_network = 0
            self._current_species += 1
            if self._current_species == len(self._species):
                self.next_generation()
                self._current_species = 0

        #print(self._current_network, self._current_species)

    def generation(self):
        return self._generation

    def current_network(self):
        return self._current_network

    def current_species(self):
        return self._current_species

    @staticmethod
    def new_innovation():
        Neat.innovation += 1
        return Neat.innovation

    def evaluate(self, input):
        species = self._species[self._current_species]
        network = species.network(self._current_network)

        return network.evaluate(input)

from .network import *
from .species import *
from .util import is_same_species
