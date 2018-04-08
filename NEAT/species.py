# -*- coding: utf-8 -*-

import numpy as np

CROSSOVER_RATE = 0.75

class Species:
    def __init__(self):
        self._networks = []
        self._max_fitness = 0
        self._adjust_fitness = 0
        self._stale_count = 0

    @staticmethod
    def from_json(obj):
        species = Species()
        species._max_fitness = obj['max_fitness']
        species._adjust_fitness = obj['adjust_fitness']
        species._stale_count = obj['stale_count']

        for network in obj['networks']:
            species._networks.append(Network.from_json(network))

        return species

    def to_json(self):
        obj = {
            'max_fitness': self._max_fitness,
            'adjust_fitness': self._adjust_fitness,
            'stale_count': self._stale_count,
            'networks': [],
        }

        for network in self._networks:
            obj['networks'].append(network.to_json())

        return obj

    def add_stale_count(self):
        self._stale_count += 1
        return self._stale_count

    def max_fitness(self):
        return self._max_fitness;

    def set_max_fitness(self, fitness):
        self._max_fitness = fitness
        self._stale_count = 0

    def max_fitness_now(self):
        sorted_networks = sorted(self._networks, key=Network.fitness, reverse=True)
        return sorted_networks[0].fitness()

    def adjust_fitness(self):
        return self._adjust_fitness;

    def calculate_adjust_fitness(self):
        sum = 0
        for network in self._networks:
            sum += network.ranking()

        self._adjust_fitness = sum / len(self._networks)

        return self._adjust_fitness

    def make_child(self):
        if len(self._networks) > 1 and np.random.rand() < CROSSOVER_RATE:
            mom = self.fetch_random_network()
            dad = self.fetch_random_network()

            child = Network.crossover(mom, dad)
        else:
            n = self.fetch_random_network()
            child = Network.copy(n)

        child.mutate()
        return child

    def num_networks(self):
        return len(self._networks)

    def network(self, idx):
        assert idx < len(self._networks)
        return self._networks[idx]

    def networks(self):
        return self._networks

    def remove_lower(self, remaining):
        if remaining == 0:
            return

        rank = sorted(self._networks, key=Network.ranking, reverse=True)
        self._networks = rank[:remaining]
        return rank[:remaining], rank[remaining:]

    def add_network(self, network):
        self._networks.append(network)

    def fetch_random_network(self):
        n = len(self._networks)
        if n == 0:
            return None

        return self.network(np.random.randint(n))

from .network import *
