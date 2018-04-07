# -*- coding: utf-8 -*-

import json
import math
import os

class Neat:
    innovation = 0
    save_path = './save'

    def __init__(self, population=30, input_size=400, output_size=4):
        #self.pool = Pool(population)
        self._population = population
        self._species = []
        self._current_species = 0
        self._current_network = 0
        self._generation = 1
        self._input_size = input_size
        self._output_size = output_size
        self._network_cache = None

    def init(self):
        for _ in range(self._population):
            network = Network.create_basic(self._input_size, self._output_size)
            network.mutate()
            self.add_species(network)

        self._network_cache = self._species[0].network(0)

    def load(self):
        base = Neat.save_path
        if not os.path.exists(base):
            self.init()
            return

        max_gen = 0
        for _, _, file in os.walk(base):
            for f in file:
                gen = int(f.split('.')[0])
                if gen > max_gen:
                    max_gen = gen

        path = os.path.join(base, '{0}.txt'.format(max_gen))
        print('loading', path)

        with open(path, 'rb') as f:
            obj = json.loads(f.read())
            self._population = obj['population']
            self._input_size = obj['input_size']
            self._output_size = obj['output_size']
            self._generation = max_gen

            for species in obj['species']:
                self._species.append(Species.from_json(species))

        self._network_cache = self._species[0].network(0)

    def save(self, base):
        if not os.path.exists(base):
            os.makedirs(base)

        path = os.path.join(base, '{0}.txt'.format(self._generation))

        with open(path, 'wb') as f:
            obj = {
                'population': self._population,
                'species': [],
                'input_size': self._input_size,
                'output_size': self._output_size,
            }

            for species in self._species:
                obj['species'].append(species.to_json())

            f.write(json.dumps(obj).encode())

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
        survived = []
        for species in self._species:
            mfn = species.max_fitness_now()
            if species.max_fitness() < mfn:
                species.set_max_fitness(mfn)
            else:
                if species.add_stale_count() >= 15:
                    # too many disappointments... remove it
                    continue

            survived.append(species)

        self._species = survived

    def _global_ranking(self):
        ranking = {}

        for species in self._species:
            for network in species.networks():
                ranking[network] = network.fitness()

        ranking = sorted(ranking, key=ranking.get)

        rank = 1
        for network in ranking:
            network.set_ranking(rank)
            rank += 1

    def _remove_weak_species(self):
        total_adjust_fitness = 0.0
        survived = []

        for species in self._species:
            total_adjust_fitness += species.calculate_adjust_fitness()

            if math.floor(species.adjust_fitness() / total_adjust_fitness * self._population) >= 1:
                survived.append(species)

        self._species = survived

    def next_generation(self):

        self._remove_stale_species()
        self._global_ranking()
        self._remove_weak_species()

        total_adjust_fitness = 0.0

        for species in self._species:
            # remove lowest performing members
            species.remove_lower(species.num_networks() // 2)

            # calculate adjust fitness
            total_adjust_fitness += species.calculate_adjust_fitness()

        children = []
        for species in self._species:
            n_children = math.floor(species.adjust_fitness() / total_adjust_fitness * self._population)

            for _ in range(n_children):
                children.append(species.make_child())

            species.remove_lower(1)

        rest = self._population - len(children) - len(self._species)

        if rest > 0:
            for _ in range(rest):
                species = self._species[np.random.randint(len(self._species))]
                children.append(species.make_child())

        for child in children:
            self.add_species(child)

        # print(list(map(Species.num_networks, self._species)))

        self._current_network = 0
        self._generation += 1

        self.save(Neat.save_path)


    def next(self):
        self._current_network += 1
        species = self._species[self._current_species]

        if self._current_network == species.num_networks():
            self._current_network = 0
            self._current_species += 1
            if self._current_species == len(self._species):
                self._current_species = 0
                self.next_generation()

        species = self._species[self._current_species]
        self._network_cache = species.network(self._current_network)
        #print(self._current_network, self._current_species)

    def generation(self):
        return self._generation

    def current_species(self):
        return self._current_species

    def current_network(self):
        return self._current_network

    @staticmethod
    def new_innovation():
        Neat.innovation += 1
        return Neat.innovation

    def evaluate(self, input):
        #species = self._species[self._current_species]
        #network = species.network(self._current_network)

        return self._network_cache.evaluate(input)

    def add_fitness(self, fitness):
        self._network_cache.add_fitness(fitness)
        return self._network_cache.fitness()


from .network import *
from .species import *
from .util import is_same_species
