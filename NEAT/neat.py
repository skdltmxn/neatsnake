# -*- coding: utf-8 -*-

import json
import math
import os

class Neat:
    innovation = 0

    def __init__(self, population=50, input_size=400, output_size=4, save_path='./save'):
        #self.pool = Pool(population)
        self._population = population
        self._species = []
        self._current_species = 0
        self._current_network = 0
        self._generation = 1
        self._input_size = input_size
        self._output_size = output_size
        self._network_cache = None
        self._save_path = save_path

    def init(self):
        for _ in range(self._population):
            network = Network.create_basic(self._input_size, self._output_size)
            network.mutate()
            self.add_species(network)

        self._network_cache = self._species[0].network(0)
        self._network_cache.generate()

    def get_graph(self):
        base = self._save_path
        if not os.path.exists(base):
            return

        with open('graph.csv', 'wb') as f_g:
            for _, _, file in os.walk(base):
                for f in file:
                    gen = int(f.split('.')[0])
                    path = os.path.join(base, f)

                    data = open(path, 'rb').read()
                    obj = json.loads(data)
                    species = []
                    for sp in obj['species']:
                        species.append(Species.from_json(sp))

                    max_fitness = -2000
                    for sp in species:
                        for nw in sp.networks():
                            if nw.fitness() > max_fitness:
                                max_fitness = nw.fitness()

                    f_g.write('{0},{1}\n'.format(gen, max_fitness).encode())


    def load(self):
        base = self._save_path
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
        self._network_cache.generate()

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
        if len(self._species) < 2:
            return

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

    def _total_adjust_fitness(self):
        total_adjust_fitness = 0.0
        minimum = 0.0

        for species in self._species:
            adjust_fitness = species.calculate_adjust_fitness()

            if adjust_fitness < minimum:
                minimum = adjust_fitness

            total_adjust_fitness += adjust_fitness

        if minimum < 0.0:
            total_adjust_fitness += abs(minimum) * len(self._species)

        return total_adjust_fitness, minimum

    def _remove_weak_species(self):
        survived = []
        total_adjust_fitness, minimum = self._total_adjust_fitness()

        for species in self._species:
            if math.floor((species.adjust_fitness() + abs(minimum)) / total_adjust_fitness * self._population) >= 1:
                survived.append(species)

        #print('{} / {} survived'.format(len(survived), len(self._species)))
        self._species = survived

    def _respeciate(self):
        unordered = []
        for species in self._species:
            _, rest = species.remove_lower(1)
            unordered += rest

        for network in unordered:
            self.add_species(network)

    def _total_networks(self):
        total = 0
        for species in self._species:
            total += species.num_networks()

        return total

    def _unspeciate(self):
        networks = []
        for species in self._species:
            for network in species.networks():
                networks.append(network)

        self._species = []

        return networks

    def _rullet(self, networks, minimum):
        minimum = abs(minimum)
        fitness_sum = sum(map(Network.fitness, networks)) + (minimum * len(networks))

        r = np.random.randint(fitness_sum)
        s = 0

        for network in networks:
            s += network.fitness() + minimum
            if s > r:
                return network

        return networks[0]

    def next_generation(self):
        networks = self._unspeciate()

        # copy top 3 networks without any mutation
        elite = 3

        ranking = sorted(networks, key=Network.fitness, reverse=True)
        #print(list(map(Network.fitness, ranking)))

        for i in range(elite):
            self.add_species(ranking[i])

        for _ in range(self._population - elite):
            mom = self._rullet(ranking, ranking[-1].fitness())
            dad = self._rullet(ranking, ranking[-1].fitness())

            child = Network.crossover(mom, dad)
            child.mutate()
            #next_networks.append(child)
            self.add_species(child)

        #self._global_ranking()

        # for species in self._species:
        #     # remove lowest performing members
        #     species.remove_lower(species.num_networks() // 2)
        #
        # self._remove_stale_species()
        # self._remove_weak_species()
        # #self._global_ranking()
        #
        # # now no species have negative average fitness
        # total_adjust_fitness, minimum = self._total_adjust_fitness()
        #
        # children = []
        # for species in self._species:
        #     n_children = math.floor((species.adjust_fitness() + abs(minimum)) / total_adjust_fitness * self._population) - 1
        #
        #     if n_children > 0:
        #         for _ in range(n_children):
        #             children.append(species.make_child())
        #
        #     species.remove_lower(1)
        #
        # rest = self._population - len(self._species) - len(children)
        # if rest > 0:
        #     for _ in range(rest):
        #         ns = len(self._species)
        #         species = self._species[np.random.randint(ns) if ns > 1 else 0]
        #         children.append(species.make_child())
        #
        # for child in children:
        #     self.add_species(child)
        #
        #print(self._total_networks(), list(map(Species.num_networks, self._species)))

        self._current_network = 0
        self._generation += 1

        self.save(self._save_path)

    def _next(self):
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

    def next(self):
        while self._network_cache.fitness() != 0:
            self._next()

        self._network_cache.generate()
        #self._network_cache.to_string()

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

    def fitness(self):
        return self._network_cache.fitness()

from .network import *
from .species import *
from .util import is_same_species
