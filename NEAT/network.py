# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np

class Neuron:
    def __init__(self):
        self._value = 0.0
        self._resolved = False
        self._bias = np.random.rand() - 0.5
        self._incoming = []

    @staticmethod
    def from_json(obj):
        neuron = Neuron()
        neuron._value = obj['value']
        neuron._resolved = obj['resolved']
        neuron._bias = obj['bias']

        for gene in obj['incoming']:
            neuron._incoming.append(Gene.from_json(gene))

        return neuron

    def to_json(self):
        obj = {
            'value': self._value,
            'resolved': self._resolved,
            'bias': self._bias,
            'incoming': [],
        }

        for gene in self._incoming:
            obj['incoming'].append(gene.to_json())

        return obj

    def reset(self):
        self._value = 0.0
        self._resolved = False

    def bias(self):
        return self._bias

    def value(self):
        return self._value

    def set_value(self, value):
        self._value = value
        self._resolved = True

    def resolved(self):
        return self._resolved

    def incoming(self):
        return self._incoming

    def add_incoming(self, gene):
        self._incoming.append(gene)

class Network:
    def __init__(self, input, output, genes=[]):
        self._fitness = 0
        self._genes = genes
        self._input = input
        self._output = output
        self._mutation_rate = {
            'MUTATE_WEIGHT': 0.25,
            'PERTURB': 0.9,
            'PERTURB_BIAS': 0.1,
            'MUTATE_GENE': 1.0,
            'MUTATE_NEURON': 0.5,
            'ENABLE': 0.2,
            'DISABLE': 0.4,
        }
        self._max_neurons = input + output
        self._neurons = {}
        self._ranking = 0

    @staticmethod
    def from_json(obj):
        '''
        obj = {
            'fitness': self._fitness,
            'genes': [],
            'input': self._input,
            'output': self._output,
            'mutation_rate': self._mutation_rate,
            'max_neurons': self._max_neurons,
            'ranking': self._ranking
            'neurons': {},
        }
        '''

        network = Network(obj['input'], obj['output'], [])
        network._fitness = obj['fitness']
        network._mutation_rate = obj['mutation_rate']
        network._max_neurons = obj['max_neurons']
        network._ranking = obj['ranking']

        for gene in obj['genes']:
            network._genes.append(Gene.from_json(gene))

        for i, neuron in obj['neurons'].items():
            network._neurons[i] = Neuron.from_json(neuron)

        return network

    def to_json(self):
        obj = {
            'fitness': self._fitness,
            'genes': [],
            'input': self._input,
            'output': self._output,
            'mutation_rate': self._mutation_rate,
            'max_neurons': self._max_neurons,
            'ranking': self._ranking,
            'neurons': {},
        }

        #print(len(self._genes))
        for gene in self._genes:
            obj['genes'].append(gene.to_json())

        for i, neuron in self._neurons.items():
            obj['neurons'][i] = neuron.to_json()

        return obj

    @staticmethod
    def create_basic(input_len, output_len):
        network = Network(input=input_len, output=output_len, genes=[])

        # for out in range(output_len):
        #     for into in range(input_len):
        #         if np.random.rand() < 0.5:
        #             gene = Gene.create(into, out)
        #             network._genes.append(gene)

        return network

    @staticmethod
    def copy(network):
        new_gene = []
        for gene in network._genes:
            new_gene.append(Gene.copy(gene, new=False))

        new = Network(input=network._input, output=network._output, genes=new_gene)
        new._mutation_rate = network._mutation_rate.copy()
        new._max_neurons = network._max_neurons

        return new

    @staticmethod
    def crossover(mom, dad):
        # ensure mom has higher fitness
        if mom.fitness() < dad.fitness():
            t = mom
            mom = dad
            dad = t

        mom_genes = mom.genes()
        dad_genes = dad.genes()

        dad_inn = {}
        for gene in dad_genes:
            dad_inn[gene.innovation()] = gene

        child_genes = []

        for mom_gene in mom_genes:
            # same innovation
            if mom_gene.innovation() in dad_inn:
                dad_gene = dad_inn[mom_gene.innovation()]

                # same fitness -> random parent
                if mom.fitness() == dad.fitness():
                    if np.random.rand() < 0.5:
                        child_gene = Gene.copy(mom_gene, new=False)
                    else:
                        child_gene = Gene.copy(dad_gene, new=False)
                # fitter parent = mom
                else:
                    child_gene = Gene.copy(mom_gene, new=False)

                if not mom_gene.enabled() or not dad_gene.enabled():
                    if np.random.rand() < 0.75:
                        child_gene.set_enable(False)

                child_genes.append(child_gene)
            # disjoints or excess
            else:
                child_genes.append(Gene.copy(mom_gene, new=False))

        if mom.fitness() == dad.fitness():
            mom_inn = {}
            for gene in mom_genes:
                mom_inn[gene.innovation()] = gene

            for dad_gene in dad_genes:
                if dad_gene.innovation() not in mom_inn:
                    if np.random.rand() < 0.5:
                        child_genes.append(Gene.copy(dad_gene, new=False))

        child = Network(input=mom._input, output=mom._output, genes=child_genes)
        child._max_neurons = max(mom._max_neurons, dad._max_neurons)

        return child

    def generate(self):
        in_len = self._input
        out_len = self._output
        self._neurons = {}

        # input neurons
        for i in range(in_len):
            self._neurons[i] = Neuron()

        # output neurons
        for i in range(in_len, in_len + out_len):
            self._neurons[i] = Neuron()

        for gene in self._genes:
            if gene.enabled():
                if gene.out() not in self._neurons:
                    self._neurons[gene.out()] = Neuron()

                self._neurons[gene.out()].add_incoming(gene)

                if gene.into() not in self._neurons:
                    self._neurons[gene.into()] = Neuron()


    def _gene_exists(self, into, out):
        for gene in self._genes:
            if gene.link() == (into, out):
                return True

        return False

    def fitness(self):
        return self._fitness

    def add_fitness(self, fitness):
        self._fitness += fitness

    def genes(self):
        return self._genes

    def ranking(self):
        return self._ranking

    def set_ranking(self, rank):
        self._ranking = rank

    def _random_neuron(self, non_input=False):
        nl = self._max_neurons
        if non_input:
            return np.random.randint(self._input, nl)
        else:
            return np.random.randint(nl)

    # change gene weight
    def mutate_weight(self, perturb_prob):
        for gene in self._genes:
            if np.random.rand() < perturb_prob:
                new_w = gene.w() + np.random.rand() * self._mutation_rate['PERTURB_BIAS'] * (-1 if np.random.rand() < 0.5 else 1)
                gene.set_w(new_w)
            else:
                gene.set_w(np.random.rand() * 4 - 2)

    # add a new gene
    def mutate_gene(self):
        n1 = self._random_neuron(False)
        n2 = self._random_neuron(True)

        # if np.random.rand() < 0.4:
        #     n1 = np.random.randint(self._input)

        if n1 == n2 or self._gene_exists(n1, n2):
            return

        gene = Gene.create(n1, n2)
        self._genes.append(gene)

    # add new neuron
    def mutate_neuron(self):
        if len(self._genes) == 0:
            return

        gene = self._genes[np.random.randint(len(self._genes))]

        if not gene.enabled():
            return

        self._max_neurons += 1

        g1 = Gene.copy(gene)
        g1.set_w(1.0)
        g1.set_out(self._max_neurons)
        g2 = Gene.copy(gene)
        g2.set_into(self._max_neurons)

        self._genes += [g1, g2]

    def mutate_enable(self, enable):
        candidates = []

        for gene in self._genes:
            if gene.enabled() != enable:
                candidates.append(gene)

        if len(candidates) > 0:
            candidates[np.random.randint(len(candidates))].set_enable(enable)

    def mutate(self):
        for key, val in self._mutation_rate.items():
            if np.random.rand() < 0.5:
                self._mutation_rate[key] = val * 0.95
            else:
                self._mutation_rate[key] = val * 1.05263

        if np.random.rand() < self._mutation_rate['MUTATE_WEIGHT']:
            self.mutate_weight(self._mutation_rate['PERTURB'])

        if np.random.rand() < self._mutation_rate['MUTATE_GENE']:
            self.mutate_gene()

        if np.random.rand() < self._mutation_rate['MUTATE_NEURON']:
            #for _ in range(2):
            self.mutate_neuron()

        if np.random.rand() < self._mutation_rate['ENABLE']:
            self.mutate_enable(True)

        if np.random.rand() < self._mutation_rate['DISABLE']:
            self.mutate_enable(False)

        # print(self._max_neurons, len(self._genes))

    def evaluate(self, input):
        #self.generate()

        in_len = self._input
        out_len = self._output
        neurons = self._neurons

        def resolve_neuron(neuron, visited):
            sum = 0.0
            for incoming in neuron.incoming():
                other = neurons[incoming.into()]

                if other.resolved():
                    other_val = other.value()
                else:
                    if other in visited:
                        other_val = 0.0
                    else:
                        other_val = resolve_neuron(other, visited + [neuron])

                sum += other_val * incoming.w() + other.bias()

            if len(neuron.incoming()) > 0:
                neuron.set_value(sigmoid(sum))
                return neuron.value()
            else:
                return 0.0

        # feed input neurons
        for i in range(in_len):
            neurons[i].set_value(input[i])

        max_output, max_value = 0, -9999
        # recursively resolve output neurons
        for i in range(in_len, in_len + out_len):
            val = resolve_neuron(neurons[i], [])

            if val > max_value:
                max_output, max_value = i, val

        return max_output - in_len

from .gene import *
from .util import *
