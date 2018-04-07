# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np

class Neuron:
    def __init__(self, type):
        self.value = 0.0
        self.bias = np.random.rand()
        self.type = type # in, out, hidden

    @staticmethod
    def create(type):
        return Neuron(type)

    def type(self):
        return self.type

class Network:
    def __init__(self, input=[], output=[], genes=[]):
        self._fitness = 0
        self._genes = genes
        self.input_neuron_len = len(input)
        self.output_neuron_len = len(output)
        self.mutation_rate = {
            'MUTATE_WEIGHT': 0.5,
            'PERTURB': 0.9,
            'PERTURB_BIAS': 0.1,
            'MUTATE_GENE': 1.0,
            'MUTATE_NEURON': 0.5,
            'ENABLE': 0.2,
            'DISABLE': 0.4,
        }
        self._max_neurons = len(input) + len(output)

    @staticmethod
    def create_random(input_len, output_len):
        input = []
        for _ in range(input_len):
            input.append(Neuron.create('in'))

        output = []
        for _ in range(output_len):
            output.append(Neuron.create('out'))

        return Network(input=input, output=output, genes=[])

    @staticmethod
    def copy(network):
        new_gene = []
        for gene in network._genes:
            new_gene.append(Gene.copy(gene, new=False))

        new = Network(genes=new_gene)
        new.mutation_rate = network.mutation_rate.copy()
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
            if gene.innovation() in dad_inn:
                dad_gene = dad_inn[gene.innovation()]

                # same fitness -> random parent
                if mom.fitness() == dad.fitness():
                    if np.random.rand() < 0.5:
                        child_genes.append(mom_gene)
                    else:
                        child_genes.append(dad_gene)
                # fitter parent = mom
                else:
                    child_genes.append(mom_gene)
            # disjoints or excess
            else:
                child_genes.append(mom_gene)

        child = Network(genes=child_genes)
        child._max_neurons = max(mom._max_neurons, dad._max_neurons)

        return child

    def fitness(self):
        return self._fitness

    def genes(self):
        return self._genes

    def _gene_exists(self, into, out):
        for gene in self._genes:
            if gene.link() == (into, out):
                return True

        return False

    def _random_neuron(self, non_input=False):
        nl = self._max_neurons
        if non_input:
            return np.random.randint(self.input_neuron_len, nl)
        else:
            return np.random.randint(nl)

    def mutate_weight(self, perturb_prob):
        for gene in self._genes:
            if np.random.rand() < perturb_prob:
                new_w = gene.w() + np.random.rand() * self.mutation_rate['PERTURB_BIAS'] * (-1 if np.random.rand() < 0.5 else 1)
                gene.set_w(new_w)
            else:
                gene.set_w(np.random.rand())

    def mutate_gene(self):
        while True:
            n1 = self._random_neuron(False)
            n2 = self._random_neuron(True)

            if n1 != n2 and not self._gene_exists(n1, n2):
                break

        gene = Gene.create(n1, n2)
        self._genes.append(gene)

    def mutate_neuron(self):
        if len(self._genes) == 0:
            return

        #n = Neuron.create_random('hidden')
        #self.neurons.append(n)

        while True:
            gene = self._genes[np.random.randint(len(self._genes))]

            if gene.enabled():
                gene.set_enable(False)
                break

        g1 = Gene.copy(gene)
        g1.set_w(1.0)
        g1.set_out(self._max_neurons)
        g2 = Gene.copy(gene)
        g2.set_into(self._max_neurons)

        self._max_neurons += 1
        self._genes += [g1, g2]

    def mutate(self):
        for key, val in self.mutation_rate.items():
            if np.random.rand() < 0.5:
                self.mutation_rate[key] = val * 0.95
            else:
                self.mutation_rate[key] = val * 1.05263

        if np.random.rand() < self.mutation_rate['MUTATE_WEIGHT']:
            self.mutate_weight(self.mutation_rate['PERTURB'])

        if np.random.rand() < self.mutation_rate['MUTATE_GENE']:
            self.mutate_gene()

        if np.random.rand() < self.mutation_rate['MUTATE_NEURON']:
            self.mutate_neuron()

    def evaluate(self):
        pass

from .gene import *
