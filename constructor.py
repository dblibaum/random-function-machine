"""Class which constructs and evolves the whole mapper.
   Data flows like this:

   Input => SemanticalMapper => Structure => SemanticalMapper => Attention => TypeComputer => OutputMapper

   Use: Initialize with the variables described in the initializer, then call evolve with number of generations to
        evolve. Saves best state to best_state.pkl.
   Guidelines:
        n_objects: The number of objects will effectively determine how fuzzily the data will be looked at.
        n_types: The number of types effectively determines the logical capacity of the constructor. Think of types as
                 general objects that have distinct behaviors with each other. Mapping an object to a type is like
                 assigning an abstract mathematical structure to a symbol. Exactly like that, actually.
        max_attention_depth: The number of layers in a Structure that the Attention will select type pairs from. E.g.
                             how high-level a representation to look at.
        max_attention_objects: The number of types in each layer of a Structure that the Attention will select. This
                               represents the overall number of types the constructor will operate with. This greatly
                               affects the number of outputs of the network, as well.
        computer_depth: This is the number of times that types will be recursively operated on. This, along with
                        max_attention_objects, drastically affect the number of outputs.
        n_functions: This is the number of random functions available to choose from to operate on types.
                     More functions means a greater logical capacity, but larger search space.
"""

from create_structure import Structure
from semantical_mapper import SemanticalMapper
from attention import Attention
from type_computer import TypeComputer
from output_mapper import OutputMapper
import cPickle
from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators
from pyevolve import Initializators
from pyevolve import GAllele
from pyevolve import Consts
import random as r


class Constructor:

    def __init__(self, datafile, n_objects, n_types, max_attention_depth, max_attention_objects,
                 computer_depth, n_functions, test_fraction=0, data_fraction=1, stochastic=False, batch_size=0,
                 save_file="best_construct.pkl", sep_features_targets=False):
        """
        :param datafile: A list containing input/target pairs, e.g. [[input, target], ...]
        :param data_fraction: Fraction of the dataset to use.
        :param test_fraction: The fraction of the dataset to reserve for testing.
        :param n_objects: The number of objects which Structure reduces the data to.
        :param n_types: The number of types SemanticalMapper maps the objects to. This is the number of separate types
                        as defined by their behavior computed by TypeComputer.
        :param max_attention_depth: The depth of reduction in the Structure that Attention will look for type pairs in.
        :param max_attention_objects: The number of type pairs in each layer of Structure that Attention will look for.
        :param computer_depth: The number of times types will be recursively passed through the functions of
                               TypeComputer.
        :param n_functions: The size of the function set of TypeComputer.
        """

        self.n_objects = n_objects
        self.n_types = n_types
        self.max_attention_depth = max_attention_depth
        self.max_attention_objects = max_attention_objects
        self.computer_depth = computer_depth
        self.n_functions = n_functions
        self.stochastic = stochastic
        self.batch_size = batch_size
        self.save_file = save_file

        self.best_fit = 0.0

        df = open(datafile, "rb")
        self.data = cPickle.load(df)
        df.close()

        features = [str(self.data[i][2:]) for i in range(len(self.data))]
        targets = [str(self.data[i][1]) for i in range(len(self.data))]

        self.data = zip(features, targets)

        if sep_features_targets:
            self.data = zip(self.data[0], self.data[1])

        if data_fraction < 1:
            self.data = [pair for pair in self.data if r.random() < data_fraction]

        print self.data[0][0]
        print self.data[0][1]

        train_len = int(round(len(self.data)*(1 - test_fraction)))
        self.train_data = self.data[:train_len]
        self.test_data = self.data[train_len:]

        print len(self.train_data)
        print len(self.test_data)

        inputs = []
        outputs = []
        input_symbols = []

        for pair in self.data:
            inputs.append(pair[0])
            outputs.append(pair[1])

        for x in inputs:
            for y in x:
                input_symbols.append(y)

        # Initialize layers
        self.input_mapper = SemanticalMapper(first_layer=True, inputs=input_symbols)
        self.structure = Structure(n_objects)
        self.semantical_mapper = SemanticalMapper(n_objects, n_types)
        self.attention = Attention(max_attention_depth, max_attention_objects, n_types)
        self.type_computer = TypeComputer(max_attention_depth*max_attention_objects*2, num_functions=self.n_functions,
                                          depth=computer_depth, n_types=self.n_types)
        self.output_mapper = OutputMapper(n_types, outputs)

    def compute(self, data):
        mapped = self.input_mapper.compute(data)
        structure = self.structure.make(mapped)
        for i in range(len(structure)):
            structure[i] = self.semantical_mapper.compute(structure[i])
        filtered = self.attention.filter(structure)
        outputs = self.type_computer.compute(filtered)
        outputs = self.output_mapper.compute(outputs)

        if len(outputs) != 0:
            output = int(outputs[-1])
        else:
            output = 1

        return output

    def set(self, savefile):
        cfile = open(savefile, "rb")
        chromosome = cPickle.load(cfile)

        index = 0
        set_input_mapper = chromosome[0:self.input_mapper.n_symbols]
        index += self.input_mapper.n_symbols
        set_structure_rmap = chromosome[index:index+len(self.structure.r_map)]
        index += len(self.structure.r_map)
        set_semantical_mapper = chromosome[index:index+len(self.semantical_mapper.map)]
        index += len(self.semantical_mapper.map)
        set_attention = chromosome[index:index+self.max_attention_depth*self.max_attention_objects*2]
        index += self.max_attention_depth*self.max_attention_objects*2
        set_computer = chromosome[index:index+len(self.type_computer.path)*len(self.type_computer.path[0])]
        index += len(self.type_computer.path)*len(self.type_computer.path[0])
        set_output_mapper = chromosome[index:]

        self.input_mapper.set(set_input_mapper)
        self.structure.set(set_structure_rmap)
        self.semantical_mapper.set(set_semantical_mapper)
        self.attention.set(set_attention)
        self.type_computer.set(set_computer)
        self.output_mapper.set(set_output_mapper)

    def eval_func(self, chromosome, report_test=True):

        error = 0.0
        error_local = 0.0
        test_error = 0.0
        test_error_local = 0.0

        index = 0
        set_input_mapper = chromosome[0:self.input_mapper.n_symbols]
        index += self.input_mapper.n_symbols
        set_structure_rmap = chromosome[index:index+len(self.structure.r_map)]
        index += len(self.structure.r_map)
        set_semantical_mapper = chromosome[index:index+len(self.semantical_mapper.map)]
        index += len(self.semantical_mapper.map)
        set_attention = chromosome[index:index+self.max_attention_depth*self.max_attention_objects*2]
        index += self.max_attention_depth*self.max_attention_objects*2
        set_computer = chromosome[index:index+len(self.type_computer.path)*len(self.type_computer.path[0])]
        index += len(self.type_computer.path)*len(self.type_computer.path[0])
        set_output_mapper = chromosome[index:]

        self.input_mapper.set(set_input_mapper)
        self.structure.set(set_structure_rmap)
        self.semantical_mapper.set(set_semantical_mapper)
        self.attention.set(set_attention)
        self.type_computer.set(set_computer)
        self.output_mapper.set(set_output_mapper)

        if self.stochastic:
            indexes = [r.randint(0, len(self.train_data) - 1) for x in range(self.batch_size)]
            train_data = [self.train_data[x] for x in indexes]
            indexes = [r.randint(0, len(self.test_data) - 1) for x in range(self.batch_size/2)]
            test_data = [self.test_data[x] for x in indexes]
        else:
            train_data = self.train_data
            test_data = self.test_data

        # print "=>Evaluating training data..."
        for pair in train_data:
            inp = pair[0]
            
            # For normal sequence:
            # target = [self.output_dict[x] for x in pair[1]]
            
            # For classification:
            target = pair[1]

            mapped = self.input_mapper.compute(inp)
            structure = self.structure.make(mapped)
            for i in range(len(structure)):
                structure[i] = self.semantical_mapper.compute(structure[i])
            filtered = self.attention.filter(structure)
            outputs = self.type_computer.compute(filtered)
            outputs = self.output_mapper.compute(outputs)

            # For normal sequence:
            # if len(outputs) >= len(target):
            #         for i in range(len(outputs)):
            #             if i < len(target):
            #                 if outputs[i] != target[i]:
            #                     error_local += 1
            #             else:
            #                 error_local += 1
            # else:
            #     for i in range(len(target)):
            #         if i < len(outputs):
            #             if outputs[i] != target[i]:
            #                 error_local += 1
            #         else:
            #             error_local += 1

            # error += error_local

            # For classification:
            if len(outputs) != 0:
                output = int(outputs[-1])
            else:
                output = 0

            if output != int(target):
                error += 1

        # print "Train acc: " + str((len(train_data) - error)/len(train_data)) + " Train error: " + str(error)

        # print "=>Evaluating testing data..."
        if report_test:
            for pair in test_data:
                inp = pair[0]
                
                # For normal sequence:
                # target = [self.output_dict[x] for x in pair[1]]
                
                # For classification:
                target = pair[1]

                mapped = self.input_mapper.compute(inp)
                structure = self.structure.make(mapped)
                for i in range(len(structure)):
                    structure[i] = self.semantical_mapper.compute(structure[i])
                filtered = self.attention.filter(structure)
                outputs = self.type_computer.compute(filtered)
                outputs = self.output_mapper.compute(outputs)

                # For normal sequence:
                # if len(outputs) >= len(target):
                #     for i in range(len(outputs)):
                #         if i < len(target):
                #             if outputs[i] != target[i]:
                #                 test_error_local += 1
                #         else:
                #             test_error_local += 1
                # else:
                #     for i in range(len(target)):
                #         if i < len(outputs):
                #             if outputs[i] != target[i]:
                #                 test_error_local += 1
                #         else:
                #             test_error_local += 1

                # test_error += test_error_local/len(target)

                # For classification:
                if len(outputs) != 0:
                    output = int(outputs[-1])
                else:
                    output = 0

                print "Output: " + str(output)
                print "Target: " + str(target)

                if output != int(target):
                    test_error += 1

            if (len(test_data) - test_error)/len(test_data) > self.best_fit:
                outfile = open(self.save_file, "wb")
                cPickle.dump(list(chromosome), outfile)
                outfile.close()

            print "Test error acc.: " + str((len(test_data) - test_error)/len(test_data)) \
                  # + " Num error: " + str(test_error)

        return (len(train_data) - error)/len(train_data)

    def evolve(self, n_generations):

        print "Initializing evolution..."

        # Genome instance
        setOfAlleles = GAllele.GAlleles()

        # Alleles for input_mapper
        for i in xrange(self.input_mapper.n_symbols):
            a = GAllele.GAlleleRange(0, self.input_mapper.n_symbols)
            setOfAlleles.add(a)

        # Alleles for structure
        for i in xrange(len(self.structure.r_map)):
            a = GAllele.GAlleleRange(0, self.n_objects)
            setOfAlleles.add(a)

        # Alleles for semantical_mapper
        for i in xrange(len(self.semantical_mapper.map)):
            a = GAllele.GAlleleRange(0, self.n_types)
            setOfAlleles.add(a)

        # Alleles for attention
        for i in xrange(self.max_attention_depth*self.max_attention_objects*2):
            a = GAllele.GAlleleRange(0, self.n_types)
            setOfAlleles.add(a)

        # Alleles for computer
        for i in xrange(len(self.type_computer.path)*len(self.type_computer.path[0])):
            a = GAllele.GAlleleRange(0, self.n_functions-1)
            setOfAlleles.add(a)

        # Alleles for output_mapper
        for i in xrange(self.n_types + 1):
            a = GAllele.GAlleleRange(0, self.output_mapper.n_symbols)
            setOfAlleles.add(a)

        genome = G1DList.G1DList(len(setOfAlleles))
        genome.setParams(allele=setOfAlleles)

        # The evaluator function (objective function)
        genome.evaluator.set(self.eval_func)
        genome.mutator.set(Mutators.G1DListMutatorAllele)
        genome.initializator.set(Initializators.G1DListInitializatorAllele)

        # Genetic Algorithm Instance
        ga = GSimpleGA.GSimpleGA(genome)
        ga.minimax = Consts.minimaxType["maximize"]
        ga.selector.set(Selectors.GRankSelector)
        ga.setGenerations(n_generations)

        print "Evolving..."

        # Do the evolution, with stats dump
        # frequency of 1 generations
        ga.evolve(freq_stats=1)

        print ga.bestIndividual()
