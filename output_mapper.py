"""Maps outputs of TypeComputer to the appropriate domain of output symbols.

   Use: Initialize with the number of types in TypeComputer output, and a string containing all symbols in the output
        domain. Call map() with vector of new mapping to update. Pass data to compute() to perform mapping on data.

   Note: This should totally just inherit from SemanticalMapper, or something.
"""

import random


class OutputMapper():

    def __init__(self, n_types, inputs):
        self.n_types = n_types

        symbol_set = set()

        for symbol in inputs:
            symbol_set.add(symbol)

        self.n_symbols = len(symbol_set)
        self.map = {x: random.randint(0, self.n_symbols) for x in range(n_types + 1)}

    def set(self, new_mapping):
        """
        :param new_mapping: :type list: A list of size n_types defining the new mapping. The elements should all be ints
                                        in the range [0, n_symbols).
        :return: None
        """
        self.map = {x: new_mapping[x] for x in range(self.n_types + 1)}

    def compute(self, data):
        """
        :param data: :type iterable: 1-d data to be mapped according to the current mapping. Data must be integers in
                                     range [0, n_symbols).
        :return: :type list: A new list of the mapped data.
        """
        return [self.map[data[x]] for x in range(len(data))]
