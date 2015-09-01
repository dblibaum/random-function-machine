"""This class defines the mapping between objects and types.

    Use: Initialize with the number of objects. To define a new mapping, pass an integer list of size equal to the
         number of objects to map(). To perform a mapping on data, pass an iterable of data to compute(). The data
         should be integers in the appropriate range (see method). If this is the initial mapping from raw data,
         initialize with the option first_layer=True and set input to a string that contains all symbols present in the
         input data (e.g. all of the input data).
"""

import random as r


class SemanticalMapper:

    def __init__(self, n_objects=0, n_types=0, first_layer=False, inputs=[]):
        self.n_symbols = n_objects
        self.first_layer = first_layer
        self.symbol_set = []
        self.map = {}
        self.first_layer = first_layer
        if first_layer:
            symbol_set = set()

            for symbol in inputs:
                symbol_set.add(symbol)

            self.symbol_set = list(symbol_set)

            self.n_symbols = len(symbol_set)
            self.map = {x: r.randint(0, self.n_symbols) for x in self.symbol_set}
        else:
            self.map = {x: r.randint(0, n_types) for x in range(self.n_symbols + 1)}

    def set(self, new_mapping):
        """
        :param new_mapping: :type list: A list of size n_objects defining the new mapping. In the case of first layer,
                                        a list of size equal to number of symbols in input.
        """
        if self.first_layer:
            i = 0
            for key, value in self.map.iteritems():
                self.map[key] = new_mapping[i]
                i += 1
        else:
            self.map = {x: new_mapping[x] for x in range(self.n_symbols + 1)}

    def compute(self, data):
        """
        :param data: :type iterable: 1-d data to be mapped according to the current mapping. Data must be integers in
                                     range [0, n_objects).
        :return: :type list: A new list of the mapped data.
        """
        if self.first_layer:
            return [self.map[data[x]] for x in range(len(data)) if data[x] in self.symbol_set]
        return [self.map[data[x]] for x in range(len(data))]
