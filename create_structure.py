"""Class defining a symbol structure.

    Use: Pass 1-d data (an iterable) to make() to create the structure,
         then call get() to return the structure.
"""

import itertools
import random as r


class Structure:

    def __init__(self, n_objects):
        """
        :param n_objects: Number of symbols to map input data to in structure.
        :return:
        """
        pairs = itertools.product([x for x in range(n_objects+1)], [x for x in range(n_objects+1)])
        sets = {frozenset(x) for x in pairs}
        self.r_map = {x: r.randint(0, n_objects) for x in sets}

    def make(self, data):
        """
        :param data: :type iterable: Data to be converted to structure. Must be 1-d.
        :return: None
        """
        structure = []
        dim = data
        while len(dim) > 1:
            dim_temp = []
            for i in range(len(dim) - 2):
                pair = frozenset([dim[i], dim[i + 1]])
                dim_temp.append(self.r_map[pair])
            dim = dim_temp
            structure.append(dim_temp)
        return structure

    def set(self, symbol_list):
        """
        :param symbol_list: A list of size r_map defining the new mapping.
        :return:
        """
        i = 0
        for key, value in self.r_map.iteritems():
            self.r_map[key] = symbol_list[i]
            i += 1
