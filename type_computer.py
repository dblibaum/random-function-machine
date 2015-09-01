"""Defines path of types through functions as a series of vectors that direct the output of each function to the next.
   Contains a symbol that indicates output of type.
"""

import random as r


class TypeComputer:

    def __init__(self, max_input, num_functions, depth, n_types):
        """
        :param max_input: :type int: Maximum number of inputs to computer (max returned by attention).
        :param num_functions: :type int: Number of functions in function bank.
        :param depth: :type int: Number of times symbols are passed through function bank.
        """

        self.max_input = max_input
        self.num_functions = num_functions
        self.depth = depth
        self.path = [[r.randint(0, num_functions-1) for x in range(max_input)] for y in range(depth)]
        self.functions = [{x: r.randint(0, n_types) for x in range(n_types + 1)} for y in range(num_functions)]

    def compute(self, symbols):
        """
        :param symbols: :type iterable: An iterable of input symbols, should be of size self.num_input.
        :return: o_vector: :type list: The outputs of the computation.
        """

        o_vector = []
        for layer in self.path:
            i = 0
            for mapping in layer:
                if i < len(symbols) - 1:
                    symbol = self.functions[mapping][symbols[i]]
                    if symbol == 0:  # At 0 the last symbol is output
                        o_vector.append(symbols[i])
                    else:
                        symbols[i] = symbol
                i += 1

        return o_vector

    def set(self, new_path):
        """Sets a new path for the computer.

        :param new_path: List of size num_input*depth
        """

        for i in range(self.depth):
            self.path[i] = new_path[self.max_input*i:self.max_input*(i + 1)]
