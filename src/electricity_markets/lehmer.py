'''
Created on 20.01.2022

@author: Fernando Penaherrera @UOL/OFFIS

A class for generation of pseudo-random numbers
'''

class LehmerRandom:
    """
    The LehmerRandom class implements a simple random number generator based on the Lehmer algorithm.
    This algorithm generates pseudo-random numbers between 0 and 1. The class provides methods to generate
    random floating-point numbers and random integers within a specified range.
    """

    def __init__(self, seed=1):
        """
        Initializes the LehmerRandom object with an optional seed value.

        Args:
            seed (int): Optional. The seed value used to initialize the random number generator. Default is 1 if no seed is provided.
        """
        self._m = 2 ** 31 - 1  # Modulus, a large prime number
        self._a = 48271       # Multiplier factor
        self._q = self._m // self._a  # m divided by a
        self._r = self._m % self._a   # Remainder when m is divided by a
        self._state = seed   # Current state or seed of the random number generator

    def _next_random(self):
        """
        Generates the next pseudo-random floating-point number in the range [0, 1).
        The internal state of the generator is updated after each call.

        Returns:
            float: A pseudo-random floating-point number in the range [0, 1).
        """
        hi = self._state // self._q
        lo = self._state % self._q
        test = self._a * lo - self._r * hi
        if test > 0:
            self._state = test
        else:
            self._state = test + self._m
        return self._state / self._m

    def random(self):
        """
        Generates a pseudo-random floating-point number in the range [0, 1).

        Returns:
            float: A pseudo-random floating-point number in the range [0, 1).
        """
        return self._next_random()

    def randint(self, low, high):
        """
        Generates a pseudo-random integer between `low` and `high`, inclusive.

        Args:
            low (int): The lower bound of the range (inclusive).
            high (int): The upper bound of the range (inclusive).

        Returns:
            int: A pseudo-random integer between `low` and `high`.
        """
        return low + int(self._next_random() * (high - low + 1))

    def seed(self, seed):
        """
        Sets the seed value of the random number generator. The seed can be used to reproduce a specific sequence of random numbers.

        Args:
            seed (int): The new seed value for the random number generator.
        """
        self._state = seed
