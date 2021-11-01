"""Convergence of rows for max weight"""
import numpy as np
import warnings
MAX_COUNT = 1000


class ConvergeRow(object):
    """
    Class designed to receive a row of weights, where at least one weight in the
    aforementioned row exceeds the permitted upper-bound, and subsequently redistributes
    the excess evenly across all cross-sections. The reason for the even distribution is
    to retain the weighting proportions received: the weights are not arbitrary but
    instead determined according to a specific method. Therefore, the Class will simply
    aim to dilute the influence any cross-section above the maximum amount but retain
    the ratio between the remaining weights.
    """

    def __init__(self, row, max_weight):
        """
        Class's Constructor.

        :param <np.ndarray> row: Array of weights.
        :param <Float> max_weight: Maximum weight.

        """
        self.row = row
        self.r_length = row.size
        self.maximum = self.max_row()
        self.max_weight = max_weight
        self.act_cross = np.sum(self.row > 0)
        self.uniform = 1 / self.act_cross
        self.flag = self.uniform <= self.max_weight
        self.m_index = self.max_index()
        self.margin = 0.001  # Todo: set as function parameter
        self.max_loops = 25  # Todo: set as function parameter

    @classmethod
    def application(cls, row, max_weight):

        cr = ConvergeRow(row=row, max_weight=max_weight)
        if cr.flag:
            cr.distribute_simple()  # if cap is above equal weight re-distribute sequentially
        else:
            cr.row = (cr.row > 0) / np.sum(cr.row > 0)  # Todo: check if valid equal weights and ditch fixed_val
        return cr, cr.row   # Todo: check if cr output is redundant

    def distribute_simple(self):

        count = 0
        close_enough = False
        ar_weights = self.row.copy()  # array with NaNs
        while (count <= self.max_loops) & (not close_enough):
            count += 1
            excesses = ar_weights - self.max_weight
            excesses[excesses <= 0] = 0
            ar_weights = ar_weights - excesses + np.mean(excesses)  # Todo: make sure mean respects NaNs
            if np.max(ar_weights) <= (self.max_weight + self.margin):
                close_enough = True

        self.row = ar_weights

    def distribute(self):
        """
        Initiates an indefinite While Loop until the weights converge below the
        (max_weight + margin). Will evenly redistribute the excess weight across all
        active cross-sections, and will compute the maximum weight, the number of cross-
        sections above the threshold and the excess weight dynamically: through each
        iteration.

        """
        count = 0
        while True:
            count += 1
            if count > MAX_COUNT:  # #  Tod: Why is MAXCOUNT not defined in class
                self.row *= np.nan
                warnings.warn("Unable to converge.")  #  Todo: collect non converging indices and print as single warning
                break

            excess = self.max_row() - self.max_weight  # excess of largest number in row

            excess_cross = self.excess_count()  # number of cross sections that excess threshold
            indexing = self.index_row()  # complete dictionary with keys as indices of and values as weights
            
            if excess_cross.size == 1:
                self.row[self.m_index] -= excess  # only index of single excess value is reduced
                
            elif excess_cross.size > 1:
                for index in excess_cross:  # loop through indices of excessive values
                    indexing[index] = self.max_weight  # replace excessive weights with cap
                    
                self.row = np.array(list(indexing.values()))  # replace row values with capped/unchanged values
                excess = 1.0 - np.nansum(self.row)  # total excess that has been taken off
            else:
                break

            amount = excess / self.act_cross
            self.row += amount  # Todo: check that only NaNs not zeroes

    def fixed_val(self):
        """
        Convert array to equal weights for all non-NA values.
        """
        row = np.ceil(self.row)  # converts all bon-zero values to one
        self.row = row * self.uniform

    def max_row(self):
        """
        Determines the maximum weight in the row. Will return the weight value.

        """
        return np.nanmax(self.row)

    def excess_count(self):
        """
        Calculates the number of weights that exceed the threshold. This can change
        through each iteration, so the calculation occurs dynamically.

        Will return an Array of the indices that satifsy the condition.
        """

        row_copy = self.row.copy()

        return np.where((row_copy - self.max_weight) > 0.001)[0]

    def max_index(self):
        """
        Will return the index of the maximum value. Integer.

        """
        return np.where(self.row == self.maximum)[0][0]

    def index_row(self):
        """
        Function used to design a dictionary where the keys are indices of the row, and
        the values are the weights of the row. Dictionaries are supported by a data
        structure called a Hashmap and are subsequently computationally very efficient
        assuming an effective Hash Function has been designed. Potentially O(1) look-up
        time if collisions are avoided.

        Returns a dictionary.
        """
        return dict(zip(range(self.r_length), self.row))


if __name__ == "__main__":

    row = np.array([0.5, 0.2, 0.2, 0.1, 0])
    max = 0.3
    ConvergeRow.application(row, max_weight=max)