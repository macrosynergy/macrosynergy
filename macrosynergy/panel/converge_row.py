"""Convergence of rows for max weight"""
import numpy as np
import warnings

class ConvergeRow(object):
    """
    Class designed to receive a row of weights, where at least one weight in the
    aforementioned row exceeds the permitted upper-bound, and subsequently redistributes
    the excess evenly across all cross-sections.
    """

    def __init__(self, row, max_weight):
        """
        Class's Constructor.

        :param <np.ndarray> row: Array of weights.
        :param <Float> max_weight: Maximum weight.

        """
        self.row = row
        self.max_weight = max_weight
        self.flag = (1 / np.sum(self.row > 0)) <= self.max_weight
        self.margin = 0.001
        self.max_loops = 25

    @classmethod
    def application(cls, row, max_weight):

        cr = ConvergeRow(row=row, max_weight=max_weight)
        if cr.flag:
            cr.distribute_simple()
        else:
            cr.row = (cr.row > 0) / np.sum(cr.row > 0)
            cr.row[cr.row == 0.0] = np.nan

        return cr.row

    def distribute_simple(self):
        """
        Initiates an indefinite While Loop until the weights converge below the
        (max_weight + margin). Will evenly redistribute the excess weight across all
        active cross-sections, and will compute the maximum weight, the number of cross-
        sections above the threshold and the excess weight dynamically: through each
        iteration.

        """

        count = 0
        close_enough = False
        ar_weights = self.row.copy()

        while (count <= self.max_loops) and (not close_enough):
            count += 1
            excesses = ar_weights - (self.max_weight + self.margin)
            excesses[excesses <= 0] = 0
            ar_weights = (ar_weights - excesses) + np.nanmean(excesses)

            if np.max(ar_weights) <= (self.max_weight + self.margin):
                close_enough = True

        self.row = ar_weights