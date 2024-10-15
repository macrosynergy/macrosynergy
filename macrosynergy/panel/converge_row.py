"""Convergence of rows for max weight"""

import numpy as np


class ConvergeRow(object):
    """
    Class designed to receive a row of weights, where at least one weight in the
    aforementioned row exceeds the permitted upper-bound, and subsequently redistributes
    the excess evenly across all cross-sections.
    :param <np.ndarray> row: Array of weights.
    :param <Float> max_weight: Maximum weight.
    :param <Float> margin: Margin of error allowed in the convergence to within the
        upper-bound, "max_weight".
    :param <Integer> max_loops: Controls the accuracy: in theory, the greater the
        number of loops allowed, the more accurate the convergence. However, will only
        become significant if a tight margin is imposed: the "looser" the margin, the less
        likely the maximum number of loops permitted will be exceeded.
    """

    def __init__(
        self,
        row: np.ndarray,
        max_weight: float,
        margin: float = 0.001,
        max_loops: int = 25,
    ):
        self.row = row
        self.max_weight = max_weight
        self.flag = (1 / np.sum(self.row > 0)) <= self.max_weight
        self.margin = margin
        self.max_loops = max_loops

    @classmethod
    def application(cls, row: np.ndarray, max_weight: float):
        cr = ConvergeRow(row=row, max_weight=max_weight)

        if cr.flag:  # if enough non-nan values in row
            cr.distribute_simple()
        else:  # apply equal weights to all non-nan cases
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

        while (count <= self.max_loops) and (not close_enough):
            count += 1
            excesses = self.row - (self.max_weight + self.margin)
            excesses[excesses <= 0] = 0
            self.row = (self.row - excesses) + np.nanmean(excesses)

            if np.max(self.row) <= (self.max_weight + self.margin):
                close_enough = True
