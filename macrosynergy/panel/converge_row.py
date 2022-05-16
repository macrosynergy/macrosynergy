"""Convergence of rows for max weight"""
import numpy as np

class ConvergeRow(object):
    """
    Class designed to receive a row of weights, where at least one weight in the
    aforementioned row exceeds the permitted upper-bound, and subsequently redistributes
    the excess evenly across all cross-sections.
    """

    def __init__(self, weights, max_weight, margin=0.001, max_loops=25):
        """
        Class's Constructor.

        :param <np.ndarray> row: Array of weights.
        :param <Float> max_weight: Maximum that a weight can be.
        :param <Float> margin: Margin of error allowed in the convergence to within the
                                upper-bound, "max_weight".
        :param <Integer> max_loops: Controls the accuracy: in theory, the greater the
                                    number of loops allowed, the more accurate the
                                    convergence. However, will only become significant if
                                    a tight margin is imposed: the "looser" the margin,
                                    the less likely the maximum number of loops permitted
                                    will be exceeded.
        """
        self.weights = weights
        self.max_weight = max_weight
        self.margin = margin
        self.max_loops = max_loops

    @classmethod
    def application(cls, weights: np.ndarray, max_weight: float) -> np.ndarray:
        """
        Given weights and the maximum that the weight can be, it returns the new
        computes the new distribution of weights.
        """
        cr = ConvergeRow(weights, max_weight)
        cr.compute_new_weights()
        return cr.weights

    def compute_new_weights(self):
        if self.able_to_distribute_excess_weight():
            self.distribute_excess_weight()
        else:
            self.equally_redistribute_weights()

    def able_to_distribute_excess_weight(self) -> bool:
        """
        Checks to see if we are able to dynamically redistribute excess weights
        of the active cross-sections of weights.
        """
        return (1 / np.sum(self.weights > 0)) <= self.max_weight

    def distribute_excess_weight(self):
        """
        Dynamically redistributes excess weight across the active cross-sections of the weights 
        until the weights are within the tolerance or the maximum loops has occured.
        """

        for _ in range(self.max_loops):
            excesses = self.weights - (self.max_weight + self.margin)
            excesses[excesses <= 0] = 0
            self.weights = (self.weights - excesses) + np.nanmean(excesses)

            if self.within_tolerance():
                break

    def within_tolerance(self) -> bool:
        return np.max(self.weights) <= (self.max_weight + self.margin)

    def equally_redistribute_weights(self):
        """
        Redistributes active cross-section of weights equally i.e.
        [0.3, 0.7, -0.1] -> [0.5, 0.5, -0.1] 
        """
        self.weights = (self.weights > 0) / np.sum(self.weights > 0)
        self.weights[self.weights == 0.0] = np.nan