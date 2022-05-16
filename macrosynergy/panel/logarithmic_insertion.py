
from random import random, shuffle
from math import log
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Construct an algorithm which achieves logarithmic insertion on a sorted data structure.
class Node(object):
    """
    Class to implement a Node, data point.
    """

    def __init__(self, key, level):
        self.key = key

        # Sentinel: List to hold references to nodes positioned on different levels.
        # There will always be a Base Level that represents a Linked List hosting all the
        # elements.
        self.forward = [None] * (level + 1)

class SkipList(object):

    """
    Class for Skip List.
    """

    def createNode(self, lvl, key):
        n = Node(key, lvl)
        return n

    # The parameter, max_lvl, is the upper bound on the number of levels in the
    # Skip List. In theory, it is an arbitrary number because a truly randomised
    # algorithm would not have an upper bound.
    # If memory capacity is not significant or a concern, an upper bound is not required.
    def __init__(self, series_length, P):
        self.MAXLVL = int(round(log(series_length, 2), ndigits = 0)) + 1

        # The second parameter is the fraction of the nodes with level i references
        # whilst having (i + 1) references. The value, axiomatically, will be a
        # probability, and in the majority of cases equal to 0.5.
        self.P = P

        # Sentinel. Level, key. Will only be instantiated once on the formation of the
        # Sentinel.
        self.header = self.createNode(self.MAXLVL, -1)

        # Current level of the Skip List.
        self.level = 0

        # Additional field incorporated specifically for determining the median element.
        self.size = 0

    # Equivalent to the Binomial Distribution: sequence of Bernoulli Random Variables.
    def randomLevel(self):
        lvl = 0
        # Imposing an upper bound on the number of possible levels which, in a purely
        # random algorithm, is not applicable.
        while random() < self.P and lvl < self.MAXLVL:
            lvl += 1

        return lvl

    # Insert a given key into the Skip List in O(log(n)) time.
    def insertElement(self, key):
        self.size += 1

        # Insert the element under the provision that it could be included in every level
        # which is the least preferred outcome.
        # The List, "update", records the Pointers that will be directed to the newly
        # inserted node, and allows for understanding where the new node's Pointers will
        # extend to.
        update = [None] * (self.MAXLVL + 1)

        # Start from the Sentinel and find the next active level.
        current = self.header

        # Commence from the "top" level which will host the fewest nodes.
        for i in range(self.level, -1, -1):
            # The Sentinel will initially be populated with NaN values.
            # The number of comparisons made on each level, to determine the index,
            # should fall by half. Each level reduces the sample space of possible
            # elements to assess over.
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]

            update[i] = current # Track the preceding element.

        current = current.forward[0]
        if current == None or current.key != key:

            rlevel = self.randomLevel()
            if rlevel > self.level:
                for i in range(self.level + 1, rlevel + 1):
                    # The pointer will be directed straight back to the Sentinel, as
                    # the current node extends beyond all the preceding nodes.
                    update[i] = self.header

                self.level = rlevel

        n = self.createNode(rlevel, key)

        for i in range(rlevel + 1):
            # Update the recently instantiated node's Pointer System: the forward field
            # references its connection with the following node.
            n.forward[i] = update[i].forward[i]
            # Update the Pointer System from the standpoint of the Sentinel.
            # On the instance of the SkipList, it is only important understanding the
            # first node on each level.
            update[i].forward[i] = n

    def searchElement(self, key):
        current = self.header

        for i in range(self.level, -1, -1):
            while (current.forward[i] and current.forward[i].key < key):
                # Check the intermediary values.
                current = current.forward[i]

        current = current.forward[0]
        if current and current.key == key:
            print(f"Found key: {key}.")

    # Conceptually the data structure has already been ordered but in form of the Pointer
    # System. Therefore, pass through the Pointers up to the median value.
    # Under asymptotic conditions O(n / 2)  is equivalent to O(n), as infinity divided by
    # two remains infinity. However, under finite conditions, the difference can be
    # salient.
    # O(n(log(n)) + n^2).
    # Obtaining the median value n times.
    def median_search(self):

        head = self.header # Sentinel.

        node = head.forward[0]
        count = 0
        flag = (self.size % 2 == 0)
        midpoint = self.size // 2
        if flag: midpoint -= 1

        while (node != None):

            if count == midpoint:
                if not flag:
                    return node.key
                else:
                    nxt_elem = node.forward[0]
                    return (node.key + nxt_elem.key) / 2

            node = node.forward[0]
            count += 1

    # The data is held in a network of Pointers, as opposed to an iterable sequence.
    # The expensive linear operation can be amortised if a high number of insertions
    # occur for each display of the data structure.
    def sorted_series(self):

        data = []
        head = self.header # Sentinel.

        # Iterate along the Linked List: all elements will be held on the base level.
        # The exterior node's Pointer will be directed to the Null Pointer.
        node = head.forward[0]
        # Linear time algorithm: iterating along the base level which acts as a Linked
        # List possessing every element, O(n).
        while (node != None):
            data.append(node.key)

            node = node.forward[0]

        return data

# Driver to test the above code.
def sorted_perf():

    r_series = list(range(1, 10000))
    shuffle(r_series)

    # The Skip List approach is expected to be computationally slower:
    # O(n(log(n))) + O(n). It will be slower by a linear factor.
    start_skip = time.time()
    length = len(r_series)
    skip_l = SkipList(length, 0.5)

    for elem in r_series:
        skip_l.insertElement(elem)

    order_series = skip_l.sorted_series()
    challenger = time.time() - start_skip

    # Performance will roughly be O(n(log(n))), as it is implemented using randomised
    # algorithms.
    start_list = time.time()
    r_series = sorted(r_series)
    base = time.time() - start_list

    assert(r_series == order_series)

    return challenger, base

def array_assemble(days, cross_sections):

    data = np.zeros((days, cross_sections))
    col = np.linspace(1, days, days)
    col = col.astype(dtype=np.float32)
    shuffle(col)
    data[:, 0] = col

    for i in range(cross_sections - 1):
        val_range = days * (i + 1)
        col = data[:, 0] + val_range
        shuffle(col)
        data[:, (i + 1)] = col

    return data

def rolling_median_perf(column_number):

    # The greater the number of columns, the greater the role of amortisation.
    # Replicate cross-sectional data.
    data = array_assemble(days = 3000, cross_sections = column_number)
    no_columns = data.shape[1]

    start = time.time()
    rows = 3000
    dates = pd.date_range(start = "2000-01-01", periods = rows, freq = 'd')
    df = pd.DataFrame(data = data, index = dates)

    median_pd = np.array([df[0:(j + 1)].stack().median() for j in range(rows)])
    time_pd = time.time() - start

    # The current naive design will achieve a performance of O(n(log(n)) + n^2).
    # The requirement of iterating through the Linked List for each insertion will have a
    # detrimental impact on performance.
    start_skip = time.time()

    no_elems = rows * no_columns
    # Flatten the pd.DataFrame to subsequently insert each element into the Skip List.
    # Compute the median value according to the column dimension of the original
    # DataFrame: insert each element and at a regulated interval, according to the number
    # of columns, access the median value.
    data_one_dimension = data.reshape((no_elems, 1))
    r_median = []

    skip_l = SkipList(no_elems, 0.5)
    count = 0
    for elem in data_one_dimension:
        count += 1

        skip_l.insertElement(elem)
        if count == no_columns:
            m = skip_l.median_search()
            r_median.append(m)
            count = 0

    time_skip = time.time() - start_skip

    r_median = np.array(r_median)
    r_median = np.squeeze(r_median, axis = 1)

    assert(np.all(median_pd == r_median))

    return time_skip, time_pd

def performance():

    no_columns = list(range(1, 30))
    skip_log = []
    pd_benchmark = []
    for elem in no_columns:

        print(f"Element: {elem}.")
        time_skip, time_pd = rolling_median_perf(elem)
        skip_log.append(time_skip)
        pd_benchmark.append(time_pd)

    ax = plt.subplot(1, 1, 1)
    ax.plot(no_columns, skip_log, label = "SkipList")
    ax.plot(no_columns, pd_benchmark, label="Computational Benchmark")
    plt.legend(loc = "upper left")

    plt.xlabel("Number of cross-sections.")
    plt.ylabel("Computational Speed of both Algorithms.")

    plt.show()

def MAIN():

    performance()


if __name__ == "__main__":
    MAIN()