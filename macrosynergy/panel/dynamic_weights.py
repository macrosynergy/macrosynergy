
## A maximum weight is applied to the weight matrix: if any of the weights, on a particuliar timestamp, exceed the maximum value exonogenously imposed, redistribute the excess weight across all the cross-sections until all weight allocations are within the margin. 
## The margin, as referenced above, is endogenously imposed and represents the error allowed on the weight allocation: if the weighting exceeds the maximum value but is within the margin of error, the weight allocated is permitted and the convergence function is terminated.

## Each cross-section will have an expected weight, which is calculated by determining the number of active cross sections, for that timestamp, and dividing one by the number to obtain the mean, and if the maximum weight imposed is less than the expected weight, apply the expected weight to all cross-sections.
## If the maximum weight is less than the expectation, it is not possible to redistribute the weights such that the entire row complies with the upper bound.
## Therefore, use the expected weight for all cross-sections as there will be a lower variance with regard to the maximum weight.


## Design an algorithm that produces the convergence feature: redistribution.
## Calculate the weights, according to the chosen algorithm, and subsequently apply the redistribution method.

import numpy as np
import pandas as pd
from random import shuffle
import time
import matplotlib.pyplot as plt


def dataframe(no_rows, no_columns):

    elements = no_rows * no_columns
    data = np.linspace(1, elements, elements, dtype = np.float32)
    midpoint = int(elements // 2)
    
    shuffle(data)

    ## Transform to a matrix.
    data = data.reshape((no_rows, no_columns))
    cols = ['cols' + str(i + 1) for i in range(no_columns)]

    length = lambda divisor: int(round(no_rows / divisor, ndigits = 0))
    
    data[0:length(10), 0] = np.nan
    midpoint = int(no_columns // 2)
    data[0:2, midpoint] = np.nan
    data[0:length(20), -1] = np.nan

    data[-3:, 0] = np.nan
    data[-5:, 1] = np.nan

    data[int(no_rows // 2), :] = np.nan ## Universal holiday that would effect all cross-sections.
    df = pd.DataFrame(data = data, columns = cols) 

    return df


## The weighting method used will be the inverse of the rolling volatility: the more volatile the cross-section, the lower the weight allocated on that timestamp.
## It is worth noting that stocks suffer from sustained periods of high & low volatility: Conditional Heteroskedasticity. Therefore, the influence of each cross-section, on the basket's performance, may fluctuate considerably over the time-period analysed.
def flat_std(x: np.ndarray, remove_zeros: bool = True):
    if remove_zeros:
	    x = x[x != 0]
    mabs = np.mean(np.abs(x))
    return mabs


## Inverse Volatility.
def w_matrix(df, lback_periods = 21):

    ## Will return NaNs until the number of periods is equal to the window size.
    dfwa = df.rolling(window = lback_periods).agg(flat_std, True)

    rolling_arr = dfwa.to_numpy(dtype = np.float32)
    inv_arr = 1 / rolling_arr
    inv_arr = np.nan_to_num(inv_arr, copy = False, nan = 0.0)

    sum_arr = np.sum(inv_arr, axis = 1)
    sum_arr[sum_arr == 0.0] = np.nan
    inv_arr[inv_arr == 0.0] = np.nan

    sum_arr = sum_arr[:, np.newaxis]

    rational = np.divide(inv_arr, sum_arr)

    return rational

def active_days(rational):

    nan_bool = np.isnan(rational)
    nan_bool = ~nan_bool ## Take the inverse.
    
    nan_bool = nan_bool.astype(dtype = np.uint8)
    days = np.sum(nan_bool, axis = 1)

    return days

## Diluting the influence of the weight that exceeds the threshold, and consequently altering the weighting balance between cross-sections.
## Aim to redistribute the excess weight, above the maximum threshold, evenly across the cross-sections. The reason for the even redistribution is to preserve the weighting proportions calculated using the inverse method. The weights are not arbitrarily determined but instead computed according to a specific method that reflecs one's investment guidelines.
class converge_row(object):

    def __init__(self, row, m_weight, active_cross, r_length):
        self.row = row
        self.maximum = self.max_row()
        self.m_weight = m_weight
        self.active_cross = active_cross
        ## Linear in the size of the row which, in computational terms, will be comparatively small.
        self.m_index = self.max_index()
        self.margin = 0.001 ## Allowance above the threshold.
        
    def distribute(self):

        ## The instance's fields will evolve with the While Loop. The row dynamically updates.
        while True:

            excess = self.max_row() - self.m_weight

            excess_cross = self.excess_count()
            indexing = self.index_row()
            
            if excess_cross.size == 1:
                self.row[self.m_index] -= excess
                
            elif excess_cross.size > 1:
                for index in excess_cross:
                    indexing[index] = self.m_weight
                    
                self.row = np.array(list(indexing.values())) ## Reconfiguring the row.
                excess = 1.0 - np.nansum(self.row)
            else:
                break

            amount = excess / self.active_cross
            ## Redistribute the excess weight evenly across all cross-sections.
            self.row += amount
            

    def max_row(self):
        return np.nanmax(self.row)

    def excess_count(self):
        row_copy = self.row.copy()

        return np.where((row_copy - self.m_weight) > 0.001)[0]

    def max_index(self):
        return np.where(self.row == self.maximum)[0][0]

    ## Logarithmic performance on any search or insertion operation, as dictionaries are implemented using a Hashmap.
    def index_row(self):
        return dict(zip(range(self.row.size), self.row))


## Understand the performance of the designed algorithm. The designed algorithm is occurring within an iterative Loop, and consequently the overall performance will likely converge to an expensive polynomial under asymptotic conditions.
## A While Loop nested within a linear operation: the convergence speed is predicated on the margin of error.
def test_row():

    weight_threshold = 0.3
    row_1 = np.array([0.35, 0.29, np.nan, 0.25, 0.11], dtype = np.float64)
    inst = converge_row(row_1, weight_threshold, 4, row_1.size)

    ## Implausible given the weighting mechanism but not impossible.
    row_2 = np.array([np.nan, 0.33, 0.32, 0.30, 0.05], dtype = np.float64)
    print(row_2)
    inst_2 = converge_row(row_2, weight_threshold, 4, row_2.size)
    inst_2.distribute()
    print(inst_2.row)

    ## A weighting distribution, in the context of the inverse standard deviation method, that allocates the majority of the weight to a single cross-section suggests a basket of highly volatile occupants and a singular stable cross-section over the duration analysed.
    ## Therefore, to avoid a highly volatile basket, transfer the weight to the stable cross-section.
    row_3 = np.array([0.67, 0.09, 0.13, np.nan, 0.11])
    print(row_3)
    inst_3 = converge_row(row_3, weight_threshold, 4, row_3.size)
    inst_3.distribute()
    print(inst_3.row)
        

def max_weight(rational, days, max_weight):

    ## By using the inverse standard deviation to compute the weights, the days up until the window has first been populated will consist of NaN values. Therefore, remove the empty rows from the output dataframe.
    nan_rows = np.where(days == 0)[0]

    ## Check an iterative sequence is formed.
    start = nan_rows[0]
    end = nan_rows[-1]
    iterator = np.array(range(start, end + 1))

    bool_size = (iterator.size == nan_rows.size)
    if bool_size and np.all(iterator == nan_rows):
        rational = rational[(end + 1):, :]
        days = days[(end + 1):]
    else:
        rational = np.delete(rational, tuple(nan_rows), axis = 0)
        days = np.delete(days, tuple(nan_rows))

    ## Use the number of days to determine the expected weight for each cross section across every timestamp.
    uniform = 1 / days
    fixed_indices = np.where(uniform > max_weight)[0]

    ## Purposefully constructed Array to determine which rows will require equally weighted values.
    rows = rational.shape[0]
    bool_arr = np.zeros(rows, dtype = bool)
    bool_arr[fixed_indices] = True

    ## Linear in the number of timestamps.
    for i, row in enumerate(rational):
        if bool_arr[i]:
            row = np.ceil(row)
            row = row * uniform[i]
            rational[i, :] = row
        else:
            inst = converge_row(row, max_weight, days[i], row.size)
            inst.distribute()
            rational[i, :] = inst.row

    return rational

## Testing the performance under asymptotic conditions: difficult to assign a defined function given the indefinite loop and the dependence on the allowed margin.
def performance_test():

    input_ = list(range(1000, 20000, 500))
    ## The number of cross-sections, given the possible range is small in computational terms, will not effect performance.
    no_columns = 5

    performance = []
    for i, rows in enumerate(input_):
        print(i)
        df = dataframe(rows, no_columns)
        rational = w_matrix(df, lback_periods = 21)
        days = active_days(rational)
        start = time.time()
        weight_matrix = max_weight(rational, days, 0.3)
        w_timed = time.time() - start
        performance.append(w_timed)
    
    ax = plt.subplot(1, 1, 1)
    p1 = ax.plot(input_, performance)
    
    plt.legend(loc = "upper left")

    plt.xlabel("Timestamps: number of data points.")
    plt.ylabel("Computational Performance.")
    plt.show()
    
        
## Utilise Numerical Computing for storing large, homogeneous Arrays of multidimensional data where mathematical functions & manipulations are applied.
## Numpy will exploit Vectorisation enabling numerical functions to operate on a section of inputs "concurrently" through the behaviour of the Cache. A Numpy Array will host homogenous data, held in a continuous block, which allows the Compiler to understand the memory requirements prior to computation. 
## The importance of a continuous block of memory is it allows for Spatial Locality: when the compiler accesses the first index of an Array from the RAM, it will "grab" a segment of adjacent memory and store the whole segment in the Cache, normally 32 Bytes. If using 16 Bit floating point values, it will save 15 "trips" to the RAM which is an expensive operation.
## Fetching an object from the RAM, which equates to a large, amorphous block of memory, can be 100x slower than the Cache which is a stratified, ordered block.
## The homogeniety of a Numpy Array is significant as it reduces the number of checks the Compiler must complete by a factor of n: the size of the data structure. Python is a dynamically-typed language, and consequently the Compiler is required to iterate through each element to determine the type and whether the operation is permissable.
## Further, the Numpy functionality is a wrapper to C code which is significantly quicker at completing any iterative operation: function is applied to a sample of data.
def MAIN():

    df = dataframe(135, 4)
    rational = w_matrix(df, lback_periods = 11)
    days = active_days(rational)
    
    weight_matrix = max_weight(rational, days, 0.3)
    print(weight_matrix)
    

if __name__ == "__main__":

    MAIN()
