
import time
import numpy as np
from random import choice, randrange
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt


def slide(arr, w, s = 1):

    return np.lib.stride_tricks.as_strided(arr,
                                           shape = ((len(arr) - w) + 1, w),
                                           strides = arr.strides * 2)[::s]

def swap(bench_row_, row_, index_b, index_e):
    
    row_[index_e] = bench_row_[index_b]
    return row_

## Same purpose as the above subroutine but with the additional feature of correcting the order.
## Achieve the correct order.
def swap_shift(bench_row_, row_, index_b, index_e, window):

    row_[index_e] = bench_row_[index_b]
    value = row_[index_e]
    output = np.concatenate((np.array([value]), row_[:index_e]))
    output = np.concatenate((output, row_[(window - index_b):]))
    
    return output
    

## The algorithm is still under trial. Will confirm once complete.
def zero_shift(arr_window, window):

    final_col = arr_window[:, (window - 1)]
    zero_ret = np.where(final_col == 0.0)[0]
    prev_ind = (zero_ret - 1)
    arr_copy = arr_window.copy()

    ## Unavoidable polynomial algorithm but the internal operations are minimal, so the performance is not comprised too greatly.
    ## The interior loop's computational performance will be O(Window): the iterator is determined by the window size with the internal operations taking constant time, as the speed of slicing does not change with the size of the Array.
    for i, elem in enumerate(prev_ind):

        bench_row = arr_copy[elem]
        start = zero_ret[i]
        index_b = 0
        index_e = -1
        for index in range(start, start + window):
            if (index + 1) > arr_window.shape[0]:
                break
            o_row = arr_copy[index]
            _row = swap_shift(bench_row, o_row, index_b, index_e, window)
            arr_copy[index] = _row
            index_b += 1
            index_e -= 1

    return arr_copy

def rolling_estimate(series: pd.Series, window: int):
    s = series[series != 0]

    s_arr = s.to_numpy()
    rolw = deque(s_arr[:window], maxlen = window)
    std_calc = []
    i = window - 1
    rolw_arr = np.array(rolw)
    std_calc.append((s.index[i], np.std(rolw_arr)))
    
    for elem in s_arr[window:]:
        rolw.append(elem)
        i += 1
        
        rolw_arr = np.array(rolw)
        std_calc.append((s.index[i], np.std(rolw_arr)))

    out = pd.DataFrame(std_calc, columns = ["date", "values"]).set_index("date")
    ## Concatenate Pandas Objects along a particular axis with optional set logic along the other axes.
    ## Create a DataFrame with a column containing the index. Concatenate along the column axis.
    out = pd.concat((out, series.index.to_frame(name = "check")), axis = 1)
    out.drop(["check"], axis = 1, inplace = True)
    out = out.squeeze() ## Squeeze 1 dimensional axis objects into scalars.

    out = out.fillna(method = "ffill", limit = 5) ## The parameter "ffill" will propagate the last valid observation forward to next valid backfill.
    assert out.shape[0] == series.shape[0]
    
    return out 

def zeroInfill(values):
    values[6:8] = 0.0
    values[12] = 0.0
    values[71] = 0.0
    values[114:117] = 0.0
    values[191] = 0.0
    values[212:215] = 0.0
    values[711:714] = 0.0
    values[781] = 0.0

def slide_std(arr, dates, w):

    o_data = np.zeros(len(dates), dtype = np.float64)
    std_arr = np.std(arr, axis = 1)

    o_data[:(w - 1)] = np.nan
    o_data[(w - 1):] = std_arr

    out = pd.Series(index = dates, data = o_data)
    
    return out

def number_generator(n):
    np.random.seed(0)
    return np.random.randn(n)

def input_modification(values, n, window):

    values = values.astype(dtype = np.float64)
    dates = pd.date_range(start = "1999-01-01", periods = n, freq = "d")

    arr_window = slide(values, window)
    arr_output = zero_shift(arr_window, window)

    s_std = slide_std(arr_output, dates, window)
    
    return s_std


def output_modification(values, n, window):
    
    dates = pd.date_range(start = "1999-01-01", periods = n, freq = "d")
    s_ex = pd.Series(index = dates, data = values)

    s_std = rolling_estimate(s_ex, window)

    return s_std

def error_check(benchmark, trial):
    
    assert len(benchmark) == len(trial)

    for i, elem in enumerate(benchmark):

        check = (elem - trial[i]) > 0.0001
        if check:
            raise AssertionError("Differing values")
        else: continue
    

def driver(values, n = 100, window = 5):

    start_bench = time.time()
    df_output = output_modification(values, n, window)
    time_bench = time.time() - start_bench
    
    start_stride = time.time()
    df_input = input_modification(values, n, window)
    time_stride = time.time() - start_stride

    deque_m = df_output.to_numpy() ## Benchmark using the simpler method of a manual iteration accompanied by a Deque.
    stride_m = df_input.to_numpy()
    error_check(deque_m, stride_m)

    return time_bench, time_stride


def performance(window = 5):
    
    benchmark = []
    stride_algo = []
    x_axis = list(range(800, 16000, 800))
    
    for i in range(800, 16000, 800):
        values = number_generator(i)
        zeroInfill(values)
        
        t_bench, t_stride = driver(values, i, window)
        benchmark.append(t_bench)
        stride_algo.append(t_stride)

    ax = plt.subplot(1, 1, 1)
    p1 = ax.plot(x_axis, benchmark, label = "Benchmark Algorithm: using Pandas / Deque.")
    p2 = plt.plot(x_axis, stride_algo, label = "Stride Algorithm.")
    plt.legend(loc = "upper left")

    plt.xlabel("Data Range: number of data points.")
    plt.ylabel("Computational Performance.")
    plt.show()
        
    
if __name__ == "__main__":

    window = 5
    performance(window)
