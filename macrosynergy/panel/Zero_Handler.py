
import time
import numpy as np
from random import choice, randrange

## If there is a zero return, ascribed to holiday etc, the prevailing Standard Deviation should remain constant. The zero value would result in a spurious output. 
def constant_std(arr, arr_std):
    condition = np.where(arr == 0.0)[0]
    non_condition = np.where(arr != 0.0)[0]
    prev_ind = condition - 1

    flag = True
    for elem in prev_ind:
        if elem in non_condition:
            continue
        else:
            flag = False
    if flag:
        for i, elem in enumerate(condition):
            ## arr_std[i] will host the computation that has been disturbed by the 0.0 return value.
            arr_std[elem] = arr_std[prev_ind[i]]
    else:
        for zero_ind in condition:
            index = zero_ind
            for elem in reversed(arr[:(zero_ind + 1)]):
                if elem == 0.0:
                    index -= 1
                    continue
                else:
                    break
            value = arr_std[index]
            arr_std[zero_ind] = value
                
    return arr_std


def slide(arr, w, s = 1):

    return np.lib.stride_tricks.as_strided(arr,
                                           shape = ((len(arr) - w) + 1, w),
                                           strides = arr.strides * 2)[::s]

def swap(bench_row_, row_, index_b, index_e):
    
    row_[index_e] = bench_row_[index_b]
    return row_


## The algorithm is still under trial. Will confirm once complete.
def zero_shift(arr_window, window):

    final_col = arr_window[:, (window - 1)]
    zero_ret = np.where(final_col == 0.0)[0]
    prev_ind = (zero_ret - 1)
    arr_copy = arr_window.copy()
    arr_window = arr_window.copy()

    ## Unavoidable polynomial algorithm but the internal operations are minimal, so the performance is not comprised too greatly.
    for i, elem in enumerate(prev_ind):

        bench_row = arr_window[elem]
        start = zero_ret[i]
        index_b = 0
        index_e = -1
        for index in range(start, start + window):
            if (index + 1) > arr_window.shape[0]:
                break
            row = arr_copy[index]
            
            row_ = swap(bench_row, row, index_b, index_e)
            arr_window[index] = row_
            index_b += 1
            index_e -= 1

    return arr_window
    

def MAIN():

    data = [3, 4, 8, 11, 9, 3, 0, 4, 2]
    data = np.array(data)
    data = data.astype(dtype = np.float32)

    window = 4
    arr_window = slide(data, window)
    print(arr_window)
    arr_window = zero_shift(arr_window, window)
    print(arr_window)
    

if __name__ == "__main__":
    MAIN()
