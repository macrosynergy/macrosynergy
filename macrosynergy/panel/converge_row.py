import numpy as np

class ConvergeRow(object):

    def __init__(self, row, m_weight, active_cross, r_length):
        self.row = row
        self.maximum = self.max_row()
        self.m_weight = m_weight
        self.active_cross = active_cross
        
        self.m_index = self.max_index()
        self.margin = 0.001 
        
    def distribute(self):

        while True:

            excess = self.max_row() - self.m_weight

            excess_cross = self.excess_count()
            indexing = self.index_row()
            
            if excess_cross.size == 1:
                self.row[self.m_index] -= excess
                
            elif excess_cross.size > 1:
                for index in excess_cross:
                    indexing[index] = self.m_weight
                    
                self.row = np.array(list(indexing.values()))
                excess = 1.0 - np.nansum(self.row)
            else:
                break

            amount = excess / self.active_cross
            self.row += amount
            

    def max_row(self):
        return np.nanmax(self.row)

    def excess_count(self):
        row_copy = self.row.copy()

        return np.where((row_copy - self.m_weight) > 0.001)[0]

    def max_index(self):
        return np.where(self.row == self.maximum)[0][0]

    def index_row(self):
        return dict(zip(range(self.row.size), self.row))
