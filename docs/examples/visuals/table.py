"""example/macrosynergy/visuals/table.py"""
import pandas as pd
from macrosynergy.visuals import view_table

data = {
    "Col1": [1, 2, 3, 4],
    "Col2": [5, 6, 7, 8],
    "Col3": [9, 10, 11, 12],
    "Col4": [13, 14, 15, 16],
}


row_labels = ["Row1", "Row2", "Row3", "Row4"]


df = pd.DataFrame(data, index=row_labels)


view_table(df, title="Table")
