import numpy as np
import pandas as pd

def to_heatmap(patch_index: pd.DataFrame, class_name: str) -> np.array:
    patch_index.columns = [colname.lower() for colname in patch_index.columns]
    class_name = class_name.lower()
    class_df = patch_index[['row', 'column', class_name]]
    # thumb_out = per_subimage(class_df)
    max_rows = int(np.max(class_df.row)) + 1
    max_cols = int(np.max(class_df.column)) + 1

    # create a blank thumbnail
    thumbnail_out = np.zeros((max_rows, max_cols))

    # for each row in dataframe set the value of the pixel specified by row and column to the probability in clazz
    for rw in range(class_df.shape[0]):
        df_row = class_df.iloc[rw]
        thumbnail_out[int(df_row.row), int(df_row.column)] = df_row[class_name]

    return thumbnail_out

