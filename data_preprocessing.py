import mat4py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(mat_file):
    dataset = mat4py.loadmat(mat_file)
    df = pd.DataFrame(dataset["VehicleInfo"])
    
    df_train, df_test = train_test_split(df,test_size=0.2,random_state=2020)
    df_train = df_train.reset_index().drop(columns=["index"])
    df_test = df_test.reset_index().drop(columns=["index"])
    return df_train, df_test

def get_annotation_file(df, SAVE_PATH, DATASET_PATH, file_name="annotation.txt"):
    f= open(os.path.join(SAVE_PATH,file_name),"w+")
    for idx, row in df.iterrows():
      if row["nVehicles"] > 1:
        for i in range(row["nVehicles"]):
          x1 = row["vehicles"]["left"][i]
          x2 = row["vehicles"]["right"][i]
          y1 = row["vehicles"]["top"][i]
          y2 = row["vehicles"]["bottom"][i]
          fileName = os.path.join(DATASET_PATH, row['name'])
          className = row["vehicles"]["category"][i]
          f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')
      else:
        x1 = row["vehicles"]["left"]
        x2 = row["vehicles"]["right"]
        y1 = row["vehicles"]["top"]
        y2 = row["vehicles"]["bottom"]
        fileName = os.path.join(DATASET_PATH, row['name'])
        className = row["vehicles"]["category"]
        f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')
    f.close()
