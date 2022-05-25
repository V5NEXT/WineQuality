import os
import glob
import pandas as pd
os.chdir("../Data_Set")


def read_combine():
    df_red = pd.read_csv('../Data_Set/winequality-red.csv')
    df_white = pd.read_csv('../Data_Set/winequality-red.csv')

    df_red['type'] = 0
    df_white['type'] = 1

    frames = [df_red, df_white]

    df_combined = pd.concat(frames)

    # # export to csv
    df_combined.to_csv("df_combined.csv", index=False, encoding='utf-8-sig')

    print(df_combined.head)


read_combine()
