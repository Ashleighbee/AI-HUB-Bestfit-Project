import pandas as pd
import numpy as np

def One_Hot(filename):
    df = pd.read_csv(filename)
    # filename = "../data/housing_all.csv"
    series1 = df["superhost"]
    # type(data) is pandas series
    suph = pd.DataFrame({'superhost':series1})
    # convert the series I extracted from the housing_all to Dataframe
    # b/c using get_dummies we have to put in dataFrame
    onehot = pd.get_dummies(suph)
    print(onehot)


def host_resp(filename):
    df = pd.read_csv(filename)
    see = pd.unique(df["host_response_time"])
    # ['within an hour' 'within a few hours' 'within a day' nan 'a few days or more']
    serie = df["host_response_time"]
    lis = []
    for i in serie:
        if (i == 'within an hour'):
            lis.append(3)
        elif (i == 'within a few hours'):
            lis.append(2)
        elif (i == 'within a day'):
            lis.append(1)
        else:
            lis.append(0)
    host_rt = pd.DataFrame(lis)
    return host_rt

def add_to_main(filename):
    original_df = pd.read_csv(filename)
    hostrr = host_resp(filename)
    original_df["response_time_num"] = hostrr
    original_df.to_csv(filename)



if __name__ == '__main__':
    print(add_to_main("../data/housing_all.csv"))

