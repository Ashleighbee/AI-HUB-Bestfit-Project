from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import csv

def One_Hot_encoder(filename):
    df = pd.read_csv(filename)
    data1= df["room_type"]
    ans = LabelBinarizer().fit_transform(data1)
    return ans
    # ans ["Entire home/apt","Private room", "Shared room"]

def add_features(filename):
    lis = One_Hot_encoder(filename)
    Entire_home = []
    Private_room = []
    Shared_room = []
    for ele in lis:
        Entire_home.append(ele[0])
        Private_room.append(ele[1])
        Shared_room.append(ele[2])
    Entire_home = pd.DataFrame(Entire_home)
    Private_room = pd.DataFrame(Private_room)
    Shared_room = pd.DataFrame(Shared_room)
    df = pd.concat([Entire_home,Private_room,Shared_room],axis=1)
    return df

def add_to_main(filename):
    original_df = pd.read_csv(filename)
    df = add_features(filename)
    ent_r = df.iloc[:,0]
    pri_r = df.iloc[:,1]
    shr_r = df.iloc[:,2]
    original_df["Entire_home"] = ent_r
    original_df["Private_room"] = pri_r
    original_df["Shared_room"] = shr_r
    original_df.to_csv(filename)


print(add_to_main("../data/housing_all.csv"))






