import csv
import time
from math import radians, cos, sin, asin, sqrt
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from collections import OrderedDict
import re
import pandas as pd
import numpy as np

# browser = webdriver.Firefox()
# url = 'https://www.openstreetmap.org/directions?engine=fossgis_osrm_foot'
# browser.get(url)
# wait = WebDriverWait(browser, 10)
housing = open('../data/housing_new.csv', 'a', newline='', encoding='utf-8')
writer = csv.writer(housing, dialect='excel')

csv.field_size_limit(int(5e8))


def load_subway(filename):
    f = csv.reader(open(filename))
    f = list(f)
    sub_statiton = []
    for i in range(1, len(f)):
        if f[i][2] == f[i - 1][2]:
            continue
        sub = [f[i][4], f[i][3]]
        r = []
        for j in range(5, 16):
            if f[i][j] != '':
                r.append(f[i][j])
        sub_statiton.append([sub, r])

    # print(sub_statiton)
    return sub_statiton


def load_university(filename):
    f = csv.reader(open(filename))
    f = list(f)
    university = []
    for i in range(1, len(f) - 1):
        s = f[i][0].strip(')')
        s = s[s.index('(') + 1:]
        pos = s.split(' ')
        university.append([float(pos[0]), float(pos[1]), f[i][1]])
        # print(university)
    return university


def load_bus(filename):
    f = csv.reader(open(filename))
    f = list(f)
    bus_stop = []
    for i in range(1, len(f) - 1):
        s = f[i][3].strip(')')
        s = s[s.index('(') + 1:]
        pos = s.split(' ')
        bus_stop.append([float(pos[0]), float(pos[1])])
    # print(bus_stop)
    return bus_stop


def load_park(filename):
    f = csv.reader(open(filename))
    f = list(f)
    park = []
    for each in f:
        park.append([each[0], each[2], each[1]])
    return park


def park_processing(_houses, _title):
    park = load_park('../data/park.csv')
    for each in park:
        _title.append(each[0])
    writer.writerow(_title)
    for house in _houses:
        for each in park:
            dis = cal_distance(each[1], each[2], house[1], house[2])
            house.append(dis)
        writer.writerow(house)


def load_scenery(filename):
    f = csv.reader(open(filename))
    f = list(f)
    scenery = []
    for each in f:
        scenery.append([each[0], each[2], each[1]])
    return scenery


def scenery_processing(_houses, _title):
    scenery = load_park('../data/scenery.csv')
    for each in scenery:
        _title.append(each[0])
    writer.writerow(_title)
    for house in _houses:
        for each in scenery:
            dis = cal_distance(each[1], each[2], house[1], house[2])
            house.append(dis)
        writer.writerow(house)


def load_list(filename):
    f = csv.reader(open(filename, 'r', encoding='utf-8', errors='ignore'))
    f = list(f)
    # housing = open('housing.csv', 'a', newline='', encoding='utf-16')
    # writer = csv.writer(housing, dialect='excel')
    house = []
    for i in range(1, len(f)):
        house.append([f[i][0], f[i][7], f[i][6]])
        # writer.writerow([pos[0], pos[1]])
        # 62 -- monthly price
        # if '$' in mon_price:
        #     print(mon_price)
    print(house)
    return house


def load_housing(filename):
    f = csv.reader(open(filename, 'r', encoding='utf-8', errors='ignore'))
    f = list(f)
    house = []
    for each in f:
        house.append(each)
    title = house[0]
    house = house[1:len(house)]
    # print(house)
    return house, title


def cal_distance(lng1, lat1, lng2, lat2):
    # lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
    return distance


# def cal_dis_on_map(ln1, la1, ln2, la2):
#     try:
#         input1 = wait.until(EC.presence_of_element_located(
#             (By.CSS_SELECTOR, '#sidebar > div:nth-child(1) > form:nth-child(2) > div:nth-child(2) > '
#                               'span:nth-child(2) > input:nth-child(1)')))
#         input2 = wait.until(EC.presence_of_element_located(
#             (By.CSS_SELECTOR, '#sidebar > div:nth-child(1) > form:nth-child(2) > div:nth-child(3) > '
#                               'span:nth-child(2) > input:nth-child(1)')))
#         btn = wait.until(EC.presence_of_element_located(
#             (By.CSS_SELECTOR, '#sidebar > div:nth-child(1) > form:nth-child(2) > div:nth-child(4) > '
#                               'input:nth-child(2)')))
#         input1.clear()
#         input2.clear()
#         input1.send_keys(la1 + ', ' + ln1)
#         input2.send_keys(la2 + ', ' + ln2)
#         btn.click()
#         time.sleep(3)
#         dis = wait.until(EC.presence_of_element_located(
#             (By.CSS_SELECTOR, '#routing_summary'))).text
#         dis = re.split('[：:m]', dis)[1]
#
#         if 'k' in dis:
#             dis = dis.strip('km')
#             dis = float(dis) * 1000
#         else:
#             dis = float(dis)
#         return dis
#     except TimeoutException:
#         # warning = wait.until(EC.presence_of_element_located(
#         #     (By.CSS_SELECTOR, '.search_results_error'))).text
#         return -1


def counting_sub(_houses):
    fa = load_subway('../data/subway.csv')
    n = 1
    for house in _houses:
        count = 0
        route = []
        for each in fa:
            dis = cal_distance(house[1], house[2], each[0][0], each[0][1])
            if dis == -1:
                continue
            elif dis < 0.5:
                # for r in each[1]:
                #     route.append(r)

                # route = list(set(route))
                count += 1
        # print(route)
        house.append(count)
        writer.writerow(house)
        if n % 500 == 0:
            print(n, 'houses done!')
        n += 1


def write_stations_distance(housefile, stationfile, load_func, new_labels:tuple):
    station_list = load_func(stationfile)
    station_cord_list = []
    # grab coordinates
    for i in station_list:
        if load_func == load_subway:
            station_cord_list.append(tuple(i[0]))
        elif load_func == load_bus:
            station_cord_list.append(tuple(i))

    house_list = list(csv.reader(open(housefile, 'r')))
    house_writer = csv.writer(open(housefile, 'w', newline=''))
    house_list[0].extend(new_labels)
    house_writer.writerow(house_list[0])

    for line in house_list:
        if line[0] == 'id':
            continue

        ln = line[1]
        la = line[2]

        dist_list = []

        for sub_cord in station_cord_list:
            dist_list.append(cal_distance(ln, la, sub_cord[0], sub_cord[1]))

        dist_list.sort()
        line.extend(dist_list[:len(new_labels)])
        house_writer.writerow(line)


def counting_(_houses, fa):
    n = 1
    for house in _houses:
        count = 0
        for each in fa:
            dis = cal_distance(house[1], house[2], each[0], each[1])
            if dis == -1:
                continue
            elif dis < 5:
                count += 1
        house.append(count)
        writer.writerow(house)
        if n % 500 == 0:
            print(n, 'houses done!')
        n += 1


def load_crime_rate_by_precinct(filename='../data/crime.json') -> dict:
    """
    :param filename: Crime rate data file (in json)
    :return: A dictionary. Keys are precinct name, values are crime per 1000 population
    """
    import json

    def my_obj_pairs_hook(lst):  # Deal with duplicate keys in json
        result = {}
        count = {}
        for key, val in lst:
            if key in count:
                count[key] = 1 + count[key]
            else:
                count[key] = 1
            if key in result:
                if count[key] > 2:
                    result[key].append(val)
                else:
                    result[key] = [result[key], val]
            else:
                result[key] = val
        return result

    raw = json.load(open(filename), object_pairs_hook=my_obj_pairs_hook)

    per1000_list = {}

    for sub_list in raw.values():
        if type(sub_list) == list:
            for item in sub_list:
                per1000_list[item['name']] = item['per1000']
        elif type(sub_list) == dict:
            per1000_list[sub_list['name']] = sub_list['per1000']

    return per1000_list


def load_police_precinct_boundary(filename='../data/police.csv') -> dict:
    """
    :param filename:
    :return: A dictionary. Keys are precinct name (only numbers), values are boudary lists.
             Some precincts have multiple regions.
             return format:
             {
                "1": [[(-70.1, 40.2), (-70.0, 40.1)], [(-70.1, 40.2), (-70.0, 40.1)], ...]
                # [[Cords of region 1], [Cords,of regions2], ...]
                ......
             }
    """
    f = csv.reader(open(filename, 'r', encoding='utf-8'))

    boundary_list = {}

    for line in f:
        if line[0] == 'the_geom':
            continue
        cords = line[0].lstrip('MULTIPOLYGON (((').rstrip(')))').split(')), ((')
        boundary_list[f'{line[3]}'] = []

        for i in range(len(cords)):
            cords[i] = cords[i].split(', ')
            boundary_list[f'{line[3]}'].append([])

            for j in range(len(cords[i])):
                lo = float(cords[i][j].split(' ')[0])
                la = float(cords[i][j].split(' ')[1])
                boundary_list[f'{line[3]}'][i].append((lo, la))

    return boundary_list


def is_pt_in_poly(aLon, aLat, pointList) -> bool:
    """
    :param aLon: double 经度
    :param aLat: double 纬度
    :param pointList: list [(lon, lat)...] 多边形点的顺序需根据顺时针或逆时针，不能乱
    """

    iSum = 0
    iCount = len(pointList)

    if iCount < 3:
        return False

    for i in range(iCount):

        pLon1 = pointList[i][0]
        pLat1 = pointList[i][1]

        if i == iCount - 1:

            pLon2 = pointList[0][0]
            pLat2 = pointList[0][1]
        else:
            pLon2 = pointList[i + 1][0]
            pLat2 = pointList[i + 1][1]

        if ((aLat >= pLat1) and (aLat < pLat2)) or ((aLat >= pLat2) and (aLat < pLat1)):

            if abs(pLat1 - pLat2) > 0:

                pLon = pLon1 - ((pLon1 - pLon2) * (pLat1 - aLat)) / (pLat1 - pLat2)

                if pLon < aLon:
                    iSum += 1

    if iSum % 2 != 0:
        return True
    else:
        return False


def cal_crime_rate_by_housing(filename='../data/housing_all.csv') -> dict:
    """
    :param filename: housing_all.csv
    :return: A dict. Keys are ids of houses. Values are crime rates.
    """
    f = csv.reader(open(filename, 'r', encoding='utf-8'))
    precinct_boundary = load_police_precinct_boundary(filename='../data/police.csv')
    crime_rate_by_precinct = load_crime_rate_by_precinct(filename='../data/crime.json')
    crime_rate_by_housing = OrderedDict()

    count = 0  # For progress output

    for listing in f:
        if listing[1] == 'id':
            continue
        house_id = listing[1]
        house_ln = float(listing[2])
        house_la = float(listing[3])

        precinct_name = 'None'  # Default precinct_name for houses not in any precincts
        for precinct in precinct_boundary.keys():
            for region in precinct_boundary[precinct]:
                if is_pt_in_poly(house_ln, house_la, region):
                    precinct_name = precinct  # Deal with different precinct name in two sets
                    if precinct.endswith('1'):
                        precinct_name += 'st'
                    elif precinct.endswith('2'):
                        precinct_name += 'nd'
                    elif precinct.endswith('3'):
                        precinct_name += 'rd'
                    else:
                        precinct_name += 'th'
                    break

        crime_rate_by_housing[house_id] = np.nan  # Default crime rate for precinct_name == 'None'
        if not precinct_name == 'None':
            if precinct_name == '14th':
                crime_rate_by_housing[house_id] = crime_rate_by_precinct['Manhattan South Precinct']
            elif precinct_name == '18th':
                crime_rate_by_housing[house_id] = crime_rate_by_precinct['Manhattan North Precinct']
            elif precinct_name == '22nd':
                crime_rate_by_housing[house_id] = crime_rate_by_precinct['Central Park Precinct']
            else:
                for precinct in crime_rate_by_precinct.keys():
                    if precinct.startswith(precinct_name):
                        crime_rate_by_housing[house_id] = crime_rate_by_precinct[precinct]
                        break

        count += 1
        if count % 500 == 0:
            print(count)  # Print progress

    return crime_rate_by_housing


def write_crime_rate(filename):
    crime_rate_dic = cal_crime_rate_by_housing()
    retVal = []
    for key in crime_rate_dic.keys():
        val = crime_rate_dic[key]
        retVal.append(val)
    dic = {"crime_rate": retVal}
    df1 = pd.DataFrame(dic)
    df2 = pd.read_csv("../data/housing_all.csv")
    data = df1["crime_rate"]
    df2["crime_rate"] = data
    df2.to_csv("../data/housing_all.csv")


def one_hot(filename):
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
        if i == 'within an hour':
            lis.append(3)
        elif i == 'within a few hours':
            lis.append(2)
        elif i == 'within a day':
            lis.append(1)
        else:
            lis.append(0)
    host_rt = pd.DataFrame(lis)
    return host_rt


def write_response_time(filename):
    original_df = pd.read_csv(filename)
    hostrr = host_resp(filename)
    original_df["response_time_num"] = hostrr
    original_df.to_csv(filename)


if __name__ == "__main__":
    # write_stations_distance(housefile='../data/housing_all.csv',
    #                         stationfile='../data/bus.csv',
    #                         load_func=load_bus,
    #                         new_labels=('bus_dist_1', 'bus_dist_2', 'bus_dist_3'))
    houses, title = load_housing('../data/housing_clean.csv')
    writer.writerow(title)
    counting_sub(houses)
