import csv
import time
from math import radians, cos, sin, asin, sqrt
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import re

# browser = webdriver.Firefox()
# url = 'https://www.openstreetmap.org/directions?engine=fossgis_osrm_foot'
# browser.get(url)
# wait = WebDriverWait(browser, 10)
housing = open('../data/housing_new.csv', 'a', newline='', encoding='utf-8')
writer = csv.writer(housing, dialect='excel')
title = ['id', 'house_ln', 'house_la', 'subway', 'bus_stop']
writer.writerow(title)


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

    print(sub_statiton)
    return sub_statiton

def load_university(filename):
    f = csv.reader(open(filename))
    f = list(f)
    university = []
    for i in range(1, len(f)-1):
        s = f[i][0].strip(')')
        s = s[s.index('(')+1:]
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
    # f = list(f)
    house = []
    for each in f:
        house.append(each)
    house = house[1:len(house)]
    # print(house)
    return house

def cal_distance(lng1, lat1, lng2, lat2):
    # lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])     # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2-lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000   # 地球平均半径，6371km
    distance = round(distance/1000, 3)
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
    new_house = []
    for house in _houses:
        count = 0
        route = []
        for each in fa:
            dis = cal_distance(house[1], house[2], each[0][0], each[0][1])
            if dis == -1:
                continue
            elif dis < 0.5:
                for r in each[1]:
                    route.append(r)

        route = list(set(route))
        count += len(route)
        # print(route)
        house.append(count)
        writer.writerow(house)
        new_house.append(house)
        if n % 100 == 0:
            print(n, 'houses done!')
        n += 1
    return new_house

def counting_(_houses):
    fa = load_bus('bus.csv')
    n = 1
    new_house = []
    for house in _houses:
        count = 0
        for each in fa:
            dis = cal_distance(house[1], house[2], each[0], each[1])
            if dis == -1:
                continue
            elif dis < 0.5:
                count += 1
        house.append(count)
        writer.writerow(house)
        new_house.append(house)
        if n % 100 == 0:
            print(n, 'houses done!')
        n += 1
    return new_house

def write_house_data(_data):
    for line in _data:
        writer.writerow(line)

def load_crime_rate_by_precinct(filename):
    import json

    def my_obj_pairs_hook(lst): # Deal with duplicate keys in json
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

    print('')

if __name__ == "__main__":
    # houses = load_list('listings.csv')
    # houses = load_housing('../data/housing.csv')
    # houses = counting_sub(houses)
    # print(houses)
    # write_house_data(houses)
    # load_subway('subway.csv')
    load_crime_rate_by_precinct('../data/crime.json')
