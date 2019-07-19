import csv
import re


def load_housing(filename):
    f = csv.reader(open(filename, 'r', encoding='utf-8', errors='ignore'))
    # f = list(f)
    house = []
    for each in f:
        house.append(each)
    house = house[1:len(house)]
    # print(house)
    return house


if __name__ == '__main__':
    guests = re.compile(r'[0-9]{1,3}')
    f = csv.reader(open('../data/housing_all.csv', encoding='utf-8', errors='ignore'))
    f = list(f)
    listing_new = open('../data/housing_new.csv', 'a', newline='', encoding='utf-8')
    writer = csv.writer(listing_new, dialect='excel')
    title = ['id', 'house_ln', 'house_la', 'subway', 'bus_stop', 'street', 'room_type', 'bedroom', 'guests',
             'num_of_review', 'review_score']
    writer.writerow(title)
    for i in range(1, len(f)):
        line = f[i]
        judge1 = line[7]
        judge2 = line[8]
        if guests.fullmatch(judge1) and guests.fullmatch(judge2):
            if int(judge2) > 30:
                continue
        else:
            continue
        # print(house)
        writer.writerow(line)
        if i % 100 == 0:
            print(i, 'houses done!')

