import csv
import re

judge1 = re.compile(r'[0-9]{1,3}')
judge2 = re.compile(r'(^[0-9]{1,3})+(.[0-9]{1,6})]?$')
g = list(csv.reader(open('../data/housing_all.csv', encoding='utf-8', errors='ignore')))
housing_new = open('../data/housing_new.csv', 'a', newline='', encoding='utf-8')
writer = csv.writer(housing_new, dialect='excel')
title = ['id', 'house_ln', 'house_la', 'subway', 'bus_stop', 'street', 'room_type', 'accommodates',
         'bathroom', 'bedroom', 'beds', 'bed_type', 'guests', 'num_of_review	', 'review_score',
         'Entire_home', '	Private_room', 'Shared_room', '	crime_rate',
         'daily_price', 'weekly_price', 'monthly_price']
writer.writerow(title)


if __name__ == '__main__':
    matched = 0
    for i in range(1, len(g)):
        if g[i][19] == '':
            continue
        elif float(g[i][19]) > 1000:
            continue
        else:
            writer.writerow(g[i])
        if i % 500 == 0:
            print(i, 'houses done!')
