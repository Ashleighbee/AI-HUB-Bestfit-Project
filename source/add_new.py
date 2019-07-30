import csv
import re

judge1 = re.compile(r'[0-9]{1,3}')
judge2 = re.compile(r'(^[0-9]{1,3})+(.[0-9]{1,6})]?$')
f = list(csv.reader(open('../data/listing_new.csv', encoding='utf-8', errors='ignore')))
g = list(csv.reader(open('../data/housing_all.csv', encoding='utf-8', errors='ignore')))
housing_new = open('../data/housing_clean.csv', 'a', newline='', encoding='utf-8')
writer = csv.writer(housing_new, dialect='excel')
title = g[0]
writer.writerow(title)


if __name__ == '__main__':
    matched = 0
    for i in range(1, len(g)):
        if g[i][19] == '':
            continue
        elif float(g[i][19]) > 500:
            continue
        else:
            writer.writerow(g[i])
        if i % 500 == 0:
            print(i, 'houses done!')
