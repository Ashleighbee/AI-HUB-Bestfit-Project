import csv
import re

judge1 = re.compile(r'[0-9]{1,3}')
judge2 = re.compile(r'(^[0-9]{1,3})+(.[0-9]{1,6})]?$')
f = list(csv.reader(open('../data/listing_new.csv', encoding='utf-8', errors='ignore')))
g = list(csv.reader(open('../data/housing_all_clean.csv', encoding='utf-8', errors='ignore')))
housing_new = open('../data/housing_all_new.csv', 'a', newline='', encoding='utf-8')
writer = csv.writer(housing_new, dialect='excel')
title = g[0] + ['availability']
writer.writerow(title)

# 77 78 79
if __name__ == '__main__':
    matched = 0
    count = 0
    for i in range(1, len(f)):
        for j in range(count, len(g)):
            if f[i][0] == g[j][0]:
                count = j
                if f[i][77] == '0' or f[i][78] == '0' or f[i][79] == '0':
                    g[j].append(0)
                    writer.writerow(g[j])
                else:
                    g[j].append(1)
                    writer.writerow(g[j])
                break
        if i % 500 == 0:
            print(i, 'houses done!')
