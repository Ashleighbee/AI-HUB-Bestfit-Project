import csv

if __name__ == '__main__':
    f = list(csv.reader(open('../data/housing_clean_bed.csv', encoding='utf-8', errors='ignore')))
    g = list(csv.reader(open('../data/price_new.csv', encoding='utf-8', errors='ignore')))
    housing = open('../data/housing_new.csv', 'a', newline='', encoding='utf-8')
    writer = csv.writer(housing, dialect='excel')
    title = f[0] + ['new_price']
    writer.writerow(title)
    j_now = 0
    for i in range(1, len(g)):
        house_id = g[i][0]
        price = g[i][2]
        if price == '0':
            continue
        for j in range(1, len(f)):
            house = f[j]
            if house_id == house[0]:
                house.append(price)
                writer.writerow(house)
                j_now = j
        if i % 100 == 0:
            print(i, 'houses done!')
