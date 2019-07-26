import csv


def price_clean(line_list: list, writer: csv.writer):
    for i in range(0, len(line_list)):
        price = line_list[i][26]
        if 'daily_price' in price:
            writer.writerow(line_list[i])
        elif price == '':
            continue
        elif float(price) > 384:
            continue
        else:
            writer.writerow(line_list[i])
        if i % 500 == 0:
            print(i, 'houses done!')


if __name__ == '__main__':
    housing_new = open('../data/housing_clean.csv', 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(housing_new, dialect='excel')
    g = list(csv.reader(open('../data/housing_all.csv', encoding='utf-8', errors='ignore')))
    price_clean(g, csv_writer)
