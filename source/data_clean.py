import csv


def price_clean(line_list, writer):
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

def bed_clean(line_list, writer):
    for i in range(0, len(line_list)):
        beds = line_list[i][12]
        accommodates = line_list[i][9]
        if 'beds' in beds:
            writer.writerow(line_list[i])
            continue
        elif beds == '' or beds == '0':
            continue
        beds = float(beds)
        accommodates = float(accommodates)
        if beds > accommodates or beds * 2 < accommodates:
            continue
        else:
            writer.writerow(line_list[i])
        if i % 500 == 0:
            print(i, 'houses done!')

def price_for_each_clean(line_list, writer):
    for i in range(0, len(line_list)):
        accommodates = line_list[i][9]
        price = line_list[i][26]
        if 'daily_price' in price:
            writer.writerow(line_list[i])
            continue
        elif price == '' or price == '0':
            continue
        accommodates = float(accommodates)
        price_for_each = float(price) / accommodates
        if price_for_each < 5 or price_for_each > 109:
            continue
        else:
            writer.writerow(line_list[i])
        if i % 500 == 0:
            print(i, 'houses done!')


if __name__ == '__main__':
    housing_new = open('../data/housing_all_clean_.csv', 'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(housing_new, dialect='excel')
    g = list(csv.reader(open('../data/housing_all_clean.csv', encoding='utf-8', errors='ignore')))

    price_for_each_clean(g, csv_writer)
