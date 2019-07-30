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
    count = 0
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
            count += 1
            continue
        else:
            writer.writerow(line_list[i])
        if i % 500 == 0:
            print(i, 'houses done!')
    return count


def price_for_each_clean(line_list, writer):
    low = 0
    high = 0
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
        if price_for_each < 15:
            low += 1
            continue
        if price_for_each > 142:
            high += 1
            continue
        else:
            writer.writerow(line_list[i])
        if i % 500 == 0:
            print(i, 'houses done!')
    print(low, high)
    return low + high

def bedroom_clean(line_list, writer):
    count = 0
    for i in range(0, len(line_list)):
        accommodates = line_list[i][9]
        bedroom = line_list[i][11]
        if 'bedroom' in bedroom:
            writer.writerow(line_list[i])
            continue
        elif bedroom == '' or bedroom == '0':
            continue
        delta = int(bedroom) - int(accommodates)
        if delta >= 1:
            count += 1
            continue
        else:
            writer.writerow(line_list[i])
        if i % 500 == 0:
            print(i, 'houses done!')
    return count


if __name__ == '__main__':
    housing_new = open('../data/housing_all_clean_.csv', 'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(housing_new, dialect='excel')
    g = list(csv.reader(open('../data/housing_all_clean.csv', encoding='utf-8', errors='ignore')))

    # delete = bed_clean(g, csv_writer)
    # delete = bedroom_clean(g, csv_writer)
    delete = price_for_each_clean(g, csv_writer)

    print('Delete num:', delete)
