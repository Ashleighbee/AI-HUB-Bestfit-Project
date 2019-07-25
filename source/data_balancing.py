import csv


def up_sampling(imbalanced_csv: csv.reader) -> list:
    imbalanced_list = list(imbalanced_csv)
    title_line = imbalanced_list[0]
    balanced_list = [title_line]
    for i in range(1, len(imbalanced_list)):
        price = float(imbalanced_list[i][23])  # get "daily price"

        if price < 200:
            balanced_list.append(imbalanced_list[i])
        elif 200 < price <= 400:
            balanced_list.extend([imbalanced_list[i]] * 4)
        elif 400 < price <= 600:
            balanced_list.extend([imbalanced_list[i]] * 20)
        elif 600 < price <= 800:
            balanced_list.extend([imbalanced_list[i]] * 40)

    return balanced_list


def down_sampling(imbalanced_csv: csv.reader) -> list:
    from numpy import random
    imbalanced_list = list(imbalanced_csv)
    title_line = imbalanced_list[0]
    balanced_list = [title_line]
    for i in range(1, len(imbalanced_list)):
        price = float(imbalanced_list[i][23])  # get "daily price"

        if 50 < price < 100:  # Drop half of data under price of 200
            if random.random() > 0.3:
                balanced_list.append(imbalanced_list[i])
        else:
            balanced_list.append(imbalanced_list[i])

    return balanced_list


if __name__ == '__main__':
    imbalanced_file = csv.reader(open('../data/housing_all.csv', 'r'))

    balanced_file = csv.writer(open('../data/housing_up_sampling.csv', 'w', encoding='utf-8', newline=''))
    for line in up_sampling(imbalanced_file):
        balanced_file.writerow(line)

    imbalanced_file = csv.reader(open('../data/housing_up_sampling.csv', 'r'))

    balanced_file = csv.writer(open('../data/housing_price_balanced.csv', 'w', encoding='utf-8', newline=''))
    for line in down_sampling(imbalanced_file):
        balanced_file.writerow(line)
