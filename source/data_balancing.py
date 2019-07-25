import csv


def up_sampling(imbalanced_csv: csv.reader) -> list:
    imbalanced_list = list(imbalanced_csv)
    title_line = imbalanced_list[0]
    balanced_list = [title_line]
    for i in range(1, len(imbalanced_list)):
        price = float(imbalanced_list[i][23])  # get "daily price"
        balanced_list.append(imbalanced_list[i])

        if 200 < price <= 400:
            balanced_list.extend([imbalanced_list[i]] * 2)
        elif 400 < price:
            balanced_list.extend([imbalanced_list[i]] * 6)

    return balanced_list


def down_sampling(imbalanced_csv: csv.reader) -> list:
    from numpy import random
    imbalanced_list = list(imbalanced_csv)
    title_line = imbalanced_list[0]
    balanced_list = [title_line]
    for i in range(1, len(imbalanced_list)):
        price = float(imbalanced_list[i][23])  # get "daily price"

        if price > 200:
            balanced_list.append(imbalanced_list[i])
        if price <= 200 and random.random() > 0.5:  # Drop half of data under price of 200
            balanced_list.append(imbalanced_list[i])

    return balanced_list


if __name__ == '__main__':
    imbalanced_file = csv.reader(open('../data/housing_all.csv'))

    balanced_file = csv.writer(open('../data/housing_up_sampling.csv', 'w', encoding='utf-8', newline=''))
    for line in up_sampling(imbalanced_file):
        balanced_file.writerow(line)
