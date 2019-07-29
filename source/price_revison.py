import csv
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import sys


def search(house_id):
    url = 'https://www.airbnb.cn/rooms/' + house_id + '&display_currency=USD'
    try:
        browser.get(url)
        _price = wait.until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, '#site-content > section > div._qmx5s9 > div > div > div._2h22gn > div._1av41w02 '
                                      '> div > div > div > div:nth-child(1) > div > div > div._16tcko6 > div > div '
                                      '> div:nth-child(2) > div._12cv0xt > div > span._8eazm0k > span'))).text
        return _price
    except TimeoutException:
        return -1


if __name__ == '__main__':
    if len(sys.argv) > 1:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
    else:
        start = 1
        end = 100

    firefox_option = webdriver.FirefoxOptions()
    firefox_option.add_argument(argument='--headless')
    browser = webdriver.Firefox(options=firefox_option)  # options=chrome_option
    wait = WebDriverWait(browser, 10)  # 最长等待时间为10S
    f = list(csv.reader(open('../data/housing_all.csv')))
    if start + 100 > len(f):
        s = len(f)
    else:
        s = start + 100
    price_new = open('../data/price/price_new_' + str(s) + '.csv', 'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(price_new, dialect='excel')
    for i in range(start, end):
        house = f[i]
        h_id = house[0]
        price = search(h_id)
        if '$ ' or '¥' in price:
            price = str(price)[1:]
        price = price.replace(',', '')
        row = [h_id, house[26], round(float(price) / 6.88)]
        print(i, row)
        csv_writer.writerow(row)
        if i % 100 == 0:
            price_new.close()
            price_new = open('../data/price/price_new_' + str(i+100) + '.csv', 'w', newline='', encoding='utf-8')
            csv_writer = csv.writer(price_new, dialect='excel')
