from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import urllib
import re
import os

proxies = {
  'http': 'http://proxy-sen.noc.sony.co.jp:10080',
  'https': 'http://proxy-sen.noc.sony.co.jp:10080'
}

food_dict ={
    'meat': ['豚肉', '鶏肉', '鶏むね肉', '豚バラ肉', '豚の角煮', '鶏ささみ', '牛肉', '鶏もも肉'],
    'Vegetables': ['白菜', '里芋', '小松菜', 'れんこん', 'ほうれん草', 'かぼちゃ'],
    'fish': ['さんま', 'さば', '牡蠣', '鮭・サーモン', 'さわら'],
    'rice': ['カレー', 'おにぎり', '雑炊・おじや', 'オムライス', 'ドリア'],
    'pasta': ['カルボナーラ', 'グラタン', 'マカロニグラタン', 'ミートソース', '明太子・たらこスパ'],
    'noodle': ['うどん', 'カレーうどん', '煮込みうどん', '焼きうどん', 'にゅうめん', '焼きそば'],
    'soup': ['ビーフシチュー', 'クリームシチュー', 'シチュー', '中華スープ', 'けんちん汁'],
    'salad': ['生春巻き', '和風のサラダ', 'ポテトサラダ', 'かぼちゃのサラダ', '洋風のサラダ'],
    'fly': ['ポテトコロッケ', 'かぼちゃコロッケ', 'メンチカツ', '里芋コロッケ', 'さつまいもコロッケ'],
    'pot': ['おでん', 'もつ鍋', '豆乳鍋', 'しゃぶしゃぶ', '水炊き'],
    'japanese_fly': ['お好み焼き', 'チヂミ', 'たこ焼き', '広島風お好み焼き', '関西風お好み焼き'],
    'boal': ['おから', '豆腐', 'オムレツ', '豆腐ハンバーグ'],
    'side_food': ['ひじき', 'わかめ', '春雨', 'きくらげ', 'こんにゃく']
}
food_dict_light ={
    'rice': ['カレー', 'オムライス'],
}


def get_source(url, proxy = True):

    if proxy:
        res = requests.get(url, proxies=proxies)
    else:
        res = requests.get(url)

    sp = BeautifulSoup(res.content, "html.parser")

    if not sp:
        return 'empty'
    else:
        return sp


def get_title(sp):

    title = sp.find_all('h1', class_='ttl')

    if not title:
        return 'empty'
    else:
        r = re.search(r'\S*', title[0].contents[-1])
        return r.group()


def get_image(sp, url_root):

    div_img = sp.find_all('div', class_='recipe--detail-main')
    img = div_img[0].find_all('img')

    if not img:
        return 'empty'
    else:
        return urllib.parse.urljoin(url_root, img[0]['src'])


def get_calory(sp):

    result = ''
    calory = sp.find_all('div', class_='recipe')
    for n in calory:
        r = re.search(r'([0-9].*cal)', n.text)
        if r:
            result = r.group()

    if not result:
        return 'empty'
    else:
        return result


def get_salt(sp):

    result = ''
    salt = sp.find_all('div', class_='recipe')
    for n in salt:
        r = re.search(r'([0-9].*g)', n.text)
        if r:
            result = r.group()

    if not result:
        return 'empty'
    else:
        return result


def get_search_result(search_root, url_root, word, max_loop = 10, proxy=True):

    url = search_root.format(word)
    search_result = []
    for i in range(max_loop):
        result = []
        req_url = url + str(i+1)
        if proxy:
            res = requests.get(req_url, proxies=proxies)
        else:
            res = requests.get(req_url)
        sp = BeautifulSoup(res.content, 'html.parser')
        div = sp.find_all('div', class_='mk-tab-contents')
        div_child = div[0].find_all('div', class_='recipe--category-recipe')

        for div_c in div_child:
            result.append(urllib.parse.urljoin(url_root, div_c.attrs['data-url']))

        result = list(set(result))
        search_result += result

    return search_result


if __name__ == '__main__':

    search_root = 'https://www.kyounoryouri.jp/search/recipe?keyword={}&pg='
    url_root = 'https://www.kyounoryouri.jp/'
    save_file_name = 'kyouno.csv'

    max_search_loop = 10

    for key, value in tqdm(food_dict.items()):
        for word in tqdm(value):
            search_result = get_search_result(search_root, url_root, word, max_search_loop, proxy=False)

            category_list, food_type_list, url_list, title_list = [], [], [], []
            calory_list, salt_list, img_list = [], [], []

            for item in tqdm(search_result):
                url_list.append(item)
                sp = get_source(item, proxy=False)
                if sp:
                    category_list.append(key)
                    food_type_list.append(word)
                    title_list.append(get_title(sp))
                    calory_list.append(get_calory(sp))
                    salt_list.append(get_salt(sp))
                    img_list.append(get_image(sp, url_root))
            df = pd.DataFrame({'category': category_list,
                               'type': food_type_list,
                               'title': title_list,
                               'calorie': calory_list,
                               'salt': salt_list,
                               'img': img_list,
                               'url': url_list})
            if os.path.exists(save_file_name):
                df.to_csv(save_file_name, index=False, header=False, mode='a')
            else:
                df.to_csv(save_file_name, index=False, header=True)