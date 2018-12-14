from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import urllib


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



def get_source(url):

    res = requests.get(url)
    # res = requests.get(url, proxies=proxies)
    sp = BeautifulSoup(res.content, "html.parser")

    if not sp:
        return 'empty'
    else:
        return sp


def get_title(sp):

    title = sp.find_all('span', attrs={'itemprop': 'headline'})

    if not title:
        return 'empty'
    else:
        return title[0].text


def get_image(sp):

    img = sp.find_all('img', attrs={'id': 'recipe_image'})
    if not img:
        return 'empty'
    else:
        return img[0]['src']


def get_calory(sp):

    calory = sp.find_all('b', class_='nutrition')
    if not calory:
        return 'empty'
    else:
        return calory[0].text


def get_salt(sp):

    salt = sp.find_all('dl', attrs={'data-label': 'salt'})
    if not salt:
        return 'empty'
    else:
        return salt[0].find("dd").text


def get_search_result(search_root, url_root, word, max_loop = 10):

    url = search_root.format(word)
    search_result = []
    for i in range(max_loop):
        result = []
        req_url = url + str(i+1)
        res =requests.get(req_url)
        # res =requests.get(req_url, proxies=proxies)
        sp = BeautifulSoup(res.content, 'html.parser')
        a_tag = sp.find_all('a')
        for a in a_tag:
            if 'search/recipe' in a['href']:
                result.append( urllib.parse.urljoin(url_root, a['href']))

        search_result += list((set)(result))

    return search_result


if __name__ == '__main__':

    category_list = []
    food_type_list = []
    url_list = []
    title_list = []
    calory_list = []
    salt_list = []
    img_list = []

    search_root = 'http://www.kikkoman.co.jp/homecook/search/select_search.html?free_word=%E3%82%AB%E3%83%AC%E3%83%BC&not_word=&C5=&C6=&C14=&site_type=J&sort=&page='
    url_root = 'http://www.kikkoman.co.jp'

    max_search_loop = 10

    for key, value in tqdm(food_dict.items()):
        for word in tqdm(value):
            search_result = get_search_result(search_root, url_root, word, max_search_loop)
            for item in tqdm(search_result):
                url_list.append(item)
                sp = get_source(item)
                if sp:
                    category_list.append(key)
                    food_type_list.append(word)
                    title_list.append(get_title(sp))
                    calory_list.append(get_calory(sp))
                    salt_list.append(get_salt(sp))
                img_list.append(get_image(sp))

    df = pd.DataFrame({'category': category_list,
                       'type': food_type_list,
                       'title': title_list,
                       'calorie': calory_list,
                       'salt': salt_list,
                       'img': img_list})

    df.to_csv('kikkoman.csv', index=False)
