from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import urllib
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
    'rice': ['カレー'],
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

    result = ''
    title = sp.find_all('span', class_='dispbl')
    for n in title:
        nc = n.findChild('h1')
        if nc:
            result = nc.text

    if not result:
        return 'empty'
    else:
        return result


def get_image(sp, url_root):

    result = ''
    img_div = sp.find_all('div', class_='inner posrltv')
    for n in img_div:
        img = n.findChild('img')
        if img:
            result = img['src']
    if not result:
        return 'empty'
    else:
        return urllib.parse.urljoin(url_root, result)


def get_calory(sp):

    result = ''
    calory = sp.find_all('span')
    for n in calory:
        nc = n.findChild('strong')
        if nc:
            result = nc.text

    if not result:
        return 'empty'
    else:
        return result


# def get_salt(sp):
#
#     salt = sp.find_all('dl', attrs={'data-label': 'salt'})
#     if not salt:
#         return 'empty'
#     else:
#         return salt[0].find("dd").text


def get_search_result(search_root, url_root, word, max_loop, proxy=True):

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
        div = sp.find_all('div', id='sch_list')
        li_tag = div[0].find_all('li')
        for li in li_tag:
            a_tag = li.findChild('a')
            if a_tag is not None:
                a = a_tag.attrs.get('href')
                if 'detail' in a :
                    result.append(urllib.parse.urljoin(url_root, a))
        result = list(set(result))
        search_result += result

    return search_result


if __name__ == '__main__':

    category_list = []
    food_type_list = []
    url_list = []
    title_list = []
    calory_list = []
    salt_list = []
    img_list = []

    search_root = 'https://erecipe.woman.excite.co.jp/search/{}/?page='
    url_root = 'https://erecipe.woman.excite.co.jp/'
    save_file_name = 'excite.csv'

    max_search_loop = 10

    for key, value in tqdm(food_dict.items()):
        for word in tqdm(value):
            search_result = get_search_result(search_root, url_root, word, max_search_loop, proxy=False)
            for item in tqdm(search_result):
                url_list.append(item)
                sp = get_source(item, proxy=False)
                if sp:
                    category_list.append(key)
                    food_type_list.append(word)
                    title_list.append(get_title(sp))
                    calory_list.append(get_calory(sp))
                    # salt_list.append(get_salt(sp))
                    img_list.append(get_image(sp, url_root))

            df = pd.DataFrame({'category': category_list,
                               'type': food_type_list,
                               'title': title_list,
                               'calorie': calory_list,
                               # 'salt': salt_list,
                               'img': img_list,
                               'url': url_list})
            if os.path.exists(save_file_name):
                df.to_csv(save_file_name, index=False, header=False, mode='a')
            else:
                df.to_csv(save_file_name, index=False, header=True)
