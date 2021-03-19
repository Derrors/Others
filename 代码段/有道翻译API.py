#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : 有道翻译 API
@Author       : Qinghe Li
@Create time  : 2020-12-15 20:40:54
@Last update  : 2020-12-15 21:13:55
"""

import hashlib
import json
import sys
import time
import uuid
from importlib import reload

import requests

reload(sys)

YOUDAO_URL = "https://openapi.youdao.com/api"
APP_KEY = "54403535666c5a49"
APP_SECRET = "i1jqxDpnk2guQZ5atBDBmvA0w9B05jdF"


def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode("utf-8"))
    return hash_algorithm.hexdigest()


def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


def do_request(data):
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    return requests.post(YOUDAO_URL, data=data, headers=headers)


def translator(sentence):
    data = {}
    data["from"] = "en"
    data["to"] = "zh-CHS"
    data["signType"] = "v3"
    curtime = str(int(time.time()))
    data["curtime"] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(sentence) + salt + curtime + APP_SECRET
    sign = encrypt(signStr)
    data["appKey"] = APP_KEY
    data["q"] = sentence
    data["salt"] = salt
    data["sign"] = sign

    response = do_request(data)
    return json.loads(response.content.decode("utf-8"))["translation"][0]
