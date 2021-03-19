#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : PRQA 任务数据标注小程序
@Author       : Qinghe Li
@Create time  : 2020-12-16 11:25:29
@Last update  : 2020-12-21 16:48:57
"""

import hashlib
import json
import os
import random
import sys
import time
import tkinter as tk
import tkinter.messagebox
import tkinter.ttk as ttk
import uuid
from importlib import reload
import threading

import requests

reload(sys)
LOG_LINE_NUM = 0

YOUDAO_URL = "https://openapi.youdao.com/api"
APP_KEY = "54403535666c5a49"
APP_SECRET = "vR4U7WbVHyVJ9MH9INYLsmbHFpoktHUJ"


class Translator():
    def __init__(self, youdao_url, app_key, app_secret):
        self.youdao_url = youdao_url
        self.app_key = app_key
        self.app_secret = app_secret

    def encrypt(self, signStr):
        hash_algorithm = hashlib.sha256()
        hash_algorithm.update(signStr.encode("utf-8"))
        return hash_algorithm.hexdigest()

    def truncate(self, q):
        if q is None:
            return None
        size = len(q)
        return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]

    def do_request(self, data):
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        return requests.post(self.youdao_url, data=data, headers=headers)

    def translator(self, sentence):
        data = {}
        data["from"] = "en"
        data["to"] = "zh-CHS"
        data["signType"] = "v3"
        curtime = str(int(time.time()))
        data["curtime"] = curtime
        salt = str(uuid.uuid1())
        signStr = self.app_key + self.truncate(sentence) + salt + curtime + self.app_secret
        sign = self.encrypt(signStr)
        data["appKey"] = self.app_key
        data["q"] = sentence
        data["salt"] = salt
        data["sign"] = sign

        response = self.do_request(data)
        return json.loads(response.content.decode("utf-8"))["translation"][0]

    
class App_UI(tk.Frame):
    # 实现界面生成功能, 具体事件处理代码在子类APP中。
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master.title("PRQA 任务数据标注 v1.5")
        self.master.geometry("1200x900")
        self.createWidgets()

    def createWidgets(self):
        self.top = self.winfo_toplevel()

        # 文件加载与保存UI
        self.frame_file = ttk.LabelFrame(self.top, text="文件信息")
        self.frame_file.place(relx=0.01, rely=0.01, relwidth=0.98, relheight=0.08)
        # 数据标注区UI
        self.frame_label_region = ttk.LabelFrame(self.top, text="数据标注区")
        self.frame_label_region.place(relx=0.01, rely=0.10, relwidth=0.98, relheight=0.73)
        # 标注数据统计信息UI
        self.frame_statistic = ttk.LabelFrame(self.top, text="当前标注数据统计")
        self.frame_statistic.place(relx=0.01, rely=0.84, relwidth=0.48, relheight=0.15)
        # 操作日志模块
        self.frame_log = ttk.LabelFrame(self.top, text="操作日志")
        self.frame_log.place(relx=0.51, rely=0.84, relwidth=0.48, relheight=0.15)

        self.load_and_save_ui()
        self.label_data_ui()
        self.statistic_data_ui()
        self.log_info_ui()

    def load_and_save_ui(self):
        # 文件加载与保存
        self.label_load = ttk.Label(self.frame_file, text="导入文件路径")
        self.label_load.place(relx=0.02, rely=0.5, relwidth=0.08, relheight=0.9, anchor="w")

        self.text_loadpath = tk.Text(self.frame_file, height=1)
        self.text_loadpath.place(relx=0.10, rely=0.5, relwidth=0.30, relheight=0.8, anchor="w")

        self.button_load = ttk.Button(self.frame_file, text="加载", command=self.button_load_cmd)
        self.button_load.place(relx=0.42, rely=0.5, relwidth=0.06, relheight=0.9, anchor="w")

        self.label_save = ttk.Label(self.frame_file, text="导出文件路径")
        self.label_save.place(relx=0.52, rely=0.5, relwidth=0.08, relheight=0.9, anchor="w")

        self.text_savepath = tk.Text(self.frame_file, height=1)
        self.text_savepath.place(relx=0.60, rely=0.5, relwidth=0.30, relheight=0.8, anchor="w")

        self.button_save = ttk.Button(self.frame_file, text="保存", command=self.button_save_cmd)
        self.button_save.place(relx=0.92, rely=0.5, relwidth=0.06, relheight=0.9, anchor="w")

    def label_data_ui(self):
        # 数据标注区
        self.label_question = ttk.Label(self.frame_label_region, text="问题 :")
        self.label_question.place(relx=0.02, rely=0.057, relwidth=0.08, relheight=0.075, anchor="w")
        self.text_question_en = tk.Text(self.frame_label_region)
        self.text_question_en.place(relx=0.06, rely=0.057, relwidth=0.4, relheight=0.075, anchor="w")
        self.text_question_zh = tk.Text(self.frame_label_region)
        self.text_question_zh.place(relx=0.465, rely=0.057, relwidth=0.4, relheight=0.075, anchor="w")

        self.class_texts = ["NA", "No", "Yes"]
        self.values = [0, 1, 2]

        self.select_answer_q = tk.IntVar()
        self.options_q = [None] * 3
        for j, (t, v) in enumerate(zip(self.class_texts, self.values)):
            self.options_q[j] = ttk.Radiobutton(self.frame_label_region, text=t, value=v, variable=self.select_answer_q)
            self.select_answer_q.set(9)
            self.options_q[j].place(relx=(0.875 + j * 0.04), rely=0.02, relwidth=0.04, relheight=0.075)

        self.select_answers = [None] * 10
        self.label_reviews = [None] * 10
        self.text_review_ens = [None] * 10
        self.text_review_zhs = [None] * 10
        self.options = [None] * 10
        for i in range(10):
            self.label_reviews[i] = ttk.Label(self.frame_label_region, text=("评论" + str(i + 1) + ":"))
            self.label_reviews[i].place(relx=0.02, rely=(0.14 + 0.08 * i), relwidth=0.08, relheight=0.075, anchor="w")
            self.text_review_ens[i] = tk.Text(self.frame_label_region)
            self.text_review_ens[i].place(relx=0.06, rely=(0.14 + 0.08 * i), relwidth=0.4, relheight=0.075, anchor="w")
            self.text_review_zhs[i] = tk.Text(self.frame_label_region)
            self.text_review_zhs[i].place(relx=0.465, rely=(0.14 + 0.08 * i), relwidth=0.4, relheight=0.075, anchor="w")

            self.options[i] = [None] * 3
            self.select_answers[i] = tk.IntVar()
            for j, (t, v) in enumerate(zip(self.class_texts, self.values)):
                self.options[i][j] = ttk.Radiobutton(
                    self.frame_label_region, text=t, value=v, variable=self.select_answers[i])
                self.select_answers[i].set(9)
                self.options[i][j].place(relx=(0.875 + j * 0.04), rely=(0.1 + 0.08 * i), relwidth=0.04, relheight=0.075)

        self.skip = ttk.Button(self.frame_label_region, text="跳过", command=self.skip_cmd)
        self.skip.place(relx=0.6, rely=0.925, relwidth=0.08, relheight=0.05)
        self.next = ttk.Button(self.frame_label_region, text="下一条", command=self.next_cmd)
        self.next.place(relx=0.7, rely=0.925, relwidth=0.08, relheight=0.05)

    def statistic_data_ui(self):
        # 标注数据统计信息
        self.unlabeled_num = ttk.Label(self.frame_statistic, text="未标注数量:")
        self.unlabeled_num.place(relx=0.03, rely=0.22, relwidth=0.15, relheight=0.22, anchor="w")
        self.text_unlabeled = tk.Text(self.frame_statistic)
        self.text_unlabeled.place(relx=0.21, rely=0.22, relwidth=0.1, relheight=0.22, anchor="w")
        self.labeled_num = ttk.Label(self.frame_statistic, text="已标注数量:")
        self.labeled_num.place(relx=0.36, rely=0.22, relwidth=0.15, relheight=0.22, anchor="w")
        self.text_labeled = tk.Text(self.frame_statistic)
        self.text_labeled.place(relx=0.54, rely=0.22, relwidth=0.1, relheight=0.22, anchor="w")

        self.num_na_q = ttk.Label(self.frame_statistic, text="NA问题数量:")
        self.num_na_q.place(relx=0.03, rely=0.5, relwidth=0.15, relheight=0.22, anchor="w")
        self.text_na_q = tk.Text(self.frame_statistic)
        self.text_na_q.place(relx=0.21, rely=0.5, relwidth=0.1, relheight=0.22, anchor="w")
        self.num_no_q = ttk.Label(self.frame_statistic, text="No问题数量:")
        self.num_no_q.place(relx=0.36, rely=0.5, relwidth=0.15, relheight=0.22, anchor="w")
        self.text_no_q = tk.Text(self.frame_statistic)
        self.text_no_q.place(relx=0.54, rely=0.5, relwidth=0.1, relheight=0.22, anchor="w")
        self.num_yes_q = ttk.Label(self.frame_statistic, text="Yes问题数量:")
        self.num_yes_q.place(relx=0.68, rely=0.5, relwidth=0.15, relheight=0.22, anchor="w")
        self.text_yes_q = tk.Text(self.frame_statistic)
        self.text_yes_q.place(relx=0.86, rely=0.5, relwidth=0.1, relheight=0.22, anchor="w")

        self.num_na_r = ttk.Label(self.frame_statistic, text="NA评论数量:")
        self.num_na_r.place(relx=0.03, rely=0.77, relwidth=0.15, relheight=0.22, anchor="w")
        self.text_na_r = tk.Text(self.frame_statistic)
        self.text_na_r.place(relx=0.21, rely=0.77, relwidth=0.1, relheight=0.22, anchor="w")
        self.num_no_r = ttk.Label(self.frame_statistic, text="No评论数量:")
        self.num_no_r.place(relx=0.36, rely=0.77, relwidth=0.15, relheight=0.22, anchor="w")
        self.text_no_r = tk.Text(self.frame_statistic)
        self.text_no_r.place(relx=0.54, rely=0.77, relwidth=0.1, relheight=0.22, anchor="w")
        self.num_yes_r = ttk.Label(self.frame_statistic, text="Yes评论数量:")
        self.num_yes_r.place(relx=0.68, rely=0.77, relwidth=0.15, relheight=0.22, anchor="w")
        self.text_yes_r = tk.Text(self.frame_statistic)
        self.text_yes_r.place(relx=0.86, rely=0.77, relwidth=0.1, relheight=0.22, anchor="w")

    def log_info_ui(self):
        # 操作日志模块
        self.text_log = tk.Text(self.frame_log)
        self.text_log.place(relx=0, rely=0, relwidth=1, relheight=1)
   
    # 更新面板文本
    def update_labeled_text(self):
        self.text_question_en.delete(1.0, tk.END)
        self.text_question_zh.delete(1.0, tk.END)
        self.text_question_en.insert(1.0, self.trans_question_en)
        self.text_question_zh.insert(1.0, self.trans_question_zh)
        self.select_answer_q.set(9)

        for i in range(len(self.trans_review_ens)):
            self.text_review_ens[i].delete(1.0, tk.END)
            self.text_review_ens[i].insert(1.0, self.trans_review_ens[i])
            self.text_review_zhs[i].delete(1.0, tk.END)
            self.text_review_zhs[i].insert(1.0, self.trans_review_zhs[i])
            self.select_answers[i].set(9)

    # 更新当前统计信息
    def update_statistic_data(self):
        self.text_unlabeled.delete(1.0, tk.END)
        self.text_unlabeled.insert(1.0, self.unlabeled_num)
        self.text_labeled.delete(1.0, tk.END)
        self.text_labeled.insert(1.0, self.labeled_num)
        self.text_na_q.delete(1.0, tk.END)
        self.text_na_q.insert(1.0, self.na_q_num)
        self.text_no_q.delete(1.0, tk.END)
        self.text_no_q.insert(1.0, self.no_q_num)
        self.text_yes_q.delete(1.0, tk.END)
        self.text_yes_q.insert(1.0, self.yes_q_num)
        self.text_na_r.delete(1.0, tk.END)
        self.text_na_r.insert(1.0, self.na_r_num)
        self.text_no_r.delete(1.0, tk.END)
        self.text_no_r.insert(1.0, self.no_r_num)
        self.text_yes_r.delete(1.0, tk.END)
        self.text_yes_r.insert(1.0, self.yes_r_num)


class App(App_UI):
    # 实现具体的事件处理回调函数,界面生成代码在App_UI中
    def __init__(self, master=None):
        App_UI.__init__(self, master)
        self.translator = Translator(youdao_url=YOUDAO_URL, app_key=APP_KEY, app_secret=APP_SECRET)

    # 获取当前时间
    def get_current_time(self):
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        return current_time

    # 日志动态打印
    def write_log_to_text(self, logmsg):
        global LOG_LINE_NUM
        current_time = self.get_current_time()
        logmsg_in = str(current_time) + " " + str(logmsg) + "\n"  # 换行
        if LOG_LINE_NUM <= 4:
            self.text_log.insert(tk.END, logmsg_in)
            LOG_LINE_NUM = LOG_LINE_NUM + 1
        else:
            self.text_log.delete(1.0, 2.0)
            self.text_log.insert(tk.END, logmsg_in)

    # 取下一条数据进行翻译
    def run_translate(self):
        self.next_item = self.unlabeled_items_list.pop()
        self.trans_question_en = self.next_item["questionText"]
        self.trans_question_zh = self.translator.translator(self.trans_question_en)

        self.trans_review_ens = []
        self.trans_review_zhs = []
        review_snippets = self.next_item["review_snippets"]
        for i, review_en in enumerate(review_snippets):
            self.trans_review_ens.append(review_en)
            self.trans_review_zhs.append(self.translator.translator(review_en))
        # 设置子线程任务完成标志
        self.flag = 1

    # 文件加载
    def button_load_cmd(self, event=None):
        # 标注相关参数初始化
        self.unlabeled_items_list = []              # 未标注的数据集
        self.labeled_items_list = []                # 完成标注的数据集
        self.unlabeled_num = 0                      # 未标注的数量
        self.labeled_num = 0                        # 已标注的数量
        self.na_q_num = 0                           # 标注为NA的问题数
        self.no_q_num = 0                           # 标注为No的问题数
        self.yes_q_num = 0                          # 标注为Yes的问题数
        self.na_r_num = 0                           # 标注为NA的评论条数
        self.no_r_num = 0                           # 标注为No的评论条数
        self.yes_r_num = 0                          # 标注为Yes的评论条数

        load_path = self.text_loadpath.get(1.0, tk.END).strip().replace("\n", "")
        current_detail = load_path.split(".jsonl")[0] + "_current_detail.json"

        if load_path:
            try:
                with open(load_path, "r") as f_load:
                    for line in f_load.readlines():
                        item = json.loads(line)
                        self.unlabeled_items_list.append(item)
                self.unlabeled_num = len(self.unlabeled_items_list)
                random.shuffle(self.unlabeled_items_list)

                # 多线程加速翻译第一条数据
                t = threading.Thread(target=self.run_translate)
                t.setDaemon(True)
                t.start()

                if os.path.exists(current_detail):
                    with open(current_detail, "r") as f_detail:
                        dataset_detail = json.load(f_detail)
                    self.unlabeled_num = dataset_detail["unlabeled_num"]
                    self.labeled_num = dataset_detail["labeled_num"]
                    self.na_q_num = dataset_detail["na_q_num"]
                    self.no_q_num = dataset_detail["no_q_num"]
                    self.yes_q_num = dataset_detail["yes_q_num"]
                    self.na_r_num = dataset_detail["na_r_num"]
                    self.no_r_num = dataset_detail["no_r_num"]
                    self.yes_r_num = dataset_detail["yes_r_num"]

            except BaseException:
                self.text_loadpath.delete(1.0, tk.END)
                tkinter.messagebox.showerror(title="文件加载结果", message="文件加载路径错误,请重新输入...")
            else:
                # 等待第一条数据翻译完成后进行面板显示
                t.join()
                self.current_item = self.next_item
                self.update_labeled_text()
                self.update_statistic_data()
                self.write_log_to_text("INFO: 文件加载完成...")
                tkinter.messagebox.showinfo(title="文件加载结果", message="加载文件完成....")

                if len(self.unlabeled_items_list) != 0:
                    # 启动新线程翻译下一条数据
                    t = threading.Thread(target=self.run_translate)
                    t.start()
        else:
            tkinter.messagebox.showerror(title="文件加载结果", message="文件加载路径为空,请重新输入...")

    # 标注文件保存
    def button_save_cmd(self, event=None):
        current_time = self.get_current_time()
        save_path = self.text_savepath.get(1.0, tk.END).strip().replace("\n", "")

        if len(self.unlabeled_items_list) != 0:
            self.unlabeled_items_list.append(self.current_item)

        if save_path:
            labeled_save_path = save_path + "_labeled_data.jsonl"               # 保存方式为追加保存
            remained_save_path = save_path + "_unlabeled_data.jsonl"
            current_dataset_detail = save_path + "_unlabeled_data_current_detail.json"

            try:
                with open(labeled_save_path, "a") as f_labeled:
                    for item in self.labeled_items_list:
                        f_labeled.write(json.dumps(item))
                        f_labeled.write("\n")

                with open(remained_save_path, "w") as f_unlabeled:
                    for item in self.unlabeled_items_list:
                        f_unlabeled.write(json.dumps(item))
                        f_unlabeled.write("\n")

                with open(current_dataset_detail, "w") as f_detail:
                    data_detail = {
                        "labeled_num": self.labeled_num,
                        "unlabeled_num": self.unlabeled_num,
                        "na_q_num": self.na_q_num,
                        "no_q_num": self.no_q_num,
                        "yes_q_num": self.yes_q_num,
                        "na_r_num": self.na_r_num,
                        "no_r_num": self.no_r_num,
                        "yes_r_num": self.yes_r_num,
                        "save_time": current_time
                    }
                    json.dump(data_detail, f_detail)

                self.write_log_to_text("INFO: 文件保存完成...")
                tkinter.messagebox.showinfo(title="文件保存结果", message="保存文件完成....")
            except BaseException:
                self.text_savepath.delete(1.0, tk.END)
                tkinter.messagebox.showerror(title="文件保存结果", message="文件保存路径错误,请重新输入...")
        else:
            tkinter.messagebox.showerror(title="文件保存结果", message="文件保存路径为空,请重新输入...")

    # 跳过该条问题
    def skip_cmd(self, event=None):
        # 直接加载显示下一条数据
        while(not self.flag):
            time.sleep(1)
        self.current_item = self.next_item
        self.flag = 0

        self.unlabeled_num -= 1
        self.update_statistic_data()
        self.update_labeled_text()
        try:
            if len(self.unlabeled_items_list) == 0:
                raise Exception("未标注数据集为空...")
        except Exception as e:
            self.write_log_to_text("INFO: {}".format(e))
        else:
            t = threading.Thread(target=self.run_translate)
            t.start()

    # 完成本条标注,跳转下一条
    def next_cmd(self, event=None):
        # 首先检查和处理当前条的标注结果
        try:
            if self.select_answer_q.get() == 0:
                self.na_q_num += 1
            elif self.select_answer_q.get() == 1:
                self.no_q_num += 1
            elif self.select_answer_q.get() == 2:
                self.yes_q_num += 1
            else:
                raise Exception("问题存在标注错误: {}".format(self.select_answer_q.get()))

            l_rs = []
            for i in range(10):
                if self.select_answers[i].get() == 0:
                    l_rs.append(0)
                    self.na_r_num += 1
                elif self.select_answers[i].get() == 1:
                    l_rs.append(1)
                    self.no_r_num += 1
                elif self.select_answers[i].get() == 2:
                    l_rs.append(2)
                    self.yes_r_num += 1
                else:
                    raise Exception("评论存在标注错误: {}".format(l_rs))
        except Exception as e:
            self.write_log_to_text("ERROR: {}".format(e))
        else:
            self.current_item["review_labels"] = l_rs
            self.current_item["answer_label"] = self.select_answer_q.get()
            self.labeled_items_list.append(self.current_item)
            self.unlabeled_num -= 1
            self.labeled_num += 1
            self.update_statistic_data()

            # 等待子线程完成下一条数据的翻译
            while(not self.flag):
                time.sleep(1)
            self.current_item = self.next_item
            self.flag = 0

            # 加载显示下一条数据
            self.update_labeled_text()
            try:
                if len(self.unlabeled_items_list) == 0:
                    raise Exception("未标注数据集为空...")
            except Exception as e:
                self.write_log_to_text("INFO: {}".format(e))
            else:
                t = threading.Thread(target=self.run_translate)
                t.start()


if __name__ == "__main__":
    # 实例化父窗口
    top = tk.Tk()
    # 父窗口进入事件循环, 保持窗口运行
    App(top).mainloop()
