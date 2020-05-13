#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import os

from xlwt import Workbook

test_list = []


def collection(file_name):
    quartz_file = open('../data/' + file_name, 'r')
    save_file_name = '../data/test_' + time.strftime('%Y.%m.%d', time.localtime(time.time())) + '.xls'
    if os.path.exists(save_file_name):
        os.remove(save_file_name)

    line_lines = quartz_file.readlines()

    xls = Workbook(encoding='utf-8')
    sheet = xls.add_sheet('trigger_bpm')
    # 生成表头
    sheet.write(0, 0, "a")
    sheet.write(0, 1, "b")
    sheet.write(0, 2, "c")
    sheet.write(0, 3, "d")
    sheet.write(0, 4, "e")
    xls.set_colour_RGB(8, 169, 169, 169)

    for i in range(0, line_lines.__len__()):
        line = line_lines[i]

        line_list = line.split('|')
        id = line_list[0]
        name = line_list[1]
        _id = line_list[2]
        enable = line_list[3]
        quartz = line_list[4]

        if id in test_list:
            continue
        sheet.write(i + 1, 0, id)
        sheet.write(i + 1, 1, name)
        sheet.write(i + 1, 2, _id)
        sheet.write(i + 1, 3, enable)
        sheet.write(i + 1, 4, quartz)

        if not line:
            break

    xls.save(save_file_name)
    quartz_file.close()


if __name__ == '__main__':
    collection('1')
