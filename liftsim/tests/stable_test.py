# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""

qa test for elevators

Authors: likejiao(likejiao@baidu.com)
Date:    2019/07/08 19:30:16
"""

import os
import sys

def main():
    print('stable test')
    print('python path: ', sys.executable)
    os.system(sys.executable + ' tests/qa_test.py 2019 > result1')
    os.system(sys.executable + ' tests/qa_test.py 2019 > result2')
    os.system('diff result1 result2 > stable_diff')
    with open('result1', 'r') as res1_f, open('result2', 'r') as res2_f:
        res1_lines_list = res1_f.readlines()
        res2_lines_list = res2_f.readlines()
        assert len(res1_lines_list) == len(res2_lines_list)
        count = len(res1_lines_list)
        for i in range(count):
            line1 = res1_lines_list[i]
            line2 = res2_lines_list[i]
            if line1.find('MainThread'):
                continue
            line1_find = line1.find('world_time')
            line2_find = line2.find('world_time')
            if (line1_find != -1 or line2_find != -1):
                if (line1_find != -1 and line2_find != -1):
                    line1 = line1[line1_find:]
                    line2 = line2[line2_find:]
                else:
                    print('line1:', line1)
                    print('line2:', line2)
                    print('line', i, 'please check')
                    exit(1)
            assert(line1 == line2)
    print('stable test finish, well done')


if __name__ == "__main__":
    main()