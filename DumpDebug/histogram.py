import re
import os
import sys
import time
import json
import pickle
import shutil
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


class ShowProcess(object):
    def __init__(self, max_steps, max_arrow=50):
        self.max_steps = max_steps
        self.i = 0
        self.max_arrow = max_arrow
        self.start = time.time()
        self.total_time = 0.0
        self.last_time = self.start

    def elapsed_time(self):
        self.last_time = time.time()
        return self.last_time - self.start

    def get_time(self, _time):
        if (_time < 86400):
            return time.strftime("%H:%M:%S", time.gmtime(_time))
        else:
            s = (str(int(_time // 3600)) + ':' +
                 time.strftime("%M:%S", time.gmtime(_time)))
            return s

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        if num_arrow < 2:
            process_bar = '\r' + '[' + '>' * num_arrow + ' ' * num_line + ']' \
                          + '%.2f' % percent + '%'
        elif num_arrow < self.max_arrow:
            process_bar = '\r' + '[' + '=' * (num_arrow-1) + '>' + ' ' * num_line + ']' \
                          + '%.2f' % percent + '%'
        else:
            process_bar = '\r' + '[' + '=' * num_arrow + ' ' * num_line + ']'\
                          + '%.2f' % percent + '%'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        self.close()

    def close(self):
        if self.i >= self.max_steps:
            self.total_time = self.elapsed_time()
            print('\nTotal time elapsed: ' + self.get_time(self.total_time))


def is_decimal(data_str):
    try:
        float(data_str)
        return True
    except ValueError:
        return False
    return False


def convert_data(data_str):
    num_list = []
    data_list = data_str.strip().split('\n')
    for data in data_list:
        if is_decimal(data.split(',')[0]):
            num_list.extend([float(i.strip()) for i in data.split(',') if i.strip() != ''])
    return num_list


def histogram_draw(Nums, line, file_data):
    layer_name = line['name']
    save_name = os.path.join(os.getcwd(), 'Histograms', Nums + '.' + layer_name.replace('/', '_') + '.png')
    name = 'name: ' + layer_name
    min_value = min(line['min'])
    max_value = max(line['max'])
    no_str = re.search(name, file_data)
    if no_str is None:
        return
    else:
        i = no_str.end()
    data = []
    data_str = str()
    while True:
        if file_data[i] == '{':
            start = i + 1
        if file_data[i] == '}':
            end = i - 1
            data_str = file_data[start: end]
            data = convert_data(data_str)
            plt.figure()
            plt.hist(data, bins='auto', range=(min_value, max_value))
            # n, bins, patches = plt.hist(data, bins='sqrt', range=(min_value, max_value))
            # plt.plot(0.5 * (bins[1:] + bins[:-1]), n, 'r--')
            plt.axvline(min_value, color='r', linestyle='--', linewidth=0.8)
            plt.axvline(max_value, color='r', linestyle='--', linewidth=0.8)
            plt.xlabel('Values')
            plt.ylabel('Histogram')
            plt.title(layer_name)
            plt.savefig(save_name)
            plt.close()
            break
        i += 1


def renew_folder(folder_name):
    if os.path.exists(folder_name):
        if os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
            os.mkdir(folder_name)
        else:
            os.remove(folder_name)
            os.mkdir(folder_name)
    else:
        os.mkdir(folder_name)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(sys.argv[0], '<dump_data_file> <tensor_min_max.json/.pkl>')
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        file_data = f.read()
    if not os.path.exists(sys.argv[2]):
        raise FileNotFoundError('No such quant_file: {}'.format(sys.argv[2]))
    else:
        try:
            with open(sys.argv[2], 'rb') as f:
                minmax_data = pickle.load(f)
        except pickle.UnpicklingError:
            with open(sys.argv[2], 'r') as f:
                minmax_data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError('tensor_min_max only support JSON or Pickle file.')
    renew_folder('Histograms')
    process_bar = ShowProcess(len(minmax_data))
    for num, line in enumerate(minmax_data):
        histogram_draw(str(num), line, file_data)
        process_bar.show_process()