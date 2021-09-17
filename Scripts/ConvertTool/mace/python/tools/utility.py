# -*- coding: utf-8 -*-

import six
import sys
import time

def annPrint(*contents, fprint=False):
    if fprint:
        six.print_(*contents)


class ShowProcess(object):
    def __init__(self, max_steps, max_arrow=50, fprint=True):
        self.max_steps = max_steps
        self.i = 0
        self.max_arrow = max_arrow
        self.start = time.time()
        self.eta = 0.0
        self.total_time = 0.0
        self.last_time = self.start
        self.fprint = fprint

    def elapsed_time(self):
        self.last_time = time.time()
        return self.last_time - self.start

    def calc_eta(self):
        elapsed = self.elapsed_time()
        if self.i == 0 or elapsed < 0.001:
            return None
        rate = float(self.i) / elapsed
        self.eta = (float(self.max_steps) - float(self.i)) / rate

    def get_time(self, _time):
        if (_time < 86400):
            return time.strftime("%H:%M:%S", time.gmtime(_time))
        else:
            s = (str(int(_time // 3600)) + ':' +
                 time.strftime("%M:%S", time.gmtime(_time)))
            return s

    def show_process(self, i=None):
        if self.fprint:
            if i is not None:
                self.i = i
            else:
                self.i += 1
            self.calc_eta()
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