import os
import time
from datetime import datetime
import json
import pandas as pd

class Logger:
    def __init__(self, proc, cpu_num):
        try:
            os.makedirs('../../../examples/forecaster_logs')
        except:
            pass

        self.proc = proc
        self.cpu_num = cpu_num

        self.filename = None
        self.log = None
        self.run = False
        self.show = True

        self.event_name = ''
        self.text = ''

        self.logline = {'timestamp': '',
                        'event_name': None,
                        'text': '',
                        'cpu%': None,
                        'ram_mb': None
                        }

    def create(self, filename, rewrite=False):
        self.filename = 'forecaster_logs/' + filename
        if rewrite:
            self.log = self.open('w')
        else:
            self.log = self.open('a')


    def event_init(self, event_name='', text=''):
        self.event_name = event_name
        self.text = text

    def show_off(self):
        self.show = False

    def show_on(self):
        self.show = True

    def daemon_logger(self, show=True):
        self.show = show
        while self.run:
            d = self.proc.as_dict(attrs=['cpu_percent', 'memory_info', 'memory_percent'])

            self.logline['cpu%'] = round(d['cpu_percent']/self.cpu_num, 2)
            self.logline['ram_mb'] = round(d['memory_info'].rss / 1024 ** 2, 2)

            self.logline['timestamp'] = str(datetime.utcnow())
            self.logline['event_name'] = self.event_name
            self.logline['text'] = self.text

            self.log.write(json.dumps(self.logline))
            self.log.write('\n')
            self.log.flush()
            os.fsync(self.log.fileno())

            if self.show:
                line = '{0}, event_type: {1}, text: {2}, CPU: {3}%, RAM: {4} Mb'.format(self.logline['timestamp'],
                                                                                        self.logline['event_name'],
                                                                                        self.logline['text'],
                                                                                        self.logline['cpu%'],
                                                                                        self.logline['ram_mb'])
                print(line)
            time.sleep(100/1000)

    def open(self, how):
        return open(self.filename, how)

    def close(self):
        self.log.close()

    def get_resources(self, event_name='all'):
        logdata = self.parse_to_dataframe(self.filename)
        resources = {'duration_sec': self.get_event_duration(logdata, event_name=event_name)}
        for res in ['cpu%', 'ram_mb']:
            for stat_param in ['min', 'mean', 'max']:
                resources[res + '_' + stat_param] = self.get_resource_stat(logdata, event_name=event_name,
                                                                           res=res, stat_param=stat_param)
        return resources

    @staticmethod
    def parse_to_dataframe(filename):
        return pd.read_json(filename, lines=True)

    @staticmethod
    def get_event_duration(data, event_name='all'):
        if event_name == 'all':
            indexA = 0
            indexB = data.index[-1]

        else:
            if event_name not in data['event_name'].unique():
                print('Wrong event name')
                exit()

            event_data_indices = data[data['event_name'] == event_name].index
            indexA = event_data_indices[0]
            indexB = event_data_indices[-1]

        start = pd.to_datetime(data.loc[indexA, 'timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
        end = pd.to_datetime(data.loc[indexB, 'timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
        duration = end - start
        duration = duration.total_seconds()
        return duration

    @staticmethod
    def get_resource_stat(data, res, stat_param, event_name='all'):

        event_data, res_data = None, None

        if event_name == 'all':
            event_data = data
        else:
            try:
                event_data = data[data['event_name']==event_name]
            except:
                print('Wrong event name')
                exit()

        try:
            res_data = event_data[res]
        except:
            print('Wrong resource name')
            exit()

        if stat_param == 'min':
            return round(res_data.min(), 3)
        elif stat_param == 'mean':
            return round(res_data.mean(), 3)
        elif stat_param == 'max':
            return round(res_data.max(), 3)
        else:
            print('Wrong statistic param')
            exit()




