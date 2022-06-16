import gpu_logger
import os 
import time

d ={'mlp03':'rarity','mlp10':'thunderlane','mlp12':'snips','mlp13':'mistmane'}
pony_dict = d

class Selector:
    '''select gpu that is not in use based on log files.
    '''
    def __init__(self, to_old = 600, free_memory = 15000, max_load = .1):
        '''select available gpu based on restrictions
        to_old          cut of for to old log data default = 10 minutes
                        log data should be update every 5 minutes with
                        a cron job
        free_memory     minimal amount of free gpu memory
        max_load        maximum load of the gpu processing power default 10%
        '''
        self.to_old = to_old
        self.free_memory = free_memory
        self.max_load = max_load
        self.filename = gpu_logger.make_filename()
        self.table = load_table(self.filename)
        self.ok = True if self.table else False
        if self.ok: self._make_gpus()
        if self.navailable_gpus == 0 and self.nbusy_gpus == 0: self.ok = False

    def __repr__(self):
        m = 'gpu selector | available gpus: ' + str(self.navailable_gpus) 
        m += ' | busy gpus: ' + str(self.nbusy_gpus)
        m += '\nselected gpu:\n'
        if self.selected_gpu: m += self.selected_gpu.__repr__()
        else: m += 'none'
        return m

    def _make_gpus(self):
        '''load gpu object based on info in the log file.
        gpus are sorted on available free memory
        '''
        self.gpus = []
        self.no_memory = []
        self.overload = []
        for line in self.table[::-1]:
            gpu = Gpu(line)
            if not gpu.ok:continue
            if gpu.free_memory < self.free_memory: 
                if gpu not in self.no_memory: self.no_memory.append(gpu)
                continue
            if gpu.load > self.max_load: 
                if gpu not in self.overload:self.overload.append(gpu)
                continue
            if gpu.delta_time > self.to_old: break
            if gpu not in self.gpus: self.gpus.append(gpu)
        self.gpus = sorted(self.gpus, reverse = True)

    @property
    def selected_gpu(self):
        '''select available gpu, if available'''
        if len(self.gpus) > 0: return self.gpus[0]
        return False

    @property
    def navailable_gpus(self):
        '''number of available gpus.'''
        return len(self.gpus)

    @property
    def nbusy_gpus(self):
        '''number of busy gpus.'''
        return len(self.overload) + len(self.no_memory)

class Gpu:
    '''object that contains info about a single gpu device.'''
    def __init__(self,line):
    '''object that contains info about a single gpu device.
    line    one line from the gpu logger file
    '''
        self.line = line.split('\t')
        if len(self.line) == 8:
            self._read_line()
        else:self.ok = False

    def __repr__(self):
        m = self.name + ' ' + self.pony_name.ljust(12) + ' '
        m += str(self.device).ljust(3) + ' ' 
        m += self.readable_time.ljust(9)
        m += ' ' + str(self.delta_time).ljust(5) 
        m += ' ' + str(self.load).ljust(5)
        m += ' ' + str(self.memory_load).ljust(5)
        return m

    def __eq__(self,other):
        '''a gpu is identical if the server name and device number are equal.'''
        if type(self) != type(other): return False
        return self.name == other.name and self.device == other.device

    def __gt__(self,other):
        '''a gpu is large if it has more memory available.'''
        return self.free_memory > other.free_memory

    def _read_line(self):
        '''loads the info in the gpu logger line in attributes of the object.'''
        names = 'name,device,load,memory_load,memory_total,memory_used'
        names += ',temperature,time'
        self.names = names.split(',')
        int_names='device,memory_total,memory_used,temperature'.split(',')
        float_names = 'load,memory_load,time'.split(',')
        for name, value in zip(self.names,self.line):
            if name in int_names: setattr(self,name,int(value))
            elif name in float_names: setattr(self,name,float(value))
            else: setattr(self,name,value)
        self.ok = True
        self.pony_name = pony_dict[self.name]
           
    @property
    def readable_time(self):
        return time.strftime('%H:%M:%S',time.localtime(self.time))

    @property
    def delta_time(self):
        return int(time.time() - self.time)

    @property
    def free_memory(self):
        return self.memory_total - self.memory_used


def load_table(filename = ''):
    '''load the gpu logger file
    this file is appended every 5 minutes; most recent information at the bottom
    updating of the log file is done with a crontab on pipsqueak
    '''
    if not filename:
        filename = gpu_logger.make_filename()
    if os.path.isfile(filename):
        with open(filename) as fin:
            table = fin.read().split('\n')
        return table
    else: return False
