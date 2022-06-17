import socket
import time

log_directory = '/vol/tensusers/mbentum/AUDIOSERVER/LOG/'

def make_activity_log_filename(device):
    name = socket.gethostname()
    filename = log_directory + name + '_' + str(device)
    return filename

def log(line, device):
    t = time.strftime('%Y-%m-%d %H:%M:%S') + '\t' + str(int(time.time())) + '\t'
    line = t + line
    with open(make_activity_log_filename(device), 'a') as fout:
        fout.write(line + '\n')
