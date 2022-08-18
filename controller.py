import argument_handler 
import gpu_selector as gpus
import os
import ponies
import sys
import threading
import time

sys.tracebacklimit=0
pony_dict = ponies.pony_dict
log_dir = '/vol/tensusers/mbentum/AUDIOSERVER/LOG/'


def activate_command():
    m='source /vol/tensusers/mbentum/AUDIOSERVER/audioserver_env/bin/activate;'
    return m

def make_ssh_command(gpu,args):
    argument_command = argument_handler.arguments_to_command(args)
    device = gpu.device
    server_name = gpu.name
    m = 'ssh ' + server_name + ' '
    m += '"' + activate_command()
    m += 'python /vol/tensusers/mbentum/AUDIOSERVER/repo/wav2vec2.py '
    m += '-device ' + str(device) + argument_command + '"'
    return m
    

def transcribe(args = None, verbose = False):
    s = gpus.Selector()
    device = s.selected_gpu.device
    server_name = s.selected_gpu.name
    print('selected gpu: ',server_name,pony_dict[server_name],
        ', device:',device)
    ssh_command = make_ssh_command(s.selected_gpu,args)
    if verbose: print(ssh_command)
    start = time.time()
    x = threading.Thread(target = show_log, args = (server_name,device,start))
    x.start()
    os.system(ssh_command)
    show_log(server_name,device,start)

def show_log(server_name,device, start = None):
    if not start: start = time.time()
    filename = log_dir + server_name + '_' + str(device)
    print('showing log',filename)
    if not os.path.isfile(filename):
        m ='cannot show log file, does not exist'
        raise FileNotFoundError(filename,m)
    lines = []
    while True:
        new = _check_log(filename,start,lines)
        done = False
        for line in new:
            print(line)
            lines.append(line)
            if 'setting status: closed' in line: done = True
        if done: 
            print('transcription is done, stop showing log file')
            break
        time.sleep(1)

def _check_log(filename, start_time,read_lines):
    with open(filename) as fin:
        lines = fin.read().split('\n')
    selection = []
    for line in lines:
        if not line:continue
        if line in read_lines: continue
        line_time =  int(line.split('\t')[1]) 
        if line_time > start_time: selection.append(line)
    return selection





if __name__ == '__main__':
    args = argument_handler.transcribe_arguments()
    transcribe(args)
