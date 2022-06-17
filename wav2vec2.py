import argparse
import audio
import glob
import gpu_selector as gpus
from logger import log
from numba import cuda
import os 
import socket
import time
import torch
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor

fremy_model1 = 'FremyCompany/xls-r-2b-nl-v2_lm-5gram-os'

''' 
More info about pipelines for ASR see:
https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline
'''
from transformers import AutomaticSpeechRecognitionPipeline as ap

default_recognizer_dir = "/vol/bigdata2/datasets2/SSHOC-T44-LISpanel-2021/"
default_recognizer_dir += "TEXT_ANALYSIS/homed_lm_recognizers/cgn/"

input_directory = '/vol/tensusers/mbentum/AUDIOSERVER/INPUT/'
output_directory = '/vol/tensusers/mbentum/AUDIOSERVER/OUTPUT/'
status_directory = '/vol/tensusers/mbentum/AUDIOSERVER/STATUS/'

def load_model(recognizer_dir = default_recognizer_dir):
    model = Wav2Vec2ForCTC.from_pretrained(recognizer_dir)
    return model

def load_processor(recognizer_dir = default_recognizer_dir):
    processor = Wav2Vec2Processor.from_pretrained(recognizer_dir)
    return processor

def load_pipeline(recognizer_dir=None, model = None,chunk_length_s = 10, 
    device = -1):
    '''
    loads a pipeline object that can transcribe audio files
    recognizer_dir      directory that stores the wav2vec2 model
    model               preloaded wav2vec model (speeds up loading pipeline)
    chunk_length_s      chunking duration of long audio files
                        wav2vec2 is memory hungry, the pipeline employs
                        a sliding window to handle long audio files
                        and the edge effects from chunking
    '''
    print('using device:',device)
    log('using device: '+ str(device),device)
    if not recognizer_dir: recognizer_dir = default_recognizer_dir
    if not model:
        print('loading model:',recognizer_dir)
        log('loading model: '+ str(recognizer_dir), device)
        model = load_model(recognizer_dir)
    print('loading processor')
    log('loading processor',device)
    p= load_processor(recognizer_dir)
    print('loading pipeline')
    log('loading pipeline',device)
    pipeline = ap(
        feature_extractor =p.feature_extractor,
        model = model,
        tokenizer = p.tokenizer,
        chunk_length_s = chunk_length_s,
        device = device
    )
    return pipeline


def decode_audiofile(filename, pipeline, start=0.0, end=None):
    '''
    transcribe an audio file with pipeline object
    loads the audio with librosa
    '''
    a = audio.load_audio(filename,start,end)
    output = pipeline(a, return_timestamps = 'word')
    return output 

def _make_decoding_output_filename(audio_filename, output_dir):
    ''' 
    make a filename for transcription based on audio_filename 
    and output_dir.
    '''
    if not output_dir: output_dir = output_directory
    filename = audio_filename.replace('.wav','.output')
    filename = output_dir + filename.split('/')[-1]
    return filename

def save_pipeline_output_to_file(output, audio_filename, output_dir = None):
    '''save pipeline output to a text file.'''
    filename = _make_decoding_output_filename(audio_filename, output_dir)
    table = pipeline_output2table(output)
    print('saving to:',filename)
    with open(filename,'w') as fout:
        fout.write(_table2str(table))

def pipeline_output2table(output):
    '''convert pipeline output to table (word\tstart\tend).'''
    table = []
    for d in output['chunks']:
        start, end = d['timestamp']
        table.append([d['text'], start, end])
    return table

def _table2str(table):
    '''convert output table to string.'''
    output = []
    for line in table:
        output.append('\t'.join(list(map(str,line))))
    return '\n'.join(output)

class Transcriber:
    '''transcribe audio files in input_dir.'''
    def __init__(self, model_dir = None, input_dir = None, output_dir = None,
        '''transcribe audio files in input_dir
        model_dir       directory of the wav2vec2 model
        input_dir       directory for audio files that need to be transcribed
        output_dir      directory for output files
        '''
        model = None, pipeline = None, device = -1):
        self.model_dir = model_dir if model_dir else default_recognizer_dir
        self.input_dir = input_dir if input_dir else input_directory
        self.output_dir = output_dir if output_dir else output_directory
        self.device = device
        if pipeline: self.pipeline = pipeline
        elif model: 
            self.model = model
            self.pipeline = load_pipeline(model = model,device = device)
        else: self.pipeline = load_pipeline(recognizer_dir = self.model_dir,
            device = device)
        self.transcribed_audio_files = {}
        self.did_transcription= False
    
    def load_audio_filenames(self):
        self.audio_filenames = glob.glob(self.input_dir + '*.wav')

    def transcribe(self):
        self.load_audio_filenames()
        self.did_transcription= False
        for filename in self.audio_filenames:
            if filename not in self.transcribed_audio_files.keys():
                print('transcribing: ',filename)
                log('transcribing: '+filename,self.device)
                o  = decode_audiofile(filename, self.pipeline)
                save_pipeline_output_to_file(o, filename,self.output_dir)
                self.transcribed_audio_files[filename] = o
                self.did_transcription = True
                

def check_device(device):
    '''
    Check whether the gpu device is valid and available, otherwise tries
    to find an available device.
    if device equals -1, cpu is used for decoding
    '''
    print('checking gpu device')
    log('checking gpu device',device)
    s = gpus.Selector()
    if device == None:
        print('no device specified, trying to find an available gpu')
        log('no device specified, trying to find an available gpu',device)
        output = s.get_available_divice_on_this_server()
        log('found: '+str(output),device)
    elif device == -1:
        print('running on cpu, this will be slow')
        output=device
    elif type(args.device) == int:
        if type(s.device_available(device)) == int: output = device
        else: output = None 
    if output == None:
        print('could not find a device, doing nothing. Device:',output)
        log('could not find a device, doing nothing. Device: '+output,device)
    return output

def check_input_dir(input_dir, device, check_for_audio_files = True):
    '''
    check wither the input dir exists 
    and optionally whether it contains any audio files.
    '''
    if not input_dir: input_dir = input_directory
    output = input_dir
    if not input_dir.endswith('/'):input_dir += '/'
    if not os.path.isdir(input_dir): 
        print(input_dir,'does not exist, doing nothing')
        log(str(input_dir)+' does not exist, doing nothing',device)
        output = False
    elif len(glob.glob(input_dir + '*.wav')) == 0:
        print(input_dir, 'does not contain .wav files, doing nothing')
        log(str(input_dir)+ ' does not contain .wav files, doing nothing',device)
        output = False
    return output
    
def check_output_dir(output_dir,device):
    ''' checks whether the output dir is an existing directory '''
    if not output_dir: output_dir = output_directory
    output = output_dir
    if not output_dir.endswith('/'):output_dir += '/'
    if not os.path.isdir(output_dir): 
        print(output_dir,'does not exist, doing nothing')
        log(str(output_dir)+' does not exist, doing nothing', device)
        output = False
    return output

def remove_status(device):
    ''' removes the old status of the transcribe function in the status_directory. '''
    print('removing current status:')
    server_name = socket.gethostname()
    fn = glob.glob(status_directory + server_name + '_' + device + '_*')
    print('\n'.join(fn))
    log('removing current status: ' + '\t'.join(fn),device)
    for f in fn:
        os.system('rm ' + f)

def set_status(device,status):
    ''' sets the status of transcribe function in the status_directory. '''
    if device == 'None': return
    remove_status(device)
    print('setting status:', status)
    log('setting status: ' + status,device)
    server_name= socket.gethostname()
    f =open(status_directory + server_name + '_' + device + '_' + status,'w')
    f.close()
    

def transcribe(args):
    set_status(str(args.device),'started')
    ok = True
    device = check_device(args.device)
    if device == None: ok = False
    input_dir =  check_input_dir(args.input_dir,device)
    if not input_dir: ok = False
    output_dir = check_output_dir(args.output_dir,device)
    if not output_dir: ok = False
    if not ok:
        set_status(str(device),'failed')
        return
    if args.keep_alive_minutes == None: args.keep_alive_minutes = 5
    keep_alive_seconds = args.keep_alive_minutes * 60
    set_status(str(device),'active')
    print('loading transcriber')
    log('loading transcriber', device)
    transcriber = Transcriber(args.model_dir, input_dir, output_dir,
        device = device)
    print('start transcribing')
    log('start transcribing', device)
    last_transcription = time.time()
    while True:
        transcriber.transcribe()
        if transcriber.did_transcription:
            last_transcription = time.time()
        time.sleep(1)
        if time.time() - last_transcription > keep_alive_seconds:
            break
    print('closing down transcriber')
    log('closing down transcriber', device)
    set_status(str(device),'closed')
    return transcriber.transcribed_audio_files
            
    
                
    
        
if __name__ == '__main__':
    m = 'transcribe wav audio file with wav2vec2 ' 
    p = argparse.ArgumentParser(description=m)
    p.add_argument('-model_dir', type=str,
        help='directory where the model is located', required = False)
    p.add_argument('-input_dir',type=str,
        help='directory to store transcription',required = False)
    p.add_argument('-output_dir',type=str,
        help='directory to store transcription',required = False)
    p.add_argument('-device',type=int,
        help='device id',required = False)
    p.add_argument('-keep_alive_minutes',type=int,
        help='how long the transcriber proces should linger',required = False)
    args = p.parse_args()
    transcribe(args)

        

def force_unload_pipeline_from_gpu(pipeline, device = -1):
    ''' this clears all gpu memory but could potentially clear other peoples
    processes. also gives an error when you try to reload pipe line'''
    # del variable in function does not work
    if device == -1: return
    # global pipeline
    del pipeline
    d = cuda.select_device(device)
    d.reset()

def unload_pipeline_from_gpu(pipeline, model = None):
    '''does not remove all memory from gpu'''
    # del variable in function does not work
    # global pipeline
    del pipeline
    if model: del model
    torch.cuda.empty_cache()

