from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import scipy.stats as stats
import os
import csv
import numpy as np

NUMBER_OF_FILES = 2138

def load_all_wav_into_csv():
    dirname = os.path.dirname(__file__)

    filename_meta = os.path.join(dirname, '../accents_data/speakers_all.csv')
    f = open(filename_meta,'r')
    metadata = csv.reader(f, delimiter=',', quotechar='"')
    count = 0
    for file in metadata:
        if file[8] == "FALSE":
            count+=1
            filename_wav = os.path.join(dirname, '../accents_data/recordings_wav/' + file[3] + '.wav')

            (rate,sig) = wav.read(filename_wav)
            mfcc_feat = mfcc(sig,rate, nfft=1200)
            d_mfcc_feat = delta(mfcc_feat, 2)
            fbank_feat = logfbank(sig,rate, nfft=1200)

            filename_out = os.path.join(dirname, '../accents_data/recordings_csv/' + file[3] + '.csv')
            with open(filename_out, "w") as f:
                writer = csv.writer(f)
                writer.writerows(fbank_feat)
            progress(count, NUMBER_OF_FILES, "Generating MFCCs")

def create_dataset():
    dirname = os.path.dirname(__file__)

    filename_meta = os.path.join(dirname, '../accents_data/speakers_all.csv')
    f = open(filename_meta, 'r')
    metadata = csv.reader(f, delimiter=',', quotechar='"')
    count = 0
    dataset = []
    labels = []
    nbColumns = 0
    for file in metadata:
        if file[8] == "FALSE":
            filename_csv = os.path.join(dirname, '../accents_data/recordings_csv/' + file[3] + '.csv')
            f_csv = open(filename_csv, 'r')
            data = csv.reader(f_csv, delimiter=',', quotechar='"')
            data_in_numbers = []
            new_features = []
            for row in data:
                if row:
                    data_in_numbers.append(list(map(float, row)))
            data_in_numbers = np.array(data_in_numbers)
            nbColumns = len(data_in_numbers[0,:])
            for i in range(nbColumns):
                new_features.append(np.average(data_in_numbers[:,i]))
                new_features.append(np.mean(data_in_numbers[:,i]))
                new_features.append(np.std(data_in_numbers[:,i]))
                new_features.append(np.var(data_in_numbers[:,i]))
                new_features.append(stats.skew(data_in_numbers[:,i]))


            dataset.append(new_features)
            progress(count, NUMBER_OF_FILES, "Building dataset")
            count+=1
    filename_out = os.path.join(dirname, '../accents_data/dataset.csv')
    for i in range(nbColumns):
        labels.append("average"+i+1)
        labels.append("mean"+i+1)
        labels.append("std"+i+1)
        labels.append("var"+i+1)
        labels.append("skew"+i+1)

    dataset.insert(0,labels)
    with open(filename_out, "w") as f:
        writer = csv.writer(f)
        writer.writerows(dataset)

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    print('[%s] %s%s ...%s\r' % (bar, percents, '%', status))