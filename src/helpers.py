from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import scipy.stats as stats
import os
import csv
import numpy as np

#For progress bar
NUMBER_OF_FILES = 971

FILE_MISSING_IDX = 8
FILE_NAME_IDX = 3
FIRST_LANGUAGE_IDX = 4
SEX_INDEX = 5
AGE_ONSET_INDEX = 1
ACCEPTED_LANGUAGES = ["english", "spanish", "arabic", "mandarin", "french"]
SEX_LIST = ['male', 'female']

dirname = os.path.dirname(__file__)

def open_file_read(path):
    filename_meta = os.path.join(dirname, path)
    f = open(filename_meta, 'r')
    return csv.reader(f, delimiter=',', quotechar='"')

def write_to_csv(path, data):
    filename_out = os.path.join(dirname, path)
    with open(filename_out, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)

def load_dataset():
    data = open_file_read('../accents_data/dataset.csv')
    dataset = []
    for index, row in enumerate(data):
        if row and index > 0:
            r = list(map(float, row[:-1]))
            r.append(row[-1])
            dataset.append(r)
    return np.array(dataset)

def load_all_wav_into_csv():
    metadata = open_file_read('../accents_data/speakers_all.csv')
    count = 0
    for file in metadata:
        if file[FILE_MISSING_IDX] == "FALSE" and file[FIRST_LANGUAGE_IDX] in ACCEPTED_LANGUAGES:
            count+=1
            filename_wav = os.path.join(dirname, '../accents_data/recordings_wav/' + file[FILE_NAME_IDX] + '.wav')

            (rate,sig) = wav.read(filename_wav)
            mfcc_feat = mfcc(sig,rate, nfft=1200, nfilt=13)
            d_mfcc_feat = delta(mfcc_feat, 2)
            fbank_feat = logfbank(sig,rate, nfft=1200, nfilt=13)

            write_to_csv('../accents_data/recordings_csv/' + file[3] + '.csv', fbank_feat)

            progress(count, NUMBER_OF_FILES, "Generating MFCCs")

def create_dataset():

    metadata = open_file_read('../accents_data/speakers_all.csv')
    count = 0
    dataset = []
    labels = []
    nbColumns = 0
    for file in metadata:
        #Check if file is not missing
        if file[FILE_MISSING_IDX] == "FALSE" and file[FIRST_LANGUAGE_IDX] in ACCEPTED_LANGUAGES :
            #Open corresponding file
            data = open_file_read('../accents_data/recordings_csv/' + file[FILE_NAME_IDX] + '.csv')
            data_in_numbers = []
            new_features = []

            #Convert strings to floats
            for row in data:
                if row:
                    data_in_numbers.append(list(map(float, row)))

            data_in_numbers = np.array(data_in_numbers)
            nbColumns = len(data_in_numbers[0,:])

            #Calculate new features out of MFCCs
            for i in range(nbColumns):
                new_features.append(np.average(data_in_numbers[:,i]))
                new_features.append(np.mean(data_in_numbers[:,i]))
                new_features.append(np.std(data_in_numbers[:,i]))
                new_features.append(stats.skew(data_in_numbers[:,i]))
            #Add correct answer at the end
            new_features.append(ACCEPTED_LANGUAGES.index(file[FIRST_LANGUAGE_IDX]))
            new_features.append(SEX_LIST.index(file[SEX_INDEX]))
           # new_features.append(file[AGE_ONSET_INDEX])


            dataset.append(new_features)
            progress(count, NUMBER_OF_FILES, "Building dataset")
            count+=1
    #Create label row
    for i in range(nbColumns):
        labels.append("average"+str(i+1))
        labels.append("mean"+str(i+1))
        labels.append("std"+str(i+1))
        labels.append("skew"+str(i+1))
    dataset.insert(0,labels)
    write_to_csv('../accents_data/dataset.csv', dataset)


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    print('[%s] %s%s ...%s\r' % (bar, percents, '%', status))