# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:09:51 2021

@author: xiatong

1. resample and normalize
2. extract opensmile features
3. save to csv

"""
import os

import librosa
import numpy as np
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import delay as de

SR = 16000  # sample rate
SR_VGG = 16000  # VGG pretrained model sample rate
FRAME_LEN = int(960)  # 6.25 ms
HOP = int(20)  #  1.25ms

##########################################
path = "../data/0426_EN_used_task1/"


def extract_opensmile(sample, delay = False, ID = 0):
    y, sr = librosa.load(path + sample, sr=SR, mono=True, offset=0.0, duration=None)
    yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
    yt = yt / np.max(np.abs(yt))  # normolized the sound
    # print(yt.shape)
    savepath = "OpenSmile_features/Full/"
    os.makedirs(savepath, exist_ok=True)
    name = sample.replace("/", "___")
    # sf.write(savepath + name + "_normalized.wav", yt, SR, subtype="PCM_24")
    if delay:
        
        feature =  de.extract_delay_timescales(yt, SR, ID, HOP, FRAME_LEN)

        np.save(savepath + name + "_delay_full.npy", feature)



    else:
        '''cmd = (
            'SMILExtract -C /home/ojg30/opensmile/config/emobase/emobase.conf -I "'
            + savepath
            + name
            + '_normalized.wav"'
            + " -O "
            + '"'
            + savepath
            + name
            + '.exemo_n.txt"'
        )
        os.system(cmd)'''
        cmd = (
            'SMILExtract -C /home/ojg30/opensmile/config/is09-13/IS09_emotion.conf -I "'
            + savepath
            + name
            + '_normalized.wav"'
            + " -O "
            + '"'
            + savepath
            + name
            + '.ex384)_fine_n.txt"'
        )
        os.system(cmd)
        cmd = (
            'SMILExtract -C /home/ojg30/opensmile/config/compare16_mod/ComParE_2016.conf -I "'
            + savepath
            + name
            + '_normalized.wav"'
            + " -O "
            + '"'
            + savepath
            + name
            + '.ex6373_pitch_n.txt"'
        )
        os.system(cmd)
        


##########################################
def extract_opensmile2(sample_):
    """opensmile ouput existing"""
    savepath = "OpenSmile_features/"
    sample = savepath + sample_.replace("/", "___") + ".ex384_n.txt"
    with open(sample, "r") as f:
        temp = f.readlines()
        features = temp[391].split(",")[1:-1]
        features = [x for x in features]
    f.close()
    # print(type(features))
    if os.path.exists(savepath + "Full/" + sample_.replace("/", "___") + "_delay_full.npy"):
        # print("we are calm")
        delay_features = np.load(
            savepath + "Full/"+ sample_.replace("/", "___") + "_delay_full.npy", allow_pickle=True
        )
        delay_features = delay_features.flatten().tolist()


        delay_features = [str(x) for x in delay_features]
        features += delay_features
    else:
        print("we are not calm")
        raise KeyError
    

    return ";".join(features)


# output = open('features_6373_delay.csv','w')



output = open("features_384_full.csv", "w")

cate = {"1": "cough", "0": "None"}

output.write("Index,cough_feature,breath_feature,voice_feature,label,uid,categs,fold" + "\n")

with open("../data/data_0426_en_task1.csv") as f:
    file_triplets = []
    for i, line in tqdm(enumerate(f)):
        if i > 0:
            
            temp = line.strip().split(";")
            uid = temp[0]
            print(i, uid)
            folder = temp[7]
            if uid == "MJQ296DCcN" and folder == "2020-11-26-17_00_54_657915":
                continue
            voice = temp[12]
            cough = temp[13]
            breath = temp[14]
            split = temp[15]
            label = temp[16]
            if "202" in uid:
                UID = "form-app-users"
            else:
                UID = uid
            fname_b = "/".join([UID, folder, breath])
            fname_c = "/".join([UID, folder, cough])
            fname_v = "/".join([UID, folder, voice])
            # print(fname_v)
            '''file_triplets += [fname_c, fname_b, fname_v]

            
            def process_sample(fname):
                extract_opensmile(fname, delay=True, ID=1)

            if __name__ == "__main__":
                with ProcessPoolExecutor(max_workers=8) as executor:
                    list(tqdm(executor.map(process_sample, file_triplets)))'''
            
            
            cough = extract_opensmile2(fname_c)
            breath = extract_opensmile2(fname_b )
            voice = extract_opensmile2(fname_v)
            
            
            output.write(",".join([str(i), cough, breath, voice, label, uid, cate[label], split]))
            output.write("\n")
