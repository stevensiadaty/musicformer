<a href="https://colab.research.google.com/github/stevensiadaty/musicformer/blob/main/notebooks/ColabGenerateMusicWithAI.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Generate music, given neural net
## Piano, single track
### Based on Google 2018 Music Transformer NN

Codes recycled from:

1) Alex https://github.com/asigalov61/SuperPiano/blob/master/Super_Piano_3.ipynb

2) Damon https://github.com/gwinndr/MusicTransformer-Pytorch

3) Jason https://github.com/jason9693/midi-neural-processor

4) Mir https://github.com/mirsiadaty

Thank you :)


# Link Your Google Drive & Setup (First Step)



## Link With Drive


```python
import time
!pip install pretty_midi
from google.colab import drive
drive.mount('/content/gdrive')
count = 1

print(time.asctime( time.localtime( time.time() ) ))
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting pretty_midi
      Downloading pretty_midi-0.2.9.tar.gz (5.6 MB)
    [K     |████████████████████████████████| 5.6 MB 13.4 MB/s 
    [?25hRequirement already satisfied: numpy>=1.7.0 in /usr/local/lib/python3.7/dist-packages (from pretty_midi) (1.21.6)
    Collecting mido>=1.1.16
      Downloading mido-1.2.10-py2.py3-none-any.whl (51 kB)
    [K     |████████████████████████████████| 51 kB 8.8 MB/s 
    [?25hRequirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from pretty_midi) (1.15.0)
    Building wheels for collected packages: pretty-midi
      Building wheel for pretty-midi (setup.py) ... [?25l[?25hdone
      Created wheel for pretty-midi: filename=pretty_midi-0.2.9-py3-none-any.whl size=5591955 sha256=ef31e9ff95e7b195aab60e67145fd2f2a7997d2d5b748413ca4a58700e12f24c
      Stored in directory: /root/.cache/pip/wheels/ad/74/7c/a06473ca8dcb63efb98c1e67667ce39d52100f837835ea18fa
    Successfully built pretty-midi
    Installing collected packages: mido, pretty-midi
    Successfully installed mido-1.2.10 pretty-midi-0.2.9
    Mounted at /content/gdrive
    Mon Sep  5 21:36:22 2022


## Prep for Generation


```python
# colab
YourHomeDir = '/content/'
YourProjectSubDir = 'gen1_1'


!mkdir $YourHomeDir/$YourProjectSubDir
%cd $YourHomeDir/$YourProjectSubDir
!ls -ltA 

#======================================

#!git clone https://github.com/asigalov61/midi-neural-processor
!git clone https://github.com/jason9693/midi-neural-processor

#!git clone https://github.com/asigalov61/MusicTransformer-Pytorch
!git clone https://github.com/gwinndr/MusicTransformer-Pytorch

#======================================
    
%cd $YourHomeDir/$YourProjectSubDir

!mkdir MusicTransformer-Pytorch/rpr
!mkdir MusicTransformer-Pytorch/rpr/results


#!wget 'https://raw.githubusercontent.com/stevensiadaty/musicformer/main/SP3_20220830_0807.ipynb'
#select_model = YourHomeDir + '/' + YourProjectSubDir + '/' + "MusicTransformer-Pytorch/rpr/results/best_loss_weights.pickle" 
#github says: Yowza, that’s a big file. Try again with a file smaller than 25MB. 

%cd $YourHomeDir/$YourProjectSubDir/MusicTransformer-Pytorch/rpr/results/

!wget 'https://raw.githubusercontent.com/stevensiadaty/musicformer/main/best_loss_weights.pickle'    

#======================================

%cd $YourHomeDir/$YourProjectSubDir/MusicTransformer-Pytorch/dataset/

!wget 'https://raw.githubusercontent.com/stevensiadaty/musicformer/main/e_piano.zip'\

!unzip e_piano.zip


#%cd /content/
%cd $YourHomeDir/$YourProjectSubDir

# this is a renaming!
!mv midi-neural-processor midi_processor

# do for code preprocess_midi.py : import third_party.midi_processor.processor as midi_processor
!cp -r midi_processor/* $YourHomeDir/$YourProjectSubDir/MusicTransformer-Pytorch/third_party/midi_processor/


#%cd /content/MusicTransformer-Pytorch/
%cd $YourHomeDir/$YourProjectSubDir/MusicTransformer-Pytorch/

#======================================

#%cd /content/
%cd $YourHomeDir/$YourProjectSubDir

# this is a renaming!
!mv midi-neural-processor midi_processor

# do for code preprocess_midi.py : import third_party.midi_processor.processor as midi_processor
!cp -r midi_processor/* $YourHomeDir/$YourProjectSubDir/MusicTransformer-Pytorch/third_party/midi_processor/


#%cd /content/MusicTransformer-Pytorch/
%cd $YourHomeDir/$YourProjectSubDir/MusicTransformer-Pytorch/

#======================================
%cd $YourHomeDir/$YourProjectSubDir/midi_processor

from google.colab import files


import processor
from processor import encode_midi, decode_midi

print(time.asctime( time.localtime( time.time() ) ))
```

    /content/gen1_1
    total 0
    Cloning into 'midi-neural-processor'...
    remote: Enumerating objects: 27, done.[K
    remote: Counting objects: 100% (1/1), done.[K
    remote: Total 27 (delta 0), reused 0 (delta 0), pack-reused 26[K
    Unpacking objects: 100% (27/27), done.
    Cloning into 'MusicTransformer-Pytorch'...
    remote: Enumerating objects: 346, done.[K
    remote: Counting objects: 100% (14/14), done.[K
    remote: Compressing objects: 100% (8/8), done.[K
    remote: Total 346 (delta 5), reused 13 (delta 5), pack-reused 332[K
    Receiving objects: 100% (346/346), 109.42 KiB | 241.00 KiB/s, done.
    Resolving deltas: 100% (190/190), done.
    /content/gen1_1
    /content/gen1_1/MusicTransformer-Pytorch/rpr/results
    --2022-09-05 21:36:37--  https://raw.githubusercontent.com/stevensiadaty/musicformer/main/best_loss_weights.pickle
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.110.133, 185.199.108.133, 185.199.111.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.110.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 59441129 (57M) [application/octet-stream]
    Saving to: ‘best_loss_weights.pickle’
    
    best_loss_weights.p 100%[===================>]  56.69M  --.-KB/s    in 0.1s    
    
    2022-09-05 21:36:37 (408 MB/s) - ‘best_loss_weights.pickle’ saved [59441129/59441129]
    
    /content/gen1_1/MusicTransformer-Pytorch/dataset
    --2022-09-05 21:36:37--  https://raw.githubusercontent.com/stevensiadaty/musicformer/main/e_piano.zip
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 30590137 (29M) [application/zip]
    Saving to: ‘e_piano.zip’
    
    e_piano.zip         100%[===================>]  29.17M  --.-KB/s    in 0.08s   
    
    2022-09-05 21:36:38 (386 MB/s) - ‘e_piano.zip’ saved [30590137/30590137]
    
    Archive:  e_piano.zip
       creating: e_piano/
       creating: e_piano/train/
      inflating: e_piano/train/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R2_2008_01-03_ORIG_MID--AUDIO_03_R2_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_066_PIANO066_MID--AUDIO-split_07-07-17_Piano-e_3-02_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_21_R1_2004_01_ORIG_MID--AUDIO_21_R1_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_19_R2_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2011_MID--AUDIO_R1-D6_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_19_R1_2006_01-07_ORIG_MID--AUDIO_19_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_10_13_Group_MID--AUDIO_11_R3_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R1_2011_MID--AUDIO_R1-D3_15_Track15_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_14-15_R1_2014_MID--AUDIO_15_R1_2014_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R1_2006_01-08_ORIG_MID--AUDIO_12_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital1-3_MID--AUDIO_01_R1_2018_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital17-19_MID--AUDIO_18_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_079_PIANO079_MID--AUDIO-split_07-09-17_Piano-e_1-04_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital5-7_MID--AUDIO_07_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital9-11_MID--AUDIO_11_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Chamber6_MID--AUDIO_20_R3_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_07_R1_2004_01_ORIG_MID--AUDIO_07_R1_2004_12_Track12_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital12_MID--AUDIO_12_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R1_2009_01-04_ORIG_MID--AUDIO_09_R1_2009_09_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_19_R1_2009_03-04_ORIG_MID--AUDIO_19_R1_2009_19_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R1_2008_01-04_ORIG_MID--AUDIO_07_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_21_R1_2014_wav--6.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_22_R1_2014_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Chamber4_MID--AUDIO_11_R3_2018_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Chamber6_MID--AUDIO_20_R3_2018_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R1_2009_01-04_ORIG_MID--AUDIO_09_R1_2009_09_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R3_2008_01-03_ORIG_MID--AUDIO_02_R3_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R1_2006_01-04_ORIG_MID--AUDIO_08_R1_2006_Disk1_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R3_2008_01-05_ORIG_MID--AUDIO_10_R3_2008_wav--4.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_10_13_Group_MID--AUDIO_18_R3_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R3_2008_01-05_ORIG_MID--AUDIO_07_R3_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R3_2008_01-05_ORIG_MID--AUDIO_10_R3_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R3_2008_01-05_ORIG_MID--AUDIO_07_R3_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_02_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2009_01-05_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2009_01-05_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R2_2008_01-05_ORIG_MID--AUDIO_11_R2_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_18_R1_2006_01-05_ORIG_MID--AUDIO_18_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_14_R1_2006_01-05_ORIG_MID--AUDIO_14_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_08_R1_2004_03_ORIG_MID--AUDIO_08_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_09-10_R1_2014_MID--AUDIO_10_R1_2014_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_24_R1_2006_01-05_ORIG_MID--AUDIO_24_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2009_03-06_ORIG_MID--AUDIO_15_R1_2009_15_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R2_2008_01-05_ORIG_MID--AUDIO_06_R2_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital20_MID--AUDIO_20_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R2_2008_01-05_ORIG_MID--AUDIO_06_R2_2008_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2011_MID--AUDIO_R1-D6_10_Track10_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_07_R1_2014_wav--6.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_15_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_10_13_Group_MID--AUDIO_12_R3_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_071_PIANO071_MID--AUDIO-split_07-08-17_Piano-e_1-04_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital1-3_MID--AUDIO_02_R1_2018_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R3_2008_01-05_ORIG_MID--AUDIO_10_R3_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R2_2008_01-04_ORIG_MID--AUDIO_04_R2_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital9-11_MID--AUDIO_10_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_07_R1_2014_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_8_13_Group__MID--AUDIO_02_R2_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_08_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_02_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_8_13_Group__MID--AUDIO_02_R2_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R1_2006_01-04_ORIG_MID--AUDIO_04_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_05_R1_2006_01-05_ORIG_MID--AUDIO_05_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital1-3_MID--AUDIO_01_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_16_R1_2014_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_14_R1_2009_01-05_ORIG_MID--AUDIO_14_R1_2009_14_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R1_2009_01-04_ORIG_MID--AUDIO_01_R1_2009_01_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital5-7_MID--AUDIO_07_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_11-13_R1_2014_MID--AUDIO_13_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R1_2009_01-04_ORIG_MID--AUDIO_09_R1_2009_09_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_14_R1_2009_01-05_ORIG_MID--AUDIO_14_R1_2009_14_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_09-10_R1_2014_MID--AUDIO_10_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_09-10_R1_2014_MID--AUDIO_10_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R1_2009_01-04_ORIG_MID--AUDIO_01_R1_2009_01_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_19-21_R3_2014_MID--AUDIO_21_R3_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_11-13_R1_2014_MID--AUDIO_13_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R1_2009_01-02_ORIG_MID--AUDIO_02_R1_2009_02_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_04_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital9-11_MID--AUDIO_09_R1_2018_wav--6.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_11-13_R1_2014_MID--AUDIO_11_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_14_R1_2009_01-05_ORIG_MID--AUDIO_14_R1_2009_14_R1_2009_05_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_083_PIANO083_MID--AUDIO-split_07-09-17_Piano-e_2_-06_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_21_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_21_R1_2014_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_21_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_070_PIANO070_MID--AUDIO-split_07-08-17_Piano-e_1-02_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_072_PIANO072_MID--AUDIO-split_07-08-17_Piano-e_1-06_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_074_PIANO074_MID--AUDIO-split_07-08-17_Piano-e_2-04_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_081_PIANO081_MID--AUDIO-split_07-09-17_Piano-e_2_-02_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_083_PIANO083_MID--AUDIO-split_07-09-17_Piano-e_2_-06_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R3_2008_01-05_ORIG_MID--AUDIO_08_R3_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R3_2008_01-05_ORIG_MID--AUDIO_10_R3_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R3_2008_01-04_ORIG_MID--AUDIO_11_R3_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_03_R2_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_07_R2_2015_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_10_13_Group_MID--AUDIO_02_R3_2013_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_10_13_Group_MID--AUDIO_11_R3_2013_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_10_13_Group_MID--AUDIO_15_R3_2013_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_10_13_Group_MID--AUDIO_17_R3_2013_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_10_13_Group_MID--AUDIO_18_R3_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R3_2011_MID--AUDIO_R3-D1_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R3_2011_MID--AUDIO_R3-D2_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R3_2011_MID--AUDIO_R3-D3_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_23_R3_2011_MID--AUDIO_R3-D8_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R3_2011_MID--AUDIO_R3-D5_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R3_2008_01-05_ORIG_MID--AUDIO_07_R3_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_06_R2_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_13_R2_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_15_R2_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R1_2006_01-08_ORIG_MID--AUDIO_12_R1_2006_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R3_2011_MID--AUDIO_R3-D1_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R2_2008_01-04_ORIG_MID--AUDIO_08_R2_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R1_2008_01-05_ORIG_MID--AUDIO_09_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R1_2008_01-04_ORIG_MID--AUDIO_10_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_18_R1_2008_01-04_ORIG_MID--AUDIO_18_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R2_2011_MID--AUDIO_R2-D5_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_09-10_R1_2014_MID--AUDIO_09_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-08-12_R3_2014_MID--AUDIO_04_R3_2014_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_8_13_Group__MID--AUDIO_04_R2_2013_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_15_R2_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_22_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R2_2011_MID--AUDIO_R2-D4_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_12_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R2_2008_01-05_ORIG_MID--AUDIO_11_R2_2008_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_16_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2011_MID--AUDIO_R1-D1_18_Track18_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_05_R1_2011_MID--AUDIO_R1-D2_11_Track11_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R3_2011_MID--AUDIO_R3-D2_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R1_2011_MID--AUDIO_R1-D3_10_Track10_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R1_2006_01-04_ORIG_MID--AUDIO_09_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R1_2006_01-04_ORIG_MID--AUDIO_10_R1_2006_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_05_R1_2009_01-02_ORIG_MID--AUDIO_05_R1_2009_05_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_09_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital1-3_MID--AUDIO_01_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_02_R1_2015_wav--6.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_12_R1_2015_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_046_PIANO046_MID--AUDIO-split_07-06-17_Piano-e_2-02_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_054_PIANO054_MID--AUDIO-split_07-07-17_Piano-e_1-02_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_066_PIANO066_MID--AUDIO-split_07-07-17_Piano-e_3-02_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-21-22_mid--AUDIO-from_mp3_22_R1_2015_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_23_R2_2011_MID--AUDIO_R2-D6_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_043_PIANO043_MID--AUDIO-split_07-06-17_Piano-e_1-03_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_081_PIANO081_MID--AUDIO-split_07-09-17_Piano-e_2_-02_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R1_2006_01-04_ORIG_MID--AUDIO_02_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_05_R1_2004_02-03_ORIG_MID--AUDIO_05_R1_2004_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_10_Track10_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R1_2009_01-03_ORIG_MID--AUDIO_17_R1_2009_17_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_8_13_Group__MID--AUDIO_11_R2_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-08-12_R3_2014_MID--AUDIO_04_R3_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_21_R1_2014_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R2_2008_01-03_ORIG_MID--AUDIO_03_R2_2008_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_02_R1_2013_wav--5.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_09_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_067_PIANO067_MID--AUDIO-split_07-07-17_Piano-e_3-03_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R1_2008_01-05_ORIG_MID--AUDIO_02_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R2_2008_01-04_ORIG_MID--AUDIO_04_R2_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R3_2008_01-05_ORIG_MID--AUDIO_08_R3_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_25_R3_2011_MID--AUDIO_R3-D9_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital1-3_MID--AUDIO_02_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_08_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_17_R2_2013_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_10_13_Group_MID--AUDIO_12_R3_2013_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_17_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R1_2006_01-04_ORIG_MID--AUDIO_04_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_18_R1_2006_01-05_ORIG_MID--AUDIO_18_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2009_03-06_ORIG_MID--AUDIO_15_R1_2009_15_R1_2009_06_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R2_2011_MID--AUDIO_R2-D3_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2008_01-04_ORIG_MID--AUDIO_03_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital4_MID--AUDIO_04_R1_2018_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R1_2006_01-04_ORIG_MID--AUDIO_06_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_24_R1_2006_01-05_ORIG_MID--AUDIO_24_R1_2006_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2009_01-02_ORIG_MID--AUDIO_03_R1_2009_03_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_19_R1_2009_01-02_ORIG_MID--AUDIO_19_R1_2009_19_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_21_R1_2009_04_ORIG_MID--AUDIO_21_R1_2009_21_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_02_R1_2014_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2009_03-06_ORIG_MID--AUDIO_15_R1_2009_15_R1_2009_05_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R1_2008_01-04_ORIG_MID--AUDIO_17_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_06_R1_2004_02-03_ORIG_MID--AUDIO_06_R1_2004_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R1_2009_04-06_ORIG_MID--AUDIO_04_R1_2009_04_R1_2009_05_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_18_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R2_2008_01-05_ORIG_MID--AUDIO_10_R2_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R3_2011_MID--AUDIO_R3-D3_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_047_PIANO047_MID--AUDIO-split_07-06-17_Piano-e_2-04_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R3_2008_01-07_ORIG_MID--AUDIO_04_R3_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_17_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R3_2008_01-07_ORIG_MID--AUDIO_04_R3_2008_wav--5.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_07_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_15_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_09-10_R1_2014_MID--AUDIO_09_R1_2014_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R2_2011_MID--AUDIO_R2-D2_07_Track07_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_052_PIANO052_MID--AUDIO-split_07-06-17_Piano-e_3-03_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_060_PIANO060_MID--AUDIO-split_07-07-17_Piano-e_2-04_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_072_PIANO072_MID--AUDIO-split_07-08-17_Piano-e_1-06_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R3_2011_MID--AUDIO_R3-D7_07_Track07_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R3_2011_MID--AUDIO_R3-D7_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R3_2011_MID--AUDIO_R3-D7_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital17-19_MID--AUDIO_19_R1_2018_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_05_R1_2014_wav--8.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_045_PIANO045_MID--AUDIO-split_07-06-17_Piano-e_2-01_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_050_PIANO050_MID--AUDIO-split_07-06-17_Piano-e_3-01_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_047_PIANO047_MID--AUDIO-split_07-06-17_Piano-e_2-04_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R3_2011_MID--AUDIO_R3-D1_07_Track07_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R2_2008_01-05_ORIG_MID--AUDIO_06_R2_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R2_2008_01-05_ORIG_MID--AUDIO_11_R2_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_075_PIANO075_MID--AUDIO-split_07-08-17_Piano-e_2-06_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_081_PIANO081_MID--AUDIO-split_07-09-17_Piano-e_2_-02_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_15_R2_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R2_2011_MID--AUDIO_R2-D1_07_Track07_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_8_13_Group__MID--AUDIO_08_R2_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_07_R2_2015_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_08_R2_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_21_R2_2015_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_8_13_Group__MID--AUDIO_11_R2_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_074_PIANO074_MID--AUDIO-split_07-08-17_Piano-e_2-04_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_080_PIANO080_MID--AUDIO-split_07-09-17_Piano-e_1-06_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_8_13_Group__MID--AUDIO_02_R2_2013_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_8_13_Group__MID--AUDIO_12_R2_2013_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_8_13_Group__MID--AUDIO_14_R2_2013_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_18_R2_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R2_2011_MID--AUDIO_R2-D2_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R2_2011_MID--AUDIO_R2-D3_10_Track10_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R2_2011_MID--AUDIO_R2-D5_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R2_2011_MID--AUDIO_R2-D5_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital9-11_MID--AUDIO_11_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R2_2008_01-03_ORIG_MID--AUDIO_03_R2_2008_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_8_13_Group__MID--AUDIO_04_R2_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_079_PIANO079_MID--AUDIO-split_07-09-17_Piano-e_1-04_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R2_2011_MID--AUDIO_R2-D2_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_25_R2_2011_MID--AUDIO_R2-D6_07_Track07_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R2_2008_01-05_ORIG_MID--AUDIO_02_R2_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R2_2008_01-05_ORIG_MID--AUDIO_09_R2_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R2_2008_01-04_ORIG_MID--AUDIO_12_R2_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_22_R2_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_070_PIANO070_MID--AUDIO-split_07-08-17_Piano-e_1-02_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_8_13_Group__MID--AUDIO_03_R2_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R2_2008_01-04_ORIG_MID--AUDIO_17_R2_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R2_2008_01-05_ORIG_MID--AUDIO_07_R2_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R2_2008_01-04_ORIG_MID--AUDIO_08_R2_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_11_R2_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_13_R2_2015_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_15_R2_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_073_PIANO073_MID--AUDIO-split_07-08-17_Piano-e_2-02_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_082_PIANO082_MID--AUDIO-split_07-09-17_Piano-e_2_-04_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_17_R2_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R2_2011_MID--AUDIO_R2-D3_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R2_2011_MID--AUDIO_R2-D4_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R2_2011_MID--AUDIO_R2-D4_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_19_R2_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital12_MID--AUDIO_12_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_17_R1_2004_01-03_ORIG_MID--AUDIO_17_R1_2004_02_Track02_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_05_R2_2006_01_ORIG_MID--AUDIO_05_R2_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R2_2006_01_ORIG_MID--AUDIO_09_R2_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R2_2006_01_ORIG_MID--AUDIO_16_R2_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_05_R2_2006_01_ORIG_MID--AUDIO_05_R2_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R2_2006_01_ORIG_MID--AUDIO_09_R2_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R2_2006_01_ORIG_MID--AUDIO_16_R2_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_05_R2_2006_01_ORIG_MID--AUDIO_05_R2_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R2_2006_01_ORIG_MID--AUDIO_09_R2_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R2_2006_01_ORIG_MID--AUDIO_16_R2_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R2_2006_01_ORIG_MID--AUDIO_01_R2_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R2_2006_01_ORIG_MID--AUDIO_07_R2_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R2_2006_01_ORIG_MID--AUDIO_08_R2_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R2_2006_01_ORIG_MID--AUDIO_01_R2_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R2_2006_01_ORIG_MID--AUDIO_07_R2_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R2_2006_01_ORIG_MID--AUDIO_08_R2_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R2_2006_01_ORIG_MID--AUDIO_01_R2_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R2_2006_01_ORIG_MID--AUDIO_07_R2_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R2_2006_01_ORIG_MID--AUDIO_08_R2_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R2_2006_01_ORIG_MID--AUDIO_07_R2_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-07-08-10-12-15-17_R2_2014_MID--AUDIO_08_R2_2014_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-07-08-10-12-15-17_R2_2014_MID--AUDIO_17_R2_2014_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_19-20-21_R2_2014_MID--AUDIO_19_R2_2014_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Schubert1-3_MID--AUDIO_07_R2_2018_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R2_2006_01_ORIG_MID--AUDIO_22_R2_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R2_2006_01_ORIG_MID--AUDIO_22_R2_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R2_2006_01_ORIG_MID--AUDIO_22_R2_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R2_2006_01_ORIG_MID--AUDIO_22_R2_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_23_R2_2006_01_ORIG_MID--AUDIO_23_R2_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_23_R2_2006_01_ORIG_MID--AUDIO_23_R2_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_23_R2_2006_01_ORIG_MID--AUDIO_23_R2_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_23_R2_2006_01_ORIG_MID--AUDIO_23_R2_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R2_2009_01_ORIG_MID--AUDIO_09_R2_2009_09_R2_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R2_2009_01_ORIG_MID--AUDIO_09_R2_2009_09_R2_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R2_2009_01_ORIG_MID--AUDIO_09_R2_2009_09_R2_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R2_2009_01_ORIG_MID--AUDIO_09_R2_2009_09_R2_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Schubert1-3_MID--AUDIO_02_R2_2018_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-07-08-10-12-15-17_R2_2014_MID--AUDIO_15_R2_2014_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_21_R2_2009_01_ORIG_MID--AUDIO_21_R2_2009_21_R2_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_21_R2_2009_01_ORIG_MID--AUDIO_21_R2_2009_21_R2_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_21_R2_2009_01_ORIG_MID--AUDIO_21_R2_2009_21_R2_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_21_R2_2009_01_ORIG_MID--AUDIO_21_R2_2009_21_R2_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R2_2009_01_ORIG_MID--AUDIO_10_R2_2009_10_R2_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-07-08-10-12-15-17_R2_2014_MID--AUDIO_04_R2_2014_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-07-08-10-12-15-17_R2_2014_MID--AUDIO_12_R2_2014_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R2_2009_01_ORIG_MID--AUDIO_10_R2_2009_10_R2_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R2_2009_01_ORIG_MID--AUDIO_10_R2_2009_10_R2_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R2_2009_01_ORIG_MID--AUDIO_10_R2_2009_10_R2_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_14_R2_2004_01_ORIG_MID--AUDIO_14_R2_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_14_R2_2004_01_ORIG_MID--AUDIO_14_R2_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_14_R2_2004_01_ORIG_MID--AUDIO_14_R2_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_14_R2_2004_01_ORIG_MID--AUDIO_14_R2_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_16_R2_2004_01_ORIG_MID--AUDIO_16_R2_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_16_R2_2004_01_ORIG_MID--AUDIO_16_R2_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_16_R2_2004_01_ORIG_MID--AUDIO_16_R2_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_16_R2_2004_01_ORIG_MID--AUDIO_16_R2_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Schubert4-6_MID--AUDIO_09_R2_2018_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Schubert4-6_MID--AUDIO_10_R2_2018_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R2_2009_01_ORIG_MID--AUDIO_17_R2_2009_17_R2_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R2_2009_01_ORIG_MID--AUDIO_17_R2_2009_17_R2_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R2_2009_01_ORIG_MID--AUDIO_17_R2_2009_17_R2_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R2_2009_01_ORIG_MID--AUDIO_17_R2_2009_17_R2_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R2_2009_01_ORIG_MID--AUDIO_07_R2_2009_07_R2_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R2_2009_01_ORIG_MID--AUDIO_12_R2_2009_12_R2_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-07-08-10-12-15-17_R2_2014_MID--AUDIO_07_R2_2014_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_19-20-21_R2_2014_MID--AUDIO_21_R2_2014_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R2_2009_01_ORIG_MID--AUDIO_07_R2_2009_07_R2_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R2_2009_01_ORIG_MID--AUDIO_12_R2_2009_12_R2_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R2_2009_01_ORIG_MID--AUDIO_07_R2_2009_07_R2_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R2_2009_01_ORIG_MID--AUDIO_12_R2_2009_12_R2_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R2_2009_01_ORIG_MID--AUDIO_07_R2_2009_07_R2_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R2_2009_01_ORIG_MID--AUDIO_12_R2_2009_12_R2_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_06_R2_2004_01_ORIG_MID--AUDIO_06_R2_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_06_R2_2004_01_ORIG_MID--AUDIO_06_R2_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_06_R2_2004_01_ORIG_MID--AUDIO_06_R2_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_19_R2_2004_01_ORIG_MID--AUDIO_19_R2_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_19_R2_2004_01_ORIG_MID--AUDIO_19_R2_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_19_R2_2004_01_ORIG_MID--AUDIO_19_R2_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Schubert1-3_MID--AUDIO_05_R2_2018_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Schubert4-6_MID--AUDIO_08_R2_2018_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Schubert7-9_MID--AUDIO_11_R2_2018_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Schubert7-9_MID--AUDIO_16_R2_2018_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-07-08-10-12-15-17_R2_2014_MID--AUDIO_10_R2_2014_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_15_R2_2004_01_ORIG_MID--AUDIO_15_R2_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_15_R2_2004_01_ORIG_MID--AUDIO_15_R2_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_15_R2_2004_01_ORIG_MID--AUDIO_15_R2_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_15_R2_2004_01_ORIG_MID--AUDIO_15_R2_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Schubert10-12_MID--AUDIO_20_R2_2018_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Schubert7-9_MID--AUDIO_15_R2_2018_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Schubert10-12_MID--AUDIO_17_R2_2018_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Schubert10-12_MID--AUDIO_18_R2_2018_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_17_R2_2004_01_ORIG_MID--AUDIO_17_R2_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_17_R2_2004_01_ORIG_MID--AUDIO_17_R2_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_17_R2_2004_01_ORIG_MID--AUDIO_17_R2_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_17_R2_2004_01_ORIG_MID--AUDIO_17_R2_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R2_2009_01_ORIG_MID--AUDIO_08_R2_2009_08_R2_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_19_R2_2009_01_ORIG_MID--AUDIO_19_R2_2009_19_R2_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R2_2009_01_ORIG_MID--AUDIO_08_R2_2009_08_R2_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_19_R2_2009_01_ORIG_MID--AUDIO_19_R2_2009_19_R2_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R2_2009_01_ORIG_MID--AUDIO_08_R2_2009_08_R2_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_19_R2_2009_01_ORIG_MID--AUDIO_19_R2_2009_19_R2_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R2_2009_01_ORIG_MID--AUDIO_08_R2_2009_08_R2_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_19_R2_2009_01_ORIG_MID--AUDIO_19_R2_2009_19_R2_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_04_R2_2004_01_ORIG_MID--AUDIO_04_R2_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_04_R2_2004_01_ORIG_MID--AUDIO_04_R2_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_20_R2_2004_01_ORIG_MID--AUDIO_20_R1_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_20_R2_2004_01_ORIG_MID--AUDIO_20_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_20_R2_2004_01_ORIG_MID--AUDIO_20_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R2_2011_MID--AUDIO_R2-D1_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2006_01-06_ORIG_MID--AUDIO_11_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_18_R1_2004_01-02_ORIG_MID--AUDIO_18_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_18_R1_2004_01-02_ORIG_MID--AUDIO_18_R1_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_25_R2_2011_MID--AUDIO_R2-D6_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R2_2011_MID--AUDIO_R2-D4_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-21-22_mid--AUDIO-from_mp3_21_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R2_2011_MID--AUDIO_R2-D5_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_06_R1_2004_01_ORIG_MID--AUDIO_06_R1_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R3_2008_01-05_ORIG_MID--AUDIO_10_R3_2008_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R3_2008_01-04_ORIG_MID--AUDIO_11_R3_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_13_R1_2009_01-03_ORIG_MID--AUDIO_13_R1_2009_13_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_21_R1_2009_01-03_ORIG_MID--AUDIO_21_R1_2009_21_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R3_2011_MID--AUDIO_R3-D2_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R1_2011_MID--AUDIO_R1-D2_17_Track17_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_071_PIANO071_MID--AUDIO-split_07-08-17_Piano-e_1-04_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_082_PIANO082_MID--AUDIO-split_07-09-17_Piano-e_2_-04_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_05_R1_2014_wav--7.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_11_R2_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_22_R2_2015_wav--4.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_14_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R1_2006_01-04_ORIG_MID--AUDIO_04_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_07_Track07_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_17_R1_2004_01-03_ORIG_MID--AUDIO_17_R1_2004_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_17_R1_2004_03-06_ORIG_MID--AUDIO_20_R2_2004_12_Track12_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_12_R2_2015_wav--5.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_8_13_Group__MID--AUDIO_12_R2_2013_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_10_13_Group_MID--AUDIO_08_R3_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R2_2011_MID--AUDIO_R2-D2_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R1_2011_MID--AUDIO_R1-D7_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_18_R1_2011_MID--AUDIO_R1-D7_10_Track10_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_046_PIANO046_MID--AUDIO-split_07-06-17_Piano-e_2-02_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R3_2008_01-03_ORIG_MID--AUDIO_02_R3_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2009_03-08_ORIG_MID--AUDIO_03_R1_2009_03_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2008_01-04_ORIG_MID--AUDIO_15_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_050_PIANO050_MID--AUDIO-split_07-06-17_Piano-e_3-01_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R2_2008_01-05_ORIG_MID--AUDIO_07_R2_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R2_2011_MID--AUDIO_R2-D3_11_Track11_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R1_2008_01-05_ORIG_MID--AUDIO_08_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R3_2008_01-04_ORIG_MID--AUDIO_12_R3_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_03_R1_2014_wav--5.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_17_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R1_2011_MID--AUDIO_R1-D4_14_Track14_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_051_PIANO051_MID--AUDIO-split_07-06-17_Piano-e_3-02_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_13_R1_2006_01-06_ORIG_MID--AUDIO_13_R1_2006_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_10_13_Group_MID--AUDIO_18_R3_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R1_2009_03-05_ORIG_MID--AUDIO_10_R1_2009_10_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R3_2008_01-05_ORIG_MID--AUDIO_08_R3_2008_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_10_13_Group_MID--AUDIO_07_R3_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_049_PIANO049_MID--AUDIO-split_07-06-17_Piano-e_2-06_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_17_R1_2004_04_ORIG_MID--AUDIO_17_R1_2004_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_17_R1_2004_04_ORIG_MID--AUDIO_17_R1_2004_11_Track11_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_03_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_10_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_12_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_20_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_19_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_042_PIANO042_MID--AUDIO-split_07-06-17_Piano-e_1-02_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_043_PIANO043_MID--AUDIO-split_07-06-17_Piano-e_1-03_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_059_PIANO059_MID--AUDIO-split_07-07-17_Piano-e_2-03_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_053_PIANO053_MID--AUDIO-split_07-06-17_Piano-e_3-04_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_15_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_06_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_070_PIANO070_MID--AUDIO-split_07-08-17_Piano-e_1-02_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_08_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_14_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R1_2011_MID--AUDIO_R1-D1_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_18_R1_2011_MID--AUDIO_R1-D7_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_21_R1_2011_MID--AUDIO_R1-D8_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2008_01-04_ORIG_MID--AUDIO_03_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_044_PIANO044_MID--AUDIO-split_07-06-17_Piano-e_1-04_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_049_PIANO049_MID--AUDIO-split_07-06-17_Piano-e_2-06_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_067_PIANO067_MID--AUDIO-split_07-07-17_Piano-e_3-03_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_09_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_17_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2008_01-04_ORIG_MID--AUDIO_15_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R1_2011_MID--AUDIO_R1-D2_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_19_R1_2011_MID--AUDIO_R1-D7_14_Track14_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_07_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_10_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_056_PIANO056_MID--AUDIO-split_07-07-17_Piano-e_1-05_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_057_PIANO057_MID--AUDIO-split_07-07-17_Piano-e_1-07_wav--4.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_18_R1_2013_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_20_R1_2011_MID--AUDIO_R1-D8_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R1_2011_MID--AUDIO_R1-D8_11_Track11_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_08_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_18_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_8_13_Group__MID--AUDIO_08_R2_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2008_01-04_ORIG_MID--AUDIO_11_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2011_MID--AUDIO_R1-D4_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_13_R1_2011_MID--AUDIO_R1-D5_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R1_2011_MID--AUDIO_R1-D6_13_Track13_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R1_2011_MID--AUDIO_R1-D7_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_051_PIANO051_MID--AUDIO-split_07-06-17_Piano-e_3-02_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_04_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_14_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R1_2011_MID--AUDIO_R1-D3_14_Track14_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R2_2008_01-05_ORIG_MID--AUDIO_09_R2_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_19_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_03_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2011_MID--AUDIO_R1-D1_17_Track17_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R1_2011_MID--AUDIO_R1-D2_16_Track16_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_18_R1_2014_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_18_R1_2014_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_24_R1_2011_MID--AUDIO_R1-D9_10_Track10_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_8_13_Group__MID--AUDIO_03_R2_2013_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_11_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_16_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_05_R1_2014_wav--6.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_02_R1_2015_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R1_2009_04_ORIG_MID--AUDIO_17_R1_2009_17_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_07_R1_2014_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_07_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_056_PIANO056_MID--AUDIO-split_07-07-17_Piano-e_1-05_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_057_PIANO057_MID--AUDIO-split_07-07-17_Piano-e_1-07_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R1_2008_01-04_ORIG_MID--AUDIO_10_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_16_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_20_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_06_R2_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_22_R2_2015_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_062_PIANO062_MID--AUDIO-split_07-07-17_Piano-e_2-07_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R1_2009_03-07_ORIG_MID--AUDIO_06_R1_2009_06_R1_2009_07_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_05_R1_2015_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_23_R1_2011_MID--AUDIO_R1-D9_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_10_R1_2004_03-04_ORIG_MID--AUDIO_10_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R1_2009_01-02_ORIG_MID--AUDIO_16_R1_2009_16_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R2_2008_01-04_ORIG_MID--AUDIO_08_R2_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_07_R1_2014_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_15_R1_2015_wav--5.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_09_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_044_PIANO044_MID--AUDIO-split_07-06-17_Piano-e_1-04_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_048_PIANO048_MID--AUDIO-split_07-06-17_Piano-e_2-05_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_053_PIANO053_MID--AUDIO-split_07-06-17_Piano-e_3-04_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R1_2006_01-04_ORIG_MID--AUDIO_07_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_20_R1_2006_01-04_ORIG_MID--AUDIO_20_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_09_R1_2004_05_ORIG_MID--AUDIO_09_R1_2004_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_09_R1_2004_05_ORIG_MID--AUDIO_09_R1_2004_07_Track07_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_09_R1_2004_05_ORIG_MID--AUDIO_09_R1_2004_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital13-15_MID--AUDIO_13_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Chamber5_MID--AUDIO_18_R3_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_06_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_22_R1_2014_wav--5.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_18_R2_2013_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_052_PIANO052_MID--AUDIO-split_07-06-17_Piano-e_3-03_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_18_R1_2004_04_ORIG_MID--AUDIO_18_R1_2004_07_Track07_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2011_MID--AUDIO_R1-D4_10_Track10_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2011_MID--AUDIO_R1-D4_11_Track11_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R1_2006_01-04_ORIG_MID--AUDIO_06_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_18_R1_2009_01-03_ORIG_MID--AUDIO_18_R1_2009_18_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital1-3_MID--AUDIO_01_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_079_PIANO079_MID--AUDIO-split_07-09-17_Piano-e_1-04_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_08_R1_2004_01-02_ORIG_MID--AUDIO_08_R1_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R1_2009_01-02_ORIG_MID--AUDIO_12_R1_2009_12_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_11-13_R1_2014_MID--AUDIO_12_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_058_PIANO058_MID--AUDIO-split_07-07-17_Piano-e_2-02_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_051_PIANO051_MID--AUDIO-split_07-06-17_Piano-e_3-02_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2006_01-05_ORIG_MID--AUDIO_15_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_21_R1_2014_wav--7.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2009_03-08_ORIG_MID--AUDIO_03_R1_2009_03_R1_2009_06_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_14_R1_2009_01-05_ORIG_MID--AUDIO_14_R1_2009_14_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_05_R1_2006_01-05_ORIG_MID--AUDIO_05_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_04_R1_2004_03-05_ORIG_MID--AUDIO_04_R1_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_04_R1_2004_03-05_ORIG_MID--AUDIO_04_R1_2004_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2009_03-08_ORIG_MID--AUDIO_03_R1_2009_03_R1_2009_08_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_17_R1_2004_03-06_ORIG_MID--AUDIO_20_R2_2004_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2009_03-08_ORIG_MID--AUDIO_03_R1_2009_03_R1_2009_07_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2006_01-06_ORIG_MID--AUDIO_11_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_11_R1_2004_01-02_ORIG_MID--AUDIO_11_R1_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R2_2008_01-05_ORIG_MID--AUDIO_06_R2_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital17-19_MID--AUDIO_19_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_05_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_19_R2_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R1_2009_03-07_ORIG_MID--AUDIO_06_R1_2009_06_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R1_2009_03-07_ORIG_MID--AUDIO_06_R1_2009_06_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_08_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital5-7_MID--AUDIO_06_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_10_13_Group_MID--AUDIO_08_R3_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R1_2008_01-05_ORIG_MID--AUDIO_08_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_18_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital13-15_MID--AUDIO_14_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_07_R2_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_14-15_R1_2014_MID--AUDIO_14_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_8_13_Group__MID--AUDIO_12_R2_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Chamber4_MID--AUDIO_11_R3_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R1_2009_01-02_ORIG_MID--AUDIO_10_R1_2009_10_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_07_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R1_2006_01-06_ORIG_MID--AUDIO_17_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_19_R1_2006_01-07_ORIG_MID--AUDIO_19_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_02_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_17_R1_2004_01-02_ORIG_MID--AUDIO_20_R2_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_16_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_13_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_19_R1_2011_MID--AUDIO_R1-D7_12_Track12_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_25_R1_2011_MID--AUDIO_R1-D9_14_Track14_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_067_PIANO067_MID--AUDIO-split_07-07-17_Piano-e_3-03_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R1_2011_MID--AUDIO_R1-D2_14_Track14_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_09_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_12_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2011_MID--AUDIO_R1-D6_07_Track07_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_052_PIANO052_MID--AUDIO-split_07-06-17_Piano-e_3-03_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_054_PIANO054_MID--AUDIO-split_07-07-17_Piano-e_1-02_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_062_PIANO062_MID--AUDIO-split_07-07-17_Piano-e_2-07_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R1_2011_MID--AUDIO_R1-D3_12_Track12_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_055_PIANO055_MID--AUDIO-split_07-07-17_Piano-e_1-04_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R1_2008_01-04_ORIG_MID--AUDIO_06_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R1_2008_01-05_ORIG_MID--AUDIO_09_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_10_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-21-22_mid--AUDIO-from_mp3_22_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_15_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_18_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R1_2011_MID--AUDIO_R1-D6_12_Track12_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_17_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_16_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_20_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_21_R1_2011_MID--AUDIO_R1-D8_07_Track07_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_24_R1_2011_MID--AUDIO_R1-D9_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R1_2011_MID--AUDIO_R1-D2_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R1_2011_MID--AUDIO_R1-D4_15_Track15_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_13_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R1_2008_01-05_ORIG_MID--AUDIO_02_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R1_2008_01-04_ORIG_MID--AUDIO_10_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_09_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_12_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_059_PIANO059_MID--AUDIO-split_07-07-17_Piano-e_2-03_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital17-19_MID--AUDIO_17_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_19_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_05_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R1_2011_MID--AUDIO_R1-D1_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R1_2011_MID--AUDIO_R1-D1_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_23_R1_2011_MID--AUDIO_R1-D9_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R1_2011_MID--AUDIO_R1-D3_07_Track07_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_049_PIANO049_MID--AUDIO-split_07-06-17_Piano-e_2-06_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_08_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_17_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2011_MID--AUDIO_R1-D1_15_Track15_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_051_PIANO051_MID--AUDIO-split_07-06-17_Piano-e_3-02_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_04_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_13_R1_2011_MID--AUDIO_R1-D5_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_061_PIANO061_MID--AUDIO-split_07-07-17_Piano-e_2-05_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_058_PIANO058_MID--AUDIO-split_07-07-17_Piano-e_2-02_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_14_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_045_PIANO045_MID--AUDIO-split_07-06-17_Piano-e_2-01_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_046_PIANO046_MID--AUDIO-split_07-06-17_Piano-e_2-02_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_047_PIANO047_MID--AUDIO-split_07-06-17_Piano-e_2-04_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R1_2008_01-04_ORIG_MID--AUDIO_16_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_066_PIANO066_MID--AUDIO-split_07-07-17_Piano-e_3-02_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2008_01-04_ORIG_MID--AUDIO_11_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_05_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_18_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_05_R1_2011_MID--AUDIO_R1-D2_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_05_R1_2011_MID--AUDIO_R1-D2_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_042_PIANO042_MID--AUDIO-split_07-06-17_Piano-e_1-02_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_048_PIANO048_MID--AUDIO-split_07-06-17_Piano-e_2-05_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R1_2008_01-04_ORIG_MID--AUDIO_07_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R1_2008_01-04_ORIG_MID--AUDIO_12_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2008_01-04_ORIG_MID--AUDIO_15_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_18_R1_2008_01-04_ORIG_MID--AUDIO_18_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-21-22_mid--AUDIO-from_mp3_21_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_07_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_057_PIANO057_MID--AUDIO-split_07-07-17_Piano-e_1-07_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_060_PIANO060_MID--AUDIO-split_07-07-17_Piano-e_2-04_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R1_2008_01-04_ORIG_MID--AUDIO_04_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_20_R1_2011_MID--AUDIO_R1-D8_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_15_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R1_2011_MID--AUDIO_R1-D8_12_Track12_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_03_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_16_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_01_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R1_2011_MID--AUDIO_R1-D1_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R1_2011_MID--AUDIO_R1-D7_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_19_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R1_2011_MID--AUDIO_R1-D4_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_06_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital8_MID--AUDIO_08_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_19-20_R1_2014_MID--AUDIO_19_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R2_2008_01-04_ORIG_MID--AUDIO_12_R2_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_18_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_14_R1_2004_01-03_ORIG_MID--AUDIO_14_R1_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_10_R1_2004_01-02_ORIG_MID--AUDIO_10_R1_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R1_2006_01-04_ORIG_MID--AUDIO_07_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_07_R1_2014_wav--7.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_13_01_2004_01-05_ORIG_MID--AUDIO_13_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_21_R1_2006_01-04_ORIG_MID--AUDIO_21_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_25_R3_2011_MID--AUDIO_R3-D9_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_25_R3_2011_MID--AUDIO_R3-D9_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_20_R1_2006_01-04_ORIG_MID--AUDIO_20_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_05_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-21-22_mid--AUDIO-from_mp3_21_R1_2015_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_074_PIANO074_MID--AUDIO-split_07-08-17_Piano-e_2-04_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R1_2006_01-04_ORIG_MID--AUDIO_02_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2008_01-04_ORIG_MID--AUDIO_11_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_054_PIANO054_MID--AUDIO-split_07-07-17_Piano-e_1-02_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_13_R1_2008_01-04_ORIG_MID--AUDIO_13_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_20_R1_2009_06-07_ORIG_MID--AUDIO_20_R1_2009_20_R1_2009_06_WAV.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_10_13_Group_MID--AUDIO_17_R3_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_18_R1_2009_04_ORIG_MID--AUDIO_18_R1_2009_18_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R1_2006_01-08_ORIG_MID--AUDIO_12_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_07_R1_2004_01_ORIG_MID--AUDIO_07_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_07_R1_2004_01_ORIG_MID--AUDIO_07_R1_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_07_R1_2004_01_ORIG_MID--AUDIO_07_R1_2004_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_11-13_R1_2014_MID--AUDIO_11_R1_2014_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital5-7_MID--AUDIO_05_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_13_R1_2015_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_21_R2_2015_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_8_13_Group__MID--AUDIO_07_R2_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_04_R1_2004_01-02_ORIG_MID--AUDIO_04_R1_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_15_R1_2004_01-02_ORIG_MID--AUDIO_15_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_19-20_R1_2014_MID--AUDIO_20_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_19_R1_2006_01-07_ORIG_MID--AUDIO_19_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R1_2006_01-04_ORIG_MID--AUDIO_02_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R1_2006_01-04_ORIG_MID--AUDIO_22_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_042_PIANO042_MID--AUDIO-split_07-06-17_Piano-e_1-02_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_19_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_10_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R3_2011_MID--AUDIO_R3-D1_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R1_2009_03-05_ORIG_MID--AUDIO_12_R1_2009_12_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_15_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-21-22_mid--AUDIO-from_mp3_21_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_08_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_22_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_13_01_2004_01-05_ORIG_MID--AUDIO_13_R1_2004_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_13_01_2004_01-05_ORIG_MID--AUDIO_13_R1_2004_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_13_01_2004_01-05_ORIG_MID--AUDIO_13_R1_2004_10_Track10_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_04_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_17_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R1_2008_01-04_ORIG_MID--AUDIO_04_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital8_MID--AUDIO_08_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_13_R1_2008_01-04_ORIG_MID--AUDIO_13_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_14_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_15_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R1_2008_01-04_ORIG_MID--AUDIO_16_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R1_2011_MID--AUDIO_R1-D1_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R1_2011_MID--AUDIO_R1-D4_13_Track13_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_054_PIANO054_MID--AUDIO-split_07-07-17_Piano-e_1-02_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital5-7_MID--AUDIO_07_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_02_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital9-11_MID--AUDIO_10_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_19_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital13-15_MID--AUDIO_14_R1_2018_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R1_2009_03-05_ORIG_MID--AUDIO_10_R1_2009_10_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2006_01-06_ORIG_MID--AUDIO_11_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-08-12_R3_2014_MID--AUDIO_12_R3_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R1_2009_01-02_ORIG_MID--AUDIO_10_R1_2009_10_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_14-15_R1_2014_MID--AUDIO_15_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Chamber1_MID--AUDIO_07_R3_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_19_R1_2011_MID--AUDIO_R1-D7_13_Track13_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R1_2011_MID--AUDIO_R1-D3_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2011_MID--AUDIO_R1-D6_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_08_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_050_PIANO050_MID--AUDIO-split_07-06-17_Piano-e_3-01_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_053_PIANO053_MID--AUDIO-split_07-06-17_Piano-e_3-04_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_055_PIANO055_MID--AUDIO-split_07-07-17_Piano-e_1-04_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_057_PIANO057_MID--AUDIO-split_07-07-17_Piano-e_1-07_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2009_01-05_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital17-19_MID--AUDIO_18_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2011_MID--AUDIO_R1-D4_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_13_R1_2011_MID--AUDIO_R1-D5_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_18_R1_2011_MID--AUDIO_R1-D7_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_03_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2008_01-04_ORIG_MID--AUDIO_15_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_07_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_08_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_05_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_03_R1_2014_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_043_PIANO043_MID--AUDIO-split_07-06-17_Piano-e_1-03_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_048_PIANO048_MID--AUDIO-split_07-06-17_Piano-e_2-05_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_056_PIANO056_MID--AUDIO-split_07-07-17_Piano-e_1-05_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_059_PIANO059_MID--AUDIO-split_07-07-17_Piano-e_2-03_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_062_PIANO062_MID--AUDIO-split_07-07-17_Piano-e_2-07_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2008_01-04_ORIG_MID--AUDIO_03_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_10_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R2_2011_MID--AUDIO_R2-D4_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_20_R1_2011_MID--AUDIO_R1-D8_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_25_R1_2011_MID--AUDIO_R1-D9_15_Track15_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_21_R1_2011_MID--AUDIO_R1-D8_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_045_PIANO045_MID--AUDIO-split_07-06-17_Piano-e_2-01_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_11_R1_2008_01-04_ORIG_MID--AUDIO_11_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_24_R1_2011_MID--AUDIO_R1-D9_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_09_R1_2004_01-02_ORIG_MID--AUDIO_09_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_09_R1_2004_01-02_ORIG_MID--AUDIO_09_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R1_2011_MID--AUDIO_R1-D6_14_Track14_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R1_2011_MID--AUDIO_R1-D7_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-21-22_mid--AUDIO-from_mp3_22_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_11-13_R1_2014_MID--AUDIO_13_R1_2014_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_17_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_18_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital13-15_MID--AUDIO_14_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital17-19_MID--AUDIO_19_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_06-08_R1_2014_MID--AUDIO_06_R1_2014_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital9-11_MID--AUDIO_10_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_02_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_14_R1_2011_MID--AUDIO_R1-D6_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R1_2009_01-02_ORIG_MID--AUDIO_06_R1_2009_06_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_19-20_R1_2014_MID--AUDIO_19_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital13-15_MID--AUDIO_15_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_01_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_02_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R1_2011_MID--AUDIO_R1-D8_13_Track13_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_17_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_060_PIANO060_MID--AUDIO-split_07-07-17_Piano-e_2-04_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_061_PIANO061_MID--AUDIO-split_07-07-17_Piano-e_2-05_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_06_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_13_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R1_2011_MID--AUDIO_R1-D2_15_Track15_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_047_PIANO047_MID--AUDIO-split_07-06-17_Piano-e_2-04_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_067_PIANO067_MID--AUDIO-split_07-07-17_Piano-e_3-03_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R1_2006_01-04_ORIG_MID--AUDIO_10_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_21_R1_2006_01-04_ORIG_MID--AUDIO_21_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_13_R1_2006_01-06_ORIG_MID--AUDIO_13_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_15_R1_2004_01-02_ORIG_MID--AUDIO_15_R1_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_05_R1_2006_01-05_ORIG_MID--AUDIO_05_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_17_R1_2004_01-03_ORIG_MID--AUDIO_17_R1_2004_02_Track02_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_17_R1_2004_01-03_ORIG_MID--AUDIO_17_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_01_R1_2004_01-02_ORIG_MID--AUDIO_01_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_01_R1_2004_01-02_ORIG_MID--AUDIO_01_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_21_R1_2009_01-03_ORIG_MID--AUDIO_21_R1_2009_21_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_10_Track10_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital5-7_MID--AUDIO_06_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_21_R1_2004_02_ORIG_MID--AUDIO_21_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_21_R1_2004_02_ORIG_MID--AUDIO_21_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital5-7_MID--AUDIO_05_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_20_R1_2009_01-05_ORIG_MID--AUDIO_20_R1_2009_20_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R3_2011_MID--AUDIO_R3-D4_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R3_2011_MID--AUDIO_R3-D6_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R3_2011_MID--AUDIO_R3-D7_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_25_R3_2011_MID--AUDIO_R3-D9_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_19_R2_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R3_2008_01-03_ORIG_MID--AUDIO_02_R3_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R3_2008_01-03_ORIG_MID--AUDIO_03_R3_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R3_2008_01-07_ORIG_MID--AUDIO_04_R3_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R3_2008_01-04_ORIG_MID--AUDIO_12_R3_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_12_R2_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_21_R2_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_22_R2_2015_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_10_13_Group_MID--AUDIO_07_R3_2013_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_10_13_Group_MID--AUDIO_08_R3_2013_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_10_13_Group_MID--AUDIO_12_R3_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_071_PIANO071_MID--AUDIO-split_07-08-17_Piano-e_1-04_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_073_PIANO073_MID--AUDIO-split_07-08-17_Piano-e_2-02_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_075_PIANO075_MID--AUDIO-split_07-08-17_Piano-e_2-06_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_078_PIANO078_MID--AUDIO-split_07-09-17_Piano-e_1-02_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_079_PIANO079_MID--AUDIO-split_07-09-17_Piano-e_1-04_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_080_PIANO080_MID--AUDIO-split_07-09-17_Piano-e_1-06_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_082_PIANO082_MID--AUDIO-split_07-09-17_Piano-e_2_-04_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_10_13_Group_MID--AUDIO_14_R3_2013_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_08_R2_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_11_R2_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R1_2008_01-05_ORIG_MID--AUDIO_02_R1_2008_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R1_2008_01-04_ORIG_MID--AUDIO_06_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R3_2011_MID--AUDIO_R3-D1_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R1_2006_01-04_ORIG_MID--AUDIO_04_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_18_R1_2006_01-05_ORIG_MID--AUDIO_18_R1_2006_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_03_R1_2004_01-02_ORIG_MID--AUDIO_03_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_21_R1_2004_03_ORIG_MID--AUDIO_21_R1_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital5-7_MID--AUDIO_07_R1_2018_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital8_MID--AUDIO_08_R1_2018_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_18_R1_2009_01-03_ORIG_MID--AUDIO_18_R1_2009_18_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_18_R1_2006_01-05_ORIG_MID--AUDIO_18_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_10_R1_2011_MID--AUDIO_R1-D4_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2009_01-02_ORIG_MID--AUDIO_15_R1_2009_15_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R1_2006_01-05_ORIG_MID--AUDIO_15_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_07_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R1_2008_01-04_ORIG_MID--AUDIO_16_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2009_03-08_ORIG_MID--AUDIO_03_R1_2009_03_R1_2009_05_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_078_PIANO078_MID--AUDIO-split_07-09-17_Piano-e_1-02_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_09-10_R1_2014_MID--AUDIO_10_R1_2014_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital16_MID--AUDIO_16_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_13_R1_2006_01-06_ORIG_MID--AUDIO_13_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R1_2008_01-05_ORIG_MID--AUDIO_08_R1_2008_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital13-15_MID--AUDIO_15_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital1-3_MID--AUDIO_02_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_18_R1_2006_01-05_ORIG_MID--AUDIO_18_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R2_2008_01-04_ORIG_MID--AUDIO_17_R2_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_13_R1_2009_01-03_ORIG_MID--AUDIO_13_R1_2009_13_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_24_R1_2006_01-05_ORIG_MID--AUDIO_24_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R1_2008_01-05_ORIG_MID--AUDIO_09_R1_2008_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_18_R1_2014_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R2_2008_01-05_ORIG_MID--AUDIO_02_R2_2008_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_16_R1_2004_01-08_ORIG_MID--AUDIO_16_R1_2004_13_Track13_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_11_R1_2004_03-04_ORIG_MID--AUDIO_11_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R2_2008_01-05_ORIG_MID--AUDIO_02_R2_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R2_2008_01-05_ORIG_MID--AUDIO_02_R2_2008_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_09_R1_2006_01-04_ORIG_MID--AUDIO_09_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_13_R1_2006_01-06_ORIG_MID--AUDIO_13_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_16_R1_2004_01-08_ORIG_MID--AUDIO_16_R1_2004_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_08_R2_2015_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R1_2011_MID--AUDIO_R1-D2_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R2_2008_01-04_ORIG_MID--AUDIO_12_R2_2008_wav--3.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_12_R1_2013_wav--5.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_13_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_8_13_Group__MID--AUDIO_08_R2_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R1_2011_MID--AUDIO_R1-D3_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_01_R1_2004_03_ORIG_MID--AUDIO_01_R1_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_10_R1_2004_05_ORIG_MID--AUDIO_10_R1_2004_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital13-15_MID--AUDIO_15_R1_2018_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_03_R1_2014_wav--6.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital9-11_MID--AUDIO_11_R1_2018_wav--5.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital17-19_MID--AUDIO_17_R1_2018_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_08_R1_2004_01-02_ORIG_MID--AUDIO_08_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_23_R1_2006_01-05_ORIG_MID--AUDIO_23_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_17_R1_2004_01-02_ORIG_MID--AUDIO_20_R2_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_17_R1_2004_01-02_ORIG_MID--AUDIO_20_R2_2004_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_05_R1_2009_03-05_ORIG_MID--AUDIO_05_R1_2009_05_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_11-13_R1_2014_MID--AUDIO_12_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_09-10_R1_2014_MID--AUDIO_10_R1_2014_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-08-12_R3_2014_MID--AUDIO_08_R3_2014_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_13_R1_2009_04_ORIG_MID--AUDIO_13_R1_2009_13_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R1_2006_01-04_ORIG_MID--AUDIO_08_R1_2006_Disk1_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_14_R1_2006_01-05_ORIG_MID--AUDIO_14_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_16_R1_2006_01-04_ORIG_MID--AUDIO_16_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital13-15_MID--AUDIO_13_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital20_MID--AUDIO_20_R1_2018_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R1_2009_01-02_ORIG_MID--AUDIO_02_R1_2009_02_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_14-15_R1_2014_MID--AUDIO_15_R1_2014_wav--6.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_15_R2_2015_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_21-22_R1_2014_MID--AUDIO_21_R1_2014_wav--8.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_15_R2_2008_01-04_ORIG_MID--AUDIO_15_R2_2008_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_08_R1_2004_04-06_ORIG_MID--AUDIO_08_R1_2004_05_Track05_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_060_PIANO060_MID--AUDIO-split_07-07-17_Piano-e_2-04_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--5.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_02_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_22_R1_2006_01-04_ORIG_MID--AUDIO_22_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R3_2011_MID--AUDIO_R3-D6_03_Track03_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R3_2011_MID--AUDIO_R3-D6_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R3_2011_MID--AUDIO_R3-D6_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R3_2011_MID--AUDIO_R3-D6_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R3_2011_MID--AUDIO_R3-D6_07_Track07_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_17_R3_2011_MID--AUDIO_R3-D6_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_11_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_08_R1_2004_04-06_ORIG_MID--AUDIO_08_R1_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R1_2006_01-08_ORIG_MID--AUDIO_12_R1_2006_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_04_R1_2014_wav--4.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_05_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital17-19_MID--AUDIO_19_R1_2018_wav--6.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R3_2011_MID--AUDIO_R3-D4_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R3_2011_MID--AUDIO_R3-D4_07_Track07_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R3_2011_MID--AUDIO_R3-D4_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R3_2011_MID--AUDIO_R3-D4_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_15_R1_2004_04_ORIG_MID--AUDIO_15_R1_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_01_R1_2004_04-05_ORIG_MID--AUDIO_01_R1_2004_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_14_R1_2009_06-08_ORIG_MID--AUDIO_14_R1_2009_14_R1_2009_08_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R1_2009_05-06_ORIG_MID--AUDIO_08_R1_2009_08_R1_2009_06_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_SMF_16_R1_2004_01-08_ORIG_MID--AUDIO_16_R1_2004_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2006_01-05_ORIG_MID--AUDIO_03_R1_2006_05_Track05_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R1_2009_04-05_ORIG_MID--AUDIO_07_R1_2009_07_R1_2009_05_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_12_R1_2006_01-08_ORIG_MID--AUDIO_12_R1_2006_08_Track08_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R1_2009_04-06_ORIG_MID--AUDIO_04_R1_2009_04_R1_2009_06_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R1_2006_01-04_ORIG_MID--AUDIO_07_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_04_R1_2009_01-03_ORIG_MID--AUDIO_04_R1_2009_04_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_24_R1_2006_01-05_ORIG_MID--AUDIO_24_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R1_2011_MID--AUDIO_R1-D1_09_Track09_wav.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_18_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_08_R1_2009_01-04_ORIG_MID--AUDIO_08_R1_2009_08_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_20_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_02_R1_2008_01-05_ORIG_MID--AUDIO_02_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital13-15_MID--AUDIO_13_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_11_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_Recital12_MID--AUDIO_12_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_14_R1_2008_01-05_ORIG_MID--AUDIO_14_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R3_2008_01-05_ORIG_MID--AUDIO_07_R3_2008_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_03_R1_2011_MID--AUDIO_R1-D1_16_Track16_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_14_R1_2009_06-08_ORIG_MID--AUDIO_14_R1_2009_14_R1_2009_06_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-UNPROCESSED_09-10_R1_2014_MID--AUDIO_09_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_12_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/train/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_12_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_05_R1_2009_01-02_ORIG_MID--AUDIO_05_R1_2009_05_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_07_R1_2011_MID--AUDIO_R1-D3_04_Track04_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_06_R1_2006_01-04_ORIG_MID--AUDIO_06_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_11_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_12_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_14_R1_2004_04_ORIG_MID--AUDIO_14_R1_2004_06_Track06_wav.midi.pickle  
      inflating: e_piano/train/MIDI-Unprocessed_XP_04_R1_2004_01-02_ORIG_MID--AUDIO_04_R1_2004_02_Track02_wav.midi.pickle  
       creating: e_piano/val/
      inflating: e_piano/val/MIDI-Unprocessed_17_R1_2006_01-06_ORIG_MID--AUDIO_17_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_07_R1_2009_04-05_ORIG_MID--AUDIO_07_R1_2009_07_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_02_R1_2009_03-06_ORIG_MID--AUDIO_02_R1_2009_02_R1_2009_05_WAV.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_06_WAV.midi.pickle  
      inflating: e_piano/val/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_03_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-UNPROCESSED_14-15_R1_2014_MID--AUDIO_15_R1_2014_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-UNPROCESSED_14-15_R1_2014_MID--AUDIO_15_R1_2014_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_22_R1_2006_01-04_ORIG_MID--AUDIO_22_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_Recital8_MID--AUDIO_08_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_Recital9-11_MID--AUDIO_10_R1_2018_wav--5.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_02_R1_2008_01-05_ORIG_MID--AUDIO_02_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/val/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_06_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_16_R1_2011_MID--AUDIO_R1-D6_15_Track15_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_09_R1_2006_01-04_ORIG_MID--AUDIO_09_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_15_R2_2008_01-04_ORIG_MID--AUDIO_15_R2_2008_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_12_R1_2009_03-05_ORIG_MID--AUDIO_12_R1_2009_12_R1_2009_05_WAV.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_03_R1_2011_MID--AUDIO_R1-D1_19_Track19_wav.midi.pickle  
      inflating: e_piano/val/ORIG-MIDI_02_7_8_13_Group__MID--AUDIO_14_R2_2013_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_10_R1_2011_MID--AUDIO_R1-D4_05_Track05_wav.midi.pickle  
      inflating: e_piano/val/ORIG-MIDI_01_7_8_13_Group__MID--AUDIO_02_R2_2013_wav--5.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_03_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_XP_11_R1_2004_03-04_ORIG_MID--AUDIO_11_R1_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_04_R1_2009_04-06_ORIG_MID--AUDIO_04_R1_2009_04_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_16_R1_2009_03-06_ORIG_MID--AUDIO_16_R1_2009_16_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_04_R3_2008_01-07_ORIG_MID--AUDIO_04_R3_2008_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_Recital9-11_MID--AUDIO_09_R1_2018_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_07_R1_2008_01-04_ORIG_MID--AUDIO_07_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_XP_01_R1_2004_04-05_ORIG_MID--AUDIO_01_R1_2004_05_Track05_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_05_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_055_PIANO055_MID--AUDIO-split_07-07-17_Piano-e_1-04_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--7.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_09_R2_2008_01-05_ORIG_MID--AUDIO_09_R2_2008_wav--5.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_08_R1_2009_01-04_ORIG_MID--AUDIO_08_R1_2009_08_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_12_R2_2015_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_15_R2_2008_01-04_ORIG_MID--AUDIO_15_R2_2008_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_16_R2_2009_01_ORIG_MID--AUDIO_16_R2_2009_16_R2_2009_03_WAV.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_Recital9-11_MID--AUDIO_09_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_14_R1_2008_01-05_ORIG_MID--AUDIO_14_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_083_PIANO083_MID--AUDIO-split_07-09-17_Piano-e_2_-06_wav--5.midi.pickle  
      inflating: e_piano/val/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_05_R1_2014_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_09_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_25_R3_2011_MID--AUDIO_R3-D9_05_Track05_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_058_PIANO058_MID--AUDIO-split_07-07-17_Piano-e_2-02_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_060_PIANO060_MID--AUDIO-split_07-07-17_Piano-e_2-04_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_062_PIANO062_MID--AUDIO-split_07-07-17_Piano-e_2-07_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_061_PIANO061_MID--AUDIO-split_07-07-17_Piano-e_2-05_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_06_R1_2009_03-07_ORIG_MID--AUDIO_06_R1_2009_06_R1_2009_06_WAV.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_12_R3_2011_MID--AUDIO_R3-D4_02_Track02_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_07_R2_2008_01-05_ORIG_MID--AUDIO_07_R2_2008_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_05_R1_2014_wav--5.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_07_R2_2008_01-05_ORIG_MID--AUDIO_07_R2_2008_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-UNPROCESSED_04-08-12_R3_2014_MID--AUDIO_12_R3_2014_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_R1_D2-21-22_mid--AUDIO-from_mp3_22_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_058_PIANO058_MID--AUDIO-split_07-07-17_Piano-e_2-02_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_02_R2_2011_MID--AUDIO_R2-D1_03_Track03_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_04_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_072_PIANO072_MID--AUDIO-split_07-08-17_Piano-e_1-06_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_06_R1_2009_03-07_ORIG_MID--AUDIO_06_R1_2009_06_R1_2009_05_WAV.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_06_R3_2011_MID--AUDIO_R3-D3_05_Track05_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_16_R1_2008_01-04_ORIG_MID--AUDIO_16_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_19_R1_2006_01-07_ORIG_MID--AUDIO_19_R1_2006_06_Track06_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_19_R1_2006_01-07_ORIG_MID--AUDIO_19_R1_2006_07_Track07_wav.midi.pickle  
      inflating: e_piano/val/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_18_R2_2013_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_01_R1_2011_MID--AUDIO_R1-D1_06_Track06_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_07_R1_2011_MID--AUDIO_R1-D3_05_Track05_wav.midi.pickle  
      inflating: e_piano/val/ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_10_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_07_R1_2009_01-03_ORIG_MID--AUDIO_07_R1_2009_07_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/val/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_03_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_02_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_Recital4_MID--AUDIO_04_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_17_R1_2008_01-04_ORIG_MID--AUDIO_17_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_044_PIANO044_MID--AUDIO-split_07-06-17_Piano-e_1-04_wav--1.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_XP_09_R1_2004_01-02_ORIG_MID--AUDIO_09_R1_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_14_R1_2011_MID--AUDIO_R1-D6_02_Track02_wav.midi.pickle  
      inflating: e_piano/val/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_14_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_065_PIANO065_MID--AUDIO-split_07-07-17_Piano-e_3-01_wav--1.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_05_R1_2008_01-04_ORIG_MID--AUDIO_05_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_10_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/val/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_06_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_043_PIANO043_MID--AUDIO-split_07-06-17_Piano-e_1-03_wav--1.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_050_PIANO050_MID--AUDIO-split_07-06-17_Piano-e_3-01_wav--1.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_XP_14_R1_2004_01-03_ORIG_MID--AUDIO_14_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_14_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_19_R1_2006_01-07_ORIG_MID--AUDIO_19_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_09_R1_2009_05-06_ORIG_MID--AUDIO_09_R1_2009_09_R1_2009_05_WAV.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_08_R1_2006_01-04_ORIG_MID--AUDIO_08_R1_2006_Disk1_02_Track02_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_23_R1_2006_01-05_ORIG_MID--AUDIO_23_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_03_R3_2011_MID--AUDIO_R3-D1_03_Track03_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_XP_19_R1_2004_01-02_ORIG_MID--AUDIO_19_R1_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_SMF_13_01_2004_01-05_ORIG_MID--AUDIO_13_R1_2004_05_Track05_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_Chamber6_MID--AUDIO_20_R3_2018_wav--1.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_XP_10_R1_2004_01-02_ORIG_MID--AUDIO_10_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_03_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_13_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_16_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_058_PIANO058_MID--AUDIO-split_07-07-17_Piano-e_2-02_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_Recital4_MID--AUDIO_04_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_07_R1_2008_01-04_ORIG_MID--AUDIO_07_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_051_PIANO051_MID--AUDIO-split_07-06-17_Piano-e_3-02_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_Recital1-3_MID--AUDIO_02_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_16_R1_2009_01-02_ORIG_MID--AUDIO_16_R1_2009_16_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_XP_04_R1_2004_06_ORIG_MID--AUDIO_04_R1_2004_08_Track08_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_13_R1_2006_01-06_ORIG_MID--AUDIO_13_R1_2006_05_Track05_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_21_R1_2011_MID--AUDIO_R1-D8_10_Track10_wav.midi.pickle  
      inflating: e_piano/val/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--5.midi.pickle  
      inflating: e_piano/val/ORIG-MIDI_03_7_10_13_Group_MID--AUDIO_15_R3_2013_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_XP_19_R1_2004_01-02_ORIG_MID--AUDIO_19_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/val/MIDI-UNPROCESSED_19-20_R1_2014_MID--AUDIO_20_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_23_R3_2011_MID--AUDIO_R3-D8_03_Track03_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_Recital4_MID--AUDIO_04_R1_2018_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_04_R1_2014_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_073_PIANO073_MID--AUDIO-split_07-08-17_Piano-e_2-02_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_09_R1_2006_01-04_ORIG_MID--AUDIO_09_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_12_R2_2011_MID--AUDIO_R2-D4_02_Track02_wav.midi.pickle  
      inflating: e_piano/val/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_01_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_SMF_16_R1_2004_01-08_ORIG_MID--AUDIO_16_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_09_R1_2008_01-05_ORIG_MID--AUDIO_09_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_049_PIANO049_MID--AUDIO-split_07-06-17_Piano-e_2-06_wav--4.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_23_R1_2011_MID--AUDIO_R1-D9_05_Track05_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_12_R1_2008_01-04_ORIG_MID--AUDIO_12_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_Recital5-7_MID--AUDIO_05_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_SMF_16_R1_2004_01-08_ORIG_MID--AUDIO_16_R1_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_12_R1_2006_01-08_ORIG_MID--AUDIO_12_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_SMF_16_R1_2004_01-08_ORIG_MID--AUDIO_16_R1_2004_06_Track06_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_12_R1_2006_01-08_ORIG_MID--AUDIO_12_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_SMF_13_01_2004_01-05_ORIG_MID--AUDIO_13_R1_2004_12_Track12_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_Recital17-19_MID--AUDIO_19_R1_2018_wav--5.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_14_R1_2006_01-05_ORIG_MID--AUDIO_14_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_SMF_16_R1_2004_01-08_ORIG_MID--AUDIO_16_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_18_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_16_R1_2006_01-04_ORIG_MID--AUDIO_16_R1_2006_01_Track01_wav.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_03_Track03_wav--1.midi.pickle  
      inflating: e_piano/val/MIDI-Unprocessed_Recital17-19_MID--AUDIO_17_R1_2018_wav--3.midi.pickle  
       creating: e_piano/test/
      inflating: e_piano/test/MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_07_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_02_R1_2009_03-06_ORIG_MID--AUDIO_02_R1_2009_02_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_24_R1_2006_01-05_ORIG_MID--AUDIO_24_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_11_R1_2009_06-09_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_08_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_Recital13-15_MID--AUDIO_15_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_09-10_R1_2014_MID--AUDIO_09_R1_2014_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_11_R1_2009_01-05_ORIG_MID--AUDIO_11_R1_2009_11_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_SMF_17_R1_2004_03-06_ORIG_MID--AUDIO_20_R2_2004_12_Track12_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_04_R3_2011_MID--AUDIO_R3-D2_05_Track05_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_05_R1_2008_01-04_ORIG_MID--AUDIO_05_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_12_R2_2015_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_Recital16_MID--AUDIO_16_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_03_R1_2006_01-05_ORIG_MID--AUDIO_03_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_19-21_R3_2014_MID--AUDIO_21_R3_2014_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_10_R2_2008_01-05_ORIG_MID--AUDIO_10_R2_2008_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_Recital17-19_MID--AUDIO_18_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_Recital9-11_MID--AUDIO_09_R1_2018_wav--5.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_11-13_R1_2014_MID--AUDIO_11_R1_2014_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_10_R2_2008_01-05_ORIG_MID--AUDIO_10_R2_2008_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_083_PIANO083_MID--AUDIO-split_07-09-17_Piano-e_2_-06_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_XP_04_R1_2004_03-05_ORIG_MID--AUDIO_04_R1_2004_06_Track06_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_059_PIANO059_MID--AUDIO-split_07-07-17_Piano-e_2-03_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_Chamber5_MID--AUDIO_18_R3_2018_wav--2.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_08_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_14_R1_2008_01-05_ORIG_MID--AUDIO_14_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_06_R1_2008_01-04_ORIG_MID--AUDIO_06_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_17_R1_2008_01-04_ORIG_MID--AUDIO_17_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_02_7_6_13_Group__MID--AUDIO_08_R1_2013_wav--5.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_02_R1_2011_MID--AUDIO_R1-D1_10_Track10_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_04_R1_2008_01-04_ORIG_MID--AUDIO_04_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R2_D2-12-13-15_mid--AUDIO-from_mp3_13_R2_2015_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_07_R1_2006_01-04_ORIG_MID--AUDIO_07_R1_2006_04_Track04_wav.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_11-13_R1_2014_MID--AUDIO_13_R1_2014_wav--6.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_13_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_04_R3_2008_01-07_ORIG_MID--AUDIO_04_R3_2008_wav--6.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_Recital17-19_MID--AUDIO_19_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_04_R3_2008_01-07_ORIG_MID--AUDIO_04_R3_2008_wav--7.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_04_R3_2008_01-07_ORIG_MID--AUDIO_04_R3_2008_wav--3.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_01_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_11_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_14_R1_2011_MID--AUDIO_R1-D6_04_Track04_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_075_PIANO075_MID--AUDIO-split_07-08-17_Piano-e_2-06_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_065_PIANO065_MID--AUDIO-split_07-07-17_Piano-e_3-01_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_08_R1_2009_01-04_ORIG_MID--AUDIO_08_R1_2009_08_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_08_R1_2009_01-04_ORIG_MID--AUDIO_08_R1_2009_08_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_080_PIANO080_MID--AUDIO-split_07-09-17_Piano-e_1-06_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_072_PIANO072_MID--AUDIO-split_07-08-17_Piano-e_1-06_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_02_R2_2011_MID--AUDIO_R2-D1_02_Track02_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_04_R2_2008_01-04_ORIG_MID--AUDIO_04_R2_2008_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_10_R2_2008_01-05_ORIG_MID--AUDIO_10_R2_2008_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_03_R2_2015_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R2_D1-2-3-6-7-8-11_mid--AUDIO-from_mp3_06_R2_2015_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R2_D2-19-21-22_mid--AUDIO-from_mp3_19_R2_2015_wav--3.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_01_7_8_13_Group__MID--AUDIO_07_R2_2013_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_23_R2_2011_MID--AUDIO_R2-D6_03_Track03_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_078_PIANO078_MID--AUDIO-split_07-09-17_Piano-e_1-02_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_071_PIANO071_MID--AUDIO-split_07-08-17_Piano-e_1-04_wav--2.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_02_7_10_13_Group_MID--AUDIO_12_R3_2013_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_083_PIANO083_MID--AUDIO-split_07-09-17_Piano-e_2_-06_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_Recital9-11_MID--AUDIO_09_R1_2018_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_16_R2_2009_01_ORIG_MID--AUDIO_16_R2_2009_16_R2_2009_01_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_16_R2_2009_01_ORIG_MID--AUDIO_16_R2_2009_16_R2_2009_02_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_16_R2_2009_01_ORIG_MID--AUDIO_16_R2_2009_16_R2_2009_04_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_19-20-21_R2_2014_MID--AUDIO_20_R2_2014_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_03_R1_2009_03-08_ORIG_MID--AUDIO_03_R1_2009_03_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_10_R1_2006_01-04_ORIG_MID--AUDIO_10_R1_2006_03_Track03_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_Recital9-11_MID--AUDIO_09_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_XP_18_R1_2004_01-02_ORIG_MID--AUDIO_18_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_XP_18_R1_2004_01-02_ORIG_MID--AUDIO_18_R1_2004_05_Track05_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_05_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--6.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_11_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D2-21-22_mid--AUDIO-from_mp3_22_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_07_R1_2011_MID--AUDIO_R1-D3_03_Track03_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_24_R1_2011_MID--AUDIO_R1-D9_11_Track11_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_12_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_22_R2_2011_MID--AUDIO_R2-D5_10_Track10_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_02_R1_2009_03-06_ORIG_MID--AUDIO_02_R1_2009_02_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_03_R1_2014_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_057_PIANO057_MID--AUDIO-split_07-07-17_Piano-e_1-07_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_17_R2_2008_01-04_ORIG_MID--AUDIO_17_R2_2008_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_09_R2_2008_01-05_ORIG_MID--AUDIO_09_R2_2008_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_15_R1_2006_01-05_ORIG_MID--AUDIO_15_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_19_R2_2013_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_22_R2_2011_MID--AUDIO_R2-D5_11_Track11_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_17_R1_2006_01-06_ORIG_MID--AUDIO_17_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_12_R1_2008_01-04_ORIG_MID--AUDIO_12_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_01_R1_2009_01-04_ORIG_MID--AUDIO_01_R1_2009_01_R1_2009_04_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_07_R1_2009_01-03_ORIG_MID--AUDIO_07_R1_2009_07_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_19_R1_2011_MID--AUDIO_R1-D7_15_Track15_wav.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_04_R1_2013_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_Recital5-7_MID--AUDIO_06_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_02_7_10_13_Group_MID--AUDIO_14_R3_2013_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_15_R1_2015_wav--4.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_01_7_10_13_Group_MID--AUDIO_07_R3_2013_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_11_R1_2011_MID--AUDIO_R1-D4_07_Track07_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_02_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_13_R1_2008_01-04_ORIG_MID--AUDIO_13_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_04_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_XP_03_R1_2004_01-02_ORIG_MID--AUDIO_03_R1_2004_01_Track01_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_03_R1_2008_01-04_ORIG_MID--AUDIO_03_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_11_R3_2008_01-04_ORIG_MID--AUDIO_11_R3_2008_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_18_R1_2011_MID--AUDIO_R1-D7_07_Track07_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_053_PIANO053_MID--AUDIO-split_07-06-17_Piano-e_3-04_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_14_R1_2008_01-05_ORIG_MID--AUDIO_14_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_11_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_11_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_07_R1_2011_MID--AUDIO_R1-D3_02_Track02_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_07_R1_2015_wav--1.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_03_R1_2013_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_056_PIANO056_MID--AUDIO-split_07-07-17_Piano-e_1-05_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_15_R1_2009_01-02_ORIG_MID--AUDIO_15_R1_2009_15_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_04_R1_2011_MID--AUDIO_R1-D2_03_Track03_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_12_R1_2008_01-04_ORIG_MID--AUDIO_12_R1_2008_wav--1.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_01_7_7_13_Group__MID--AUDIO_14_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_19_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_XP_19_R1_2004_01-02_ORIG_MID--AUDIO_19_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_04_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_05_R1_2011_MID--AUDIO_R1-D2_10_Track10_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_Recital13-15_MID--AUDIO_14_R1_2018_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_14-15_R1_2014_MID--AUDIO_15_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_09_R1_2011_MID--AUDIO_R1-D3_13_Track13_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_06_R1_2008_01-04_ORIG_MID--AUDIO_06_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_044_PIANO044_MID--AUDIO-split_07-06-17_Piano-e_1-04_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_046_PIANO046_MID--AUDIO-split_07-06-17_Piano-e_2-02_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_049_PIANO049_MID--AUDIO-split_07-06-17_Piano-e_2-06_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_05_R1_2008_01-04_ORIG_MID--AUDIO_05_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_17_R1_2008_01-04_ORIG_MID--AUDIO_17_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_09_R1_2008_01-05_ORIG_MID--AUDIO_09_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_18_R1_2008_01-04_ORIG_MID--AUDIO_18_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_04_R1_2014_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_066_PIANO066_MID--AUDIO-split_07-07-17_Piano-e_3-02_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_10_R1_2008_01-04_ORIG_MID--AUDIO_10_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_05_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_09_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_065_PIANO065_MID--AUDIO-split_07-07-17_Piano-e_3-01_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_23_R1_2011_MID--AUDIO_R1-D9_03_Track03_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_16_R1_2015_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_09_R1_2009_01-04_ORIG_MID--AUDIO_09_R1_2009_09_R1_2009_03_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_XP_15_R1_2004_03_ORIG_MID--AUDIO_15_R1_2004_03_Track03_wav.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_16_R1_2013_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_17_R1_2009_01-03_ORIG_MID--AUDIO_17_R1_2009_17_R1_2009_01_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_16_R1_2006_01-04_ORIG_MID--AUDIO_16_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_17_R1_2009_01-03_ORIG_MID--AUDIO_17_R1_2009_17_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_02_R1_2014_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_XP_11_R1_2004_01-02_ORIG_MID--AUDIO_11_R1_2004_02_Track02_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_23_R3_2011_MID--AUDIO_R3-D8_04_Track04_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_06_R1_2006_01-04_ORIG_MID--AUDIO_06_R1_2006_02_Track02_wav.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_11-13_R1_2014_MID--AUDIO_13_R1_2014_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_19-20_R1_2014_MID--AUDIO_19_R1_2014_wav--8.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_10_Track10_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_20_R1_2009_01-05_ORIG_MID--AUDIO_20_R1_2009_20_R1_2009_02_WAV.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_Recital17-19_MID--AUDIO_17_R1_2018_wav--2.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_08_R1_2008_01-05_ORIG_MID--AUDIO_08_R1_2008_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_04_R1_2014_wav--5.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_17_R1_2015_wav--3.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_04_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_05_R1_2008_01-04_ORIG_MID--AUDIO_05_R1_2008_wav--4.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_08_Track08_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_12_R1_2006_01-08_ORIG_MID--AUDIO_12_R1_2006_07_Track07_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_12_R3_2008_01-04_ORIG_MID--AUDIO_12_R3_2008_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_11_R1_2015_wav--5.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_XP_08_R1_2004_04-06_ORIG_MID--AUDIO_08_R1_2004_05_Track05_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_03_R2_2011_MID--AUDIO_R2-D1_06_Track06_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_05_R1_2011_MID--AUDIO_R1-D2_12_Track12_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_08_R1_2008_01-05_ORIG_MID--AUDIO_08_R1_2008_wav--2.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_03_7_10_13_Group_MID--AUDIO_17_R3_2013_wav--1.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_052_PIANO052_MID--AUDIO-split_07-06-17_Piano-e_3-03_wav--3.midi.pickle  
      inflating: e_piano/test/ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_09_R1_2013_wav--3.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_XP_14_R1_2004_04_ORIG_MID--AUDIO_14_R1_2004_04_Track04_wav.midi.pickle  
      inflating: e_piano/test/MIDI-Unprocessed_XP_14_R1_2004_04_ORIG_MID--AUDIO_14_R1_2004_05_Track05_wav.midi.pickle  
    /content/gen1_1
    /content/gen1_1/MusicTransformer-Pytorch
    /content/gen1_1
    mv: cannot stat 'midi-neural-processor': No such file or directory
    /content/gen1_1/MusicTransformer-Pytorch
    /content/gen1_1/midi_processor
    Mon Sep  5 21:36:39 2022


# One-Click Generation (After Setup): 


## Assign Values for Generation


```python
#@title Generate, Plot, Graph, Save, Download, and Render the resulting output
%cd $YourHomeDir/$YourProjectSubDir/MusicTransformer-Pytorch/

number_of_tokens_to_generate = 2048 #@param {type:"slider", min:1, max:2048, step:1}
priming_sequence_length = 505 #@param {type:"slider", min:1, max:2048, step:8}
maximum_possible_output_sequence = 2048

select_model = YourHomeDir + '/' + YourProjectSubDir + '/' + "MusicTransformer-Pytorch/rpr/results/best_loss_weights.pickle" 

custom_MIDI = "" #@param {type:"string"}


# get to the right dir
%cd $YourHomeDir/$YourProjectSubDir/midi_processor


print(time.asctime( time.localtime( time.time() ) ))
```

    /content/gen1_1/MusicTransformer-Pytorch
    /content/gen1_1/midi_processor
    Mon Sep  5 21:36:39 2022


## Make Your Music! (repeat as many times as you like) 🎵


```python
# auto-time-stamped subdir to save the music
DirToSaveGeneratedMusic = YourHomeDir + '/' + YourProjectSubDir + '/' + "MusicTransformer-Pytorch/generated/" + str(int(time.time()))
#save to gdrive
DirToSaveGeneratedMusic = "/content/gdrive/MyDrive/generated/" + str(int(time.time()))



# get to the right dir
%cd $YourHomeDir/$YourProjectSubDir/MusicTransformer-Pytorch

# time keeping
startt = time.asctime( time.localtime( time.time() ) )

!python3 generate.py -output_dir $DirToSaveGeneratedMusic -model_weights=$select_model --rpr \
 -target_seq_length=$number_of_tokens_to_generate -num_prime=$priming_sequence_length \
 -max_sequence=$maximum_possible_output_sequence $custom_MIDI #

print('Successfully exported the output to output folder. To primer.mid and rand.mid')

count = count+1

print(startt)
print(time.asctime( time.localtime( time.time() ) ))
```

    /content/gen1_1/MusicTransformer-Pytorch
    =========================
    midi_root: ./dataset/e_piano/
    output_dir: /content/gdrive/MyDrive/generated/1662413936
    primer_file: None
    force_cpu: False
    
    target_seq_length: 2048
    num_prime: 505
    model_weights: /content//gen1_1/MusicTransformer-Pytorch/rpr/results/best_loss_weights.pickle
    beam: 0
    
    rpr: True
    max_sequence: 2048
    n_layers: 6
    num_heads: 8
    d_model: 512
    
    dim_feedforward: 1024
    =========================
    
    Using primer index: 1 ( ./dataset/e_piano/test/MIDI-Unprocessed_02_R2_2011_MID--AUDIO_R2-D1_02_Track02_wav.midi.pickle )
    RAND DIST
    Generating sequence of max length: 2048
    550 / 2048
    600 / 2048
    650 / 2048
    700 / 2048
    750 / 2048
    800 / 2048
    850 / 2048
    900 / 2048
    950 / 2048
    1000 / 2048
    1050 / 2048
    1100 / 2048
    1150 / 2048
    1200 / 2048
    1250 / 2048
    1300 / 2048
    1350 / 2048
    1400 / 2048
    1450 / 2048
    1500 / 2048
    1550 / 2048
    1600 / 2048
    1650 / 2048
    1700 / 2048
    1750 / 2048
    1800 / 2048
    1850 / 2048
    1900 / 2048
    1950 / 2048
    2000 / 2048
    info removed pitch: 76
    info removed pitch: 69
    Successfully exported the output to output folder. To primer.mid and rand.mid
    Mon Sep  5 21:38:56 2022
    Mon Sep  5 21:40:29 2022


# Check the 'Generated' folder in your Google Drive to listen to your midis! 🎉🎉🎉
