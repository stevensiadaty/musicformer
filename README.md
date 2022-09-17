<a href="https://colab.research.google.com/github/stevensiadaty/musicformer/blob/main/notebooks/ColabGenerateMusicWithAI.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## :point_up: Click the "Open Colab" button
# Generate music, given neural net (Google Colab Version)
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

  

# Check the 'Generated' folder in your Google Drive to listen to your midis! 🎉🎉🎉
