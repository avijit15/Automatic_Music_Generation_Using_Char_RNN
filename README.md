# Automatic_Music_Generation_Using_Char_RNN
In this project I have tried to build a deep learning model, where it automatically generate some music.

# Input
Here I have used ABC notation of some musics to train my model. The input.txt file consists all the ABC notation of different musics. Here is an example of how this input file looks like

Example 1:

X: 1

T:A and D

% Nottingham Music Database

S:EF

Y:AB

M:4/4

K:A

M:6/8

P:A
f|"A"ecc c2f|"A"ecc c2f|"A"ecc c2f|"Bm"BcB "E7"B2f|
"A"ecc c2f|"A"ecc c2c/2d/2|"D"efe "E7"dcB| [1"A"Ace a2:|
 [2"A"Ace ag=g||
K:D
P:B
"D"f2f Fdd|"D"AFA f2e/2f/2|"G"g2g ecd|"Em"efd "A7"cBA|
"D"f^ef dcd|"D"AFA f=ef|"G"gfg "A7"ABc |[1"D"d3 d2e:|[2"D"d3 d2||




Example 2:


X: 2

T:Abacus

% Nottingham Music Database

S:By Hugh Barwell, via Phil Rowe

M:6/8

K:G

"G"g2g B^AB|d2d G3|"Em"GAB "Am"A2A|"D7"ABc "G"BAG|
"G"g2g B^AB|d2d G2G|"Em"GAB "Am"A2G|"D7"FGA "G"G3:||:
"D7"A^GA DFA|"G"B^AB G3|"A7"^c=c^c A^ce|"D7"fef def|
"G"g2g de=f|"E7"e2e Bcd|"Am"c2c "D7"Adc| [1"G"B2A G3:|
 [2"G"B2A G2F||"Em"E2E G2G|B2B e2e|"Am"c2A "B7"FBA|"Em"G2F E3|"Em"EFG "Am"ABc|
"B7"B^c^d "Em"e2e|"F#7"f2f f2e|"B7"^def BAF|"Em"E2E G2G|B2B e2e|
"Am"c2A "B7"FBA|"Em"G2F E3|"Em"EFG "Am"ABc|"B7"B^c^d "Em"e2e|
"F#7"f2e "B7"^def |[1"Em"e3 "D7"d3:|[2"Em"e3 "E7"e3||

# Model
For training I used Char RNN (LSTM). I have to divide the input in batches and as there are some sequence information hence I have to use sequence model. here is the model architectutre I have used for this

Model: "sequential"

Layer (type)                 Output Shape              Param   

embedding (Embedding)        (16, 64, 512)             44032     
_________________________________________________________________
lstm (LSTM)                  (16, 64, 256)             787456    
_________________________________________________________________
dropout (Dropout)            (16, 64, 256)             0         
_________________________________________________________________
lstm_1 (LSTM)                (16, 64, 256)             525312    
_________________________________________________________________
dropout_1 (Dropout)          (16, 64, 256)             0         
_________________________________________________________________
lstm_2 (LSTM)                (16, 64, 256)             525312    
_________________________________________________________________
dropout_2 (Dropout)          (16, 64, 256)             0         
_________________________________________________________________
time_distributed (TimeDistri (16, 64, 86)              22102     
_________________________________________________________________
activation (Activation)      (16, 64, 86)              0         
_________________________________________________________________
Total params: 1,904,214

Trainable params: 1,904,214

Non-trainable params: 0
_________________________________________________________________

# How to run the code?
For running the code, you can clone this repo to your local and then you can run the code from the terminal using below command

**python music_generation.py 100 --len 1000**

Note: 
1. Here 100 denotes the epoch number, that means depending on this number it will take the model from the ./model folder. 100 means model is trained for 100 epochs. similiarly you can use 50 or 60 to get the model which is trained for 50 and 60 epochs respectively.
2. --len 1000 denotes the output size. That means it will generate a output of length of 1000. The output can contain more than one musics in ABC notation format separeted by some empty lines.
3. You can take any one of the abc notation and for creating the sound file. Luckily there is a open source site, which converts the abc notation to sound file, you can use that one for testing. Here is the link for that website
https://www.abcjs.net/abcjs-editor.html


