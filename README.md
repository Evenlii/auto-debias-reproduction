1. Context-Debias code is adapted from this repository of the original paper (https://github.com/kanekomasahiro/context-debias). Run preprocessing and then run debiasing on BERT-based pre-trained model using the word lists (age or disability). Save the model in checkpoint-best. These experiments were executed on a computer with local GPU. 

2. To perform SEAT tests, make sure the model being loaded is from step 1 and then execute run_seat.sh. GPU not required.

3. To perform downstream GLEU tests, run GLEU.ipynb. Since GPU is required, we ran this code on Google Colab. 
