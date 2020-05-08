# Speech-Emotion-Detection

![](https://miro.medium.com/max/1400/1*c808FqIfxkC5-ZKqZIv30A.jpeg)

Classifying speech to emotion is challenging because of its subjective nature. This is easy to observe since this task can be challenging for humans, let alone machines. Potential applications for classifying speech to emotion are numerous, including but not exclusive to, call centers, AI assistants, counseling, and veracity tests.

This repository contains our Fall 2019 term project for MIS 281N Advanced Predictive Modeling as part of the MS Business Analytics curriculum at UT Austin. It pulls from data found in the Ryerson Audio-Visual Database of Emotional Speech and Song (see here: https://zenodo.org/record/1188976).

## RAVDESS Data Set

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 7356 files (total size: 24.8 GB). The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. All conditions are available in three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).

Each of the 7356 RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 02–01–06–01–02–01–12.mp4). These identifiers define the stimulus characteristics:

Filename identifiers:

•	Modality (01 = full-AV, 02 = video-only, 03 = audio-only)

•	Vocal channel (01 = speech, 02 = song).

•	Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).

•	Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.

•	Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).

•	Repetition (01 = 1st repetition, 02 = 2nd repetition).

•	Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


## Approach

- Read WAV files in by using the libROSA package in Python
- Extract features from the audio time series created by libROSA using functions from the libROSA package (MFCCs, Chroma, and Mel spectrograms)
- Construct a series of models from various readily available Python packages
- Tune hyper-parameters for the models using the Optuna framework
- Ensemble models using soft voting classifier to improve performance
- Use PCA to extract features for use in a CNN
- Use VGG 16 model on images of spectrograms
- Ensemble the CNN with the VGG 16 to improve performance



## Modelling Approach

### Traditional Machine Learning Models:
- Simple models: K-Nearest Neighbors, Logistic Regression, Decision Tree
- Ensemble models: Bagging (Random Forest), Boosting (XG Boost, LightGBM)
- Multilayer Perceptron Classifier
- Soft Voting Classifier ensembles

### Deep Learning:
- Design a convolutional neural network and train it on a combination of MFCC, Mel Scale, and Chroma features
- Take a more robust, widely tested convolutional architecture and train it on Mel spectrograms

### Final Approach:
In our final approach, we decided to create an ensemble of the neural nets developed in Approach 1 and Approach 2. To do this, we used the soft voting technique to combine the resultant posterior probabilities from both models. We found that giving a weight of three to posterior probabilities from Approach 1 and weight of two to posterior probabilities from Approach 2 resulted in better overall accuracy.


## Future Work

An alternate approach that could be explored for this problem is splitting the classifying task into two distinct problems. A separate model could be used to classify gender and then separate models for each gender to classify emotion could be utilized. This could possibly lead to a performance improvement by segregating the task of emotion classification by gender.
As with many data science projects, different features could be used and/or engineered. Some possible features to explore concerning speech would be MFCC Filterbanks or features extracted using the perceptual linear predictive (PLP) technique. These features could affect the performance of models in the emotion classification task.
It would be interesting to see how a human classifying the audio would measure up to our models, however, finding someone willing to listen to more than 2,400 audio clips may be a challenge in of itself because a person can only listen to “the children are talking by the door” or “the dogs are sitting by the door” so many times.

## References:

[1] https://zenodo.org/record/1188976#.XeqDKej0mMo

[2] http://conference.scipy.org/proceedings/scipy2015/pdfs/brian_mcfee.pdf

[3] https://towardsdatascience.com/ok-google-how-to-do-speech-recognition-f77b5d7cbe0b

[4] http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

[5] https://en.wikipedia.org/wiki/Frequency_domain

[6] http://www.nyu.edu/classes/bello/MIR_files/tonality.pdf

[7] https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/EmotionsRecognition.ipynb

[8] https://towardsdatascience.com/speech-emotion-recognition-with-convolution-neural-network-1e6bb7130ce3

[9] https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/

[10] https://labrosa.ee.columbia.edu/matlab/chroma-ansyn/

[11] https://librosa.github.io/librosa/index.html

[12]https://www.researchgate.net/publication/283864379_Proposed_combination_of_PCA_and_MFCC_feature_extraction_in_speech_recognition_system



## [Link to Towards Data Science Blog](https://medium.com/@tushar.gupta_47854/speech-emotion-detection-74337966cf2)
