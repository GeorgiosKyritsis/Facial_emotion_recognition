# Facial_emotion_recognition

VGG13, VGG16, VGG19, ResNet18, ResNet34, ResNet50, GoogLeNet, wideResNet -> Train from scratch these models by running the train.py. Get the predictions by loading the weigths by using the predict.py. All these folders contain the necessary files to train the model (create_datasets, callbacks, etc.).

Colorize_grayscale_images ->  contains the code that loads a pre-trained network to convert grayscale images to rgb.

Action_units -> train svm classifiers for all emotion pairs and for emotion triples and save them to files for later use.

SVM_transfer_learning -> train svm classifiers. As features we use the features extracted from CNNs

Transfer_learning -> Extract features from CNNs and save them to files for later use. Fine-tune pre-trained networks by freezing some layers and updating the rest.

Age ->  CNN model for age estimation (train.py for training from scratch)
Gender -> CNN model for gender classification (train.py for training from scratch)
Emotion -> CNN model for emotion recognition (train.py for training from scratch)

Multi-task learning -> Multi-task learning with 3 signals (age, gender, emotion)

demo.ipynb -> Gives a demo by using the best CNN architecture (wide ResNet) and pre-trained Action Unit models

All data and log files can be downloaded by the following dropbox link:

[data and logs](https://www.dropbox.com/sh/3f20b8l9x9aevbf/AAAPFnmWLCKo1m4nbesSdW81a?dl=0)

Just replace the folders data and logs with the folders downloaded by dropbox.
