![poster](docs/poster.png)

This repository contains a solution for the video analytics challenge organized by Sibur Digital, which was held from May 20 to June 7, 2023. It is required to develop an action recognition model to predict the current state of a railway tank.
The key idea of the challenge is to develop a model that generalizes well on unseen data using a small dataset.

## Model requirements and limitations:

1. 4 classes: no_action, bridge_up, bridge_down, train_in_out 
2. Only 496 videos in the dataset covering limited types of train stations and situations
3. Test set is more diverse and contains cases that differ a lot from train data
4. Model runs on edge device (camera) using just one CPU
5. Up to 18 minutes to predict 1200 test videos (~0.9 sec for 1 video)


https://github.com/LawrenceGrigoryan/wagon-state-sibur/assets/57874123/9a73dea5-5bbc-4821-8cc4-4993bff50800


## Team:
- Lavrentiy Grigoryan (https://www.linkedin.com/in/lawrencegrigoryan/)
- Vladislav Alekseev (https://www.linkedin.com/in/vladislav-alekseev/)
- Galina Burdukovskaya (https://www.linkedin.com/in/galina-burdukovskaia-7502ab21a/)


## Hypotheses tested

1. Average a video to one frame and classify just this one averaged frame
2. Classify whole videos (some frames from it) using a 3D convolutional network
3. 
