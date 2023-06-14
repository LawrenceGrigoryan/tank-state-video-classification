![poster](docs/poster.png)

This repository contains a solution for the video analytics challenge organized by Sibur Digital, which was held from May 20 to June 7, 2023. 
It is required to develop an action recognition model to predict the current state of a railway tank. \
The key idea of the challenge is to develop a model that generalizes well on unseen data using a small dataset.

## Results

- **4th place**, falling behind the first place by ~0.003 points


## Model requirements and limitations:

1. Classes: no_action, bridge_up, bridge_down, train_in_out 
2. Evaluation metric: F1 Macro
3. Only 496 videos in the dataset covering limited types of train stations, weather and situations
4. Test set is more diverse and contains cases that differ a lot from train data
5. Model runs on edge devices (camera) using just one CPU
6. Up to 18 minutes to predict 1200 test videos (~0.9 sec for 1 video)


https://github.com/LawrenceGrigoryan/wagon-state-sibur/assets/57874123/9a73dea5-5bbc-4821-8cc4-4993bff50800


## Team (`ararat_tennis_club` on leaderboard):
- Lavrentiy Grigoryan (https://www.linkedin.com/in/lawrencegrigoryan/)
- Vladislav Alekseev (https://www.linkedin.com/in/vladislav-alekseev/)
- Galina Burdukovskaya (https://www.linkedin.com/in/galina-burdukovskaia-7502ab21a/)


## Hypotheses tested

1. Average a video to one frame and classify just this one averaged frame
2. Classify whole videos (some frames from it) using a 3D convolutional network
3. Object detection and its post-processing to classify videos

