# Can we break CAPTCHA?

## Overview
CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) has for long been a trusted mechanism to prevent automated bots from accessing various, especially web, services. However, with the advent of machine learning and deep learning, it has also become increasingly vulnerable to automated attacks.

With this project, we aim to explore the feasibility of breaking CAPTCHA using deep learning techniques. Specifically, can we train a local on-system model to extract text from CAPTCHA images? The project involves the following main tasks:

1. **Classification:** Train a neural network to categorize CAPTCHA images into predefined labels.
2. **Generation:** Train a model to extract text directly from CAPTCHA images.

## Table of Contents
1. [Task 0 - Dataset](#task-0---dataset)
2. [Task 1 - Classification](#task-1---classification)
3. [Task 2 - Generation](#task-2---generation)
5. [Challenges and Insights](#challenges-and-insights)
6. [Future Work](#future-work)
7. [References](#references)


## Task 0 - Dataset
To train the model, a synthetic dataset was generated with three levels of difficulty, with the following characteristics. I used the OpenCV library to generate the synthetic images.

### Easy Set
- Fixed font style and size.
- Constant capitalization.
- Plain white background.

### Hard Set
- Multiple font style and sizes.
- Varying capitalization.
- Noisy backgrounds.

The number of labels (and their respective frequencies) have been set using MACROS in the code. This allows for easy adjustment of the dataset size and complexity, based on the task requirements.

## Task 1 - Classification

### DataSet:

A total of 50 labels, each with 200 images, was used for each complexity; giving us a total of **20,000 images** across all sets. We then had a 80-20 split for training and testing, giving us 16,000 training images and 4,000 testing images.

### Model Architecture
- CNN with 3 convolutional blocks (32→64→128 channels)
- Each block: 2x Conv2D(3x3) → BatchNorm → ReLU → MaxPool(2x2)
- FC layers: 128→256→128→num_classes
- Input: 128x128x3 RGB images 
- Optimizer: Adam, Loss: CrossEntropy

### Process:
- For the task, I took the model to be a fixed entity and **focused on hyperparameter tuning**, specifically learning rate and number of epochs. Once I found a sweet spot for these, I **increased the sample size until I reached a reasonable accuracy**. This is not the best approach, but it reduces the variability of the problem, while still maintaining the essence of the task.
- So, I struggled in the beginning with the various parameters understanding how they affect the model and how to tune them. It took me almost a whole day to get to the double digit accuracy (and 30-40 minutes runtime for each cycle of training wasnt quite entertaining).
- But once I got the hang of it (basically how various parameters affect the model), I was able to quickly iterate through different hyperparameters and sample sizes to get to a good accuracy. Within another day I was up to 50% (with 150 images per label), and soon enough I had crossed 70% mark (with 170 images per label).
- At this point, it was only a matter of increasing the sample size. However, with the increasing runtimes (59:40:86 for 150 images per label), I had to limit the number of samples to 200 for each label (otherwise my PC would literally be cooked!). This was a good tradeoff between accuracy and runtime.

### Analysis:
- The model seem to not have made a single mis-match between similar-looking characters like 'O' and '0', 'l' and '1', etc. This shows that either the model was able to learn the features of the characters well; or was independent of them.
- It was also interesting to note that the mdoel couldn't retain the number of characters in the label. For instance, "jTpey" to "jPVHHsOgJ" or "ZFUmS" to "iFgynLNU". These are quite rare in the dataset, but reflect some inherent issues in the learnings of the model.
- However, the most interesting trend was that certain labels to got misclassified into a specific label consistently. For instance, "gaYed" was misclassified 5 times, out of which it was labelled "daxVt" 3 timws and "jTpey" 2 times. And "QNOSs" got misclassified as "UQOyGTb" 7 out of 7 times, "Cpkry" to "eFfoyv" 4 out of 4 times. This shows that the model was able to learn some commonality between these words, which are not apparent to the human eye.

### Results:
- I was able to achieve an accuracy of **70.33% with 150 images for each label** at 0.001 LR and 25 epochs. The training process took 59:40:86 (hours:minutes:seconds) to complete, and one can find the train plot [here](./Task1/Plots/16.png).
- The final model achieved **76.62% accuracy with 200 images for each label** at 0.001 LR and 25 epochs. The train plot for the same can be accesed [here](./Task1/Plots/19.png).
- There is also a [results.csv](./Task1/results.csv) file containing the test results for the final (most accurate) model.
- One can access various plots for each model tested and trained during the whole process of tuning the hyperparameters [here](./Task1/Plots).

### Challenges:
- The biggest challenge was to understand the various **hyperparameters** and how they affect the model. This was a new concept for me and took some time to get used to.
- The second challenge was the **runtime**. With each cycle of training taking 30-40 minutes (and later with larger datasets, almost an hour!), it was difficult to iterate through different hyperparameters and sample sizes.


## Task 2 - Generation

### Note
It was quite difficult (at least with the given resources) to train an all-sufficient model that would work for both the easy and hard sets together. So, rather than jumping to that all-supreme model, I have broken down the problem into the following sub-problems. We would start with the easy set and slowly progress to the hard set. This would give us a better understanding of the problem and the challenges associated with it (and also zero down on where the problem gains its true complexity from).

### Model Architecture:
1. CNN with 4 blocks (32→64→128→256 channels)
2. Each CNN block: Conv2D + BatchNorm + ReLU + Dropout + MaxPool
3. Bidirectional GRU (2 layers, hidden=256)
4. Final layer: BatchNorm + Dropout + Linear → VOCAB_SIZE
5. Mixed precision training with gradient scaling
6. Loss: Per-character CrossEntropy with stability checks
7. Regularization: Spatial Dropout (0.2), Gradient Clipping

### Sub-Problems and Results:

#### Easy Set
- 20000 images from the easy set were used for model training, with a 80-20 split for training and validation.
- Achieved a **Character Accuracy of 99.46%**. The train plot for the same can be accessed [here](./Task2/Plots/2.png).
- On testing with another (external to the train-validation set) 1000 image set, the model achieved a **Word Level Accuracy of 79.40%**.

#### + Variable Capitalisation
- 25000 images with contrasting background and variable capitalisation were used for model training, with a 80-20 split for training and validation.
- Achieved a **Character Accuracy of 95.53%** and a Word Level Accuracy of 74.32%. The train plot for the same can be accessed [here](./Task2/Plots/9.png).
- On testing with another (external to the train-validation set) 5000 image set, the model achieved a **Word Level Accuracy of 67.32%**.
- The model was able to generalize well to the unseen data, but the accuracy dropped due to the increased complexity of the task.
- The final model parameters were set as LR=0.001, EPOCHS=100 and BATCH_SIZE=32.

#### + Noisy Background
- Added noisy background and elements over the text.
- Achieved a **Character Accuracy of 95.43** and a Word Level Accuracy of 74.18%. The train plot for the same can be accessed [here](./Task2/Plots/10.png).
- On testing with another (external to the train-validation set) 5000 image set, the model achieved a **Word Level Accuracy of 74.88%**.
- Not only was the model able to maintain the accuracy, but also improve upon it. This shows that the model was able to learn the additional complexity of the task.
- The final model parameters were set as LR=0.001, EPOCHS=100 and BATCH_SIZE=32.

#### Extreme
- Added Noise and variable font styles, sizes and scales to the dataset.
- Achieved a **Character Accuracy of 56.76** but a Word Level Accuracy of a mere 8.96%. The train plot for the same can be accessed [here](./Task2/Plots/11.png).
- On testing with another (external to the train-validation set) 5000 image set, the model achieved a **Word Level Accuracy of 9.74%**.
- The accuracy dropped significantly due to the increased complexity of the task.
- Same model parameters were used; LR=0.001, EPOCHS=100 and BATCH_SIZE=32.

#### Further Improvements
- Further work is needed to exactly pinpoint the areas where the model is failing. This would involve individually checking for variable font styles, sizes and scales, and how they affect the model.
- I also wanted to expand the work to cover variable positions and orientations of the text, but the runtime was a big issue. With each cycle of training taking almost 2 hours, it was difficult to iterate through different hyperparameters and sample sizes.
- Additionally, I have used OpenCV for image and dataset generation. This is not the most efficient way to do it, and more task-specific libraries like PIL could be used.

### Results:
- While, as stated, there is still a lot of scope of improvement before reaching the complete CAPTCHA solving model, the current model is able to solve the easy set with a **Word Level Accuracy of 79.4%** and the hard set with a **Word Level Accuracy of 74.88%**.
- I was able to achieve an accuracy of **70.33% with 150 images for each label** at 0.001 LR and 25 epochs. The training process took 59:40:86 (hours:minutes:seconds) to complete, and one can find the train plot [here](./Task1/Plots/16.png).
- The final model achieved **76.62% accuracy with 200 images for each label** at 0.001 LR and 25 epochs. The train plot for the same can be accesed [here](./Task1/Plots/19.png).
- There is also a [results.csv](./Task1/results.csv) file containing the test results for the final (most accurate) model.
- One can access various plots for each model tested and trained during the whole process of tuning the hyperparameters [here](./Task1/Plots).

### Analysis:
The following inferences are based on the actual and predicted text from the external 5000 image test set.

#### Variable Capitalisation
- Single character off prediction constituted over 78% (1290 out of 1638) of the errors; signifying that the model was able to learn the features of the characters well and was able to sequence them also.
- And most of these character mismatch occured between similar-looking characters like 'o' and 'c' ("zZocdNc" to "zZccdNc"), 'l' and 'I' ("zzPoQNI" to "zzPoQNl"), etc. This shows tat the model learnings were somewhat dependent on the visual features of the characters.

#### Noisy Background
- Similar trend for single character off predictions (1034 out of 1256) and similar-looking characters causing confusion ("zQAjvguWk" to "zQAjyguWk").
- Only 4 words had more than 3 characters not matching (hRiVliYlH -> hRiIIilIH, jliIdGWxJ -> jjlldGWJJ, jvYhouakD -> jvYYbaukD, rjIyiJKa -> rjjvJKaa), a reduction from 9 for the variable capitalisation set.

#### Extreme
- While roughly 66% of the words had less than or equal to 3 characters not matching, this is a significant reduction from the previous sets.
- Also, a lot of duplicated characters in close vicinity are present in the predicted text, signifying that the model wasn't certain about the characters it was predicting. This might be due to parts of the text getting overlayed by noise elements.

### Issues/Challenges:
- The biggest challenge was the **complexity** (and noveltty for me and I believe in the local setup) of the problem. It was difficult to prepare a model that would work for all the sets together. I had to, unwillingly, break down the problem into sub-problems and even take help from the internet.
- The second challenge was the **runtime**. Each cycle towards the end took well over 2 hours, significantly reducing the number of iterations I could do.

## Future Work
- The biggest scope of improvement is to **increase the complexity of the problem**. This would involve adding variable fonts, sizes and positions to the dataset. This would make the model more robust and generalizable.
- The second scope of improvement is to **train a single model that can solve all the sets together**. This would require a lot of resources and time, but would be a great achievement.
- The third scope of improvement is to **improve the model architecture**. Currently, with little to no knowledge, I am not that confident in the model architecture (or even the internal workings of my current model). I believe there is a scope of improvement there. And that with more knowledge and experience, I would be able to capitalize on that.

## References
Statquest
1. https://www.youtube.com/watch?v=L35fFDpwIM4
2. https://www.youtube.com/watch?v=L35fFDpwIM4
3. https://www.youtube.com/watch?v=iyn2zdALii8
4. https://www.youtube.com/watch?v=GKZoOHXGcLo

3Blue1Brown
1. https://www.youtube.com/watch?v=aircAruvnKk
2. https://www.youtube.com/watch?v=IHZwWFHWa-w

ANN vs CNN vs RNN
1. https://www.youtube.com/watch?v=u7obuspdQu4

## Acknowledgements
- Precog Team for providing the opportunity to work on this project.
- Various online resources for providing the knowledge and guidance required to complete the project.
- (Inadvertently) AI tools (for pointing out how I cannot parse a tensor as a string).

<hr>

### For more details, please refer to [CODEBASE.md](./CODEBASE.md).

<hr>
<hr>
<hr>
