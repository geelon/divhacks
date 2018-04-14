---
title: Classifying Handwriting Digits
---


## Define the Problem

The description of the challange can be found under the GitHub page [IBM DivHacks Project](https://github.com/JinjunXiong/divhacks)


### Some Statistis of the MNIST dataset

- Average accuracy over all algorithms

| Rank    | Accuracy  | Number |
|---------|:---------:|--------|
|1        | 0.915099  | 7      |
|2        | 0.905362  | 1      |
|3        | 0.897518  | 0      |
|4        | 0.88166   | 6      |
|5        | 0.864772  | 4      |
|6        | 0.856436  | 3      |
|7        | 0.850741  | 2      |
|8        | 0.846041  | 9      |
|9        | 0.830316  | 5      |
|10       | 0.822651  | 8      |

- Accuracy of each algorithm

| Algorithm    | Accuracy  | 
|---------|:---------:|
|ALG1|0.890490909091|
|ALG2|0.928381818182|
|ALG3|0.997109090909|
|ALG4|0.928054545455|
|ALG5|0.999109090909|
|ALG6|0.131472727273|
|ALG7|0.871054545455|
|ALG8|0.174563636364|
|ALG9|0.927763636364|
|ALG10|0.957654545455|
|ALG11|0.852309090909|
|ALG12|0.926127272727|
|ALG13|0.924654545455|
|ALG14|1.0|
|ALG15|1.0|
|ALG16|0.885|
|ALG17|0.929709090909|
|ALG18|0.923872727273|
|ALG19|0.999109090909|
|ALG20|0.983309090909|
|ALG21|0.996381818182|

- Accuracy for each number

| Digit    | Accuracy  | 
|---------|:---------:|
|0|0.0427725991636|
|1|0.0431331794362|
|2|0.0404640538741|
|3|0.0407437254518|
|4|0.0411503299929|
|5|0.0395603200646|
|6|0.0419722218051|
|7|0.0435424257391|
|8|0.0391540599813|
|9|0.0402745315714|

We processed the original data file, using a threshold value to filter our training data. You can do so by entering in command line:

```bash
python make_binary.py filename threshold
```

From there we created a MLP that attempts to predict the percent of classifiers that correctly classified an image.

## Design of Project
Our initial goal was to build a regression model whose input is an MNIST image, and whose output predicts the percentage of algorithms that would classify it correctly (PERCENT\_CORRECT). From there, we would determine a threshold to classify these images as EASY or HARD to classify. The rationale was the assumption that certain features that lead to misclassification may appear to different degrees. That is, these are not binary features. As such, it seems that we would lose information if we were to just train on EASY/HARD labels rather than PERCENT_CORRECT.

However, the regression model did not converge, and given our time/compute constraints, we opted to resort to the binary classification model. Our classifier is a two-layer MLP, with hidden dimensions 1024 and 128. The results of this are shown below.

## Results

### Training Results

|Result        |Value              |
|--------------|-------------------|
|Train Accuracy|0.7108727097511292 |
|Val Accuracy  | 0.7368000149726868|
|Test Accuracy |0.670799970626831  |

### Characteristics of the Data Set

The numbers 1, 7, and 0 were the easiest to predict, likely because they are either curved or stroked but not both. 1 generally contains a single stroke, so no features are missing from it, even when it's written quickly. Like 1, 7 has few features. It is easily distinguished from 1 because of the horizontal bar at its top and the occasional horizontal bar at its middle. 0 has the shape of an oval, but it can be recognized even when it has a slight opening at the top. No other digit has one complete loop and nothing else. 6 and 4 had less accurate predictions than 1, 7, and 0, but they were easier to predict than 2, 3, 5, 8, and 9.

## Future Direction
We saw, in the analysis of the training data, that two algorithms, ALG14 and ALG15 consistently had the correct predictions. It follows that they contribute nothing to the learner. On the other hand, ALG6 obtained a 0.131 accuracy (note that guessing at random should obtain about 0.10 accuracy). This suggests that we should take these accuracies into account when generating the EASY/HARD classification. 

It is unsurprising if a classifier that is near random classifies an image incorrectly, and therefore, we should take that less into account when deciding when an image is EASY/HARD. Conversely, if an algorithm that is almost always correct errs, then we should give more weight to that error.

In fact, generating EASY/HARD labels using the weighted accuracies may reflect more 'real' features in the data (e.g. reducing noise from poor/near-random classifiers).

We also may consider using different representations of MNIST images. Similar to word embeddings, we can take the last hidden layer of the output of a CNN (trained on the original MNIST problem of classifying digits) as the representation of the image. This may allow relevant features to be more readily accessible and tied with the EASY/HARD labels as well.
