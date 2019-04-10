# Overview: Regression / Multiclass Classification

### Target: Prdict how long it takes for a pet to be adopted
### Metric: Quadratic weighted kappa
### Data: Tabluar data + Image data + Text data
### Special feature: Unstable evaluation metric; Various sorts of data 

## 6th solution by Benjamin Minixhofer: https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88690#latest-512360

**source code:** https://www.kaggle.com/bminixhofer/6th-place-solution-code

**Solution overview:**

- Special validation 1: repeated stratified cross-validation with 5 splits and 5 repeats in my submission to reduce the effect of randomness

- Special validation 2: I used a kind of nested K-Fold CV. In an outer split, I split the dataset into a local test set and a train+valid set. The train+valid set is then also split into K-Folds and a model is trained on each fold. The predictions of each model on the data in the local test set are then averaged to get a QWK score. 

- binned RescuerID count into 10 quantile bins. This greatly increased my score (almost 0.01 on the public LB) and intuitively makes sense because rather than the exact amount of pets rescued, the type of rescuer should matter (e. g. whether it is corporation or a single person). 

- All my neural networks use the same basic structure. To make the neural networks as diverse as possible, I modified:
(1) The loss function; (2) The image activations; (3) How the network treats text. 

- All network were trained using Cyclic LR and the Adam optimizer.

- Also, instead of using the regular image features I used image features extracted from the same densenet121 model on flipped images to increase diversity. Flipping the images significantly increased my score though.

- I found that raw image activations were too high dimensional to work well, so I trained an image activation extractor NN model. This model has the top layer activations from Densenet121 as input and tries to predict the Type, Age and Breed1 of the pet. It is trained on the train + test set.

- Additionally, I used LightGBM and xlearn. Text is encoded using SVD of a TfIdf matrix with 3-grams. The SVD has 10 components for xlearn and 120 components for LightGBM.

- For xlearn, it worked best to treat all features as categorical, so I binned the image extractor activations and the SVD representation into 10 quantile bins. You can see the structure of the input features for the two models below.

- The second problem was that neural networks were weighted significantly lower than expected. Of course, I didn't want to overfit to the public LB so adjusting the coefficients manually was definitely a bad idea. **What i came up with is setting the sample weights of the linear regression to the out-of-fold predictions of an adversarial validation model. As such, the samples which were seen as closer to the test distribution were weighted higher and the coefficients of the neural networks increased strongly.**

- Binning the pet age into 20 quantile bins greatly increased my CV score and it intuitively makes sense to me because rather than looking at the exact age of a pet, adopters will typically divide pets into categories like young, middle aged or old. I guess there is little difference between chance for a pet that is e. g. 6 months old and another one that is 8 months old. However, this decreased my score on the public LB significantly.







