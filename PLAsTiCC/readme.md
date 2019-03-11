# Overview: Astronomical Classification

### Target: Multiclass classification
### Metric: Logloss (weighted multi-class logarithmic loss)
### Data: Large tabular and time series data (Unevenly sampled time series)
### Special feature: Small training set (~8000 samples) versus large testing set (~3.5 million samples) 

## 1st solution by Kyle Boone: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75033

**source code:** https://github.com/kboone/plasticc

**Solution overview:**

(1) Augmented the training set by degrading the well-observed lightcurves in the training set to match the properties of the test set. (**Very important for this competition**)

(2) Use Gaussian processes to predict the lightcurves.

(3) Measured 200 features on the raw data and Gaussian process predictions.

(4) Trained a single LGBM model with 5-fold cross-validation.

## 2nd solution by Mike & Silogram: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75059

**RNN source code:** https://www.kaggle.com/zerrxy/plasticc-rnn/

**Solution overview:**

(1) Hostgalspecz pseudo-labeling. One thing that worked for both the NN and LGB models was to build a separate model to predict the hostgalspecz values (using both the training set and the test set objects that have this value), and then using oof predictions for these values in our models.

(2) Ensemble. The final ensemble was a very shallow (max_depth=2) LGB model that was very effective.

**About the RNN model:**

* NN model is a simple one layer Bidirectional GRU (we used 80 - 160 units) with global max pooling. Then max pooling output is combined with meta data and followed by 3 additional dense layers and softmax activation.

* Input data for RNN consists of flux, flux_err, intervals between measurements, emitted wavelength and passband (with embedding layer). flux and fluxerr are divided by a value of (fluxmax - fluxmin) for each object.

* Meta data consists of hostgalphotoz, hostgalphotozerr, ddf, mwebv and log2 of flux range.

* When preparing data for RNN we converted all time and wavelength related features to not depend on red shift. Namely time and wavelength are divided by (hostgalphotoz + 1)

**Augmentation works well for this model. The following steps were used:**

* Drop 30% of measurements

* Modify red shift using normal distribution with sigma = hostgalphotozerr * 2/3. When changing red shift, all time and wavelength related features are also modified accordingly.

* Modify flux using normal distribution with sigma = fluxerr * 2/3


## 3rd solution by Major Tom:

### Part1 Solution: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75116

**CNN model:** https://www.kaggle.com/yuval6967/3rd-place-cnn

~~~
def build_model():
   input_timeseries = Input(shape=(num_samples, 6,),name='input_timeseries')
   input_timeseries0 = Input(shape=(num_samples, 6,),name='input_timeseries0')
   input_timeseriese = Input(shape=(num_samples, 6,),name='input_timeseriese')
   input_meta = Input(shape=(6,),name='input_meta')
   input_gal = Input(shape=(1,),name='input_gal')
   _series=concatenate([input_timeseries,input_timeseries0,input_timeseriese])
   x = Conv1D(256,8,padding='same',name='Conv1')(_series)
   x = BatchNormalization()(x)
   x = LeakyReLU(alpha=0.1)(x)
   x = Dropout(0.2)(x)
   x = Conv1D(256,5,padding='same',name='Conv2')(x)
   x = BatchNormalization()(x)
   x = LeakyReLU(alpha=0.1)(x)
   x = Dropout(0.2)(x)
   x = Conv1D(256,3,padding='same',name='Conv5')(x)
   x = BatchNormalization()(x)
   x = LeakyReLU(alpha=0.1)(x)
   x = GlobalMaxPooling1D()(x)
   x1 = Dense(16,activation='relu',name='dense0')(input_meta)
   x1 = Dense(32,activation='relu',name='dense1')(x1)
   xc = concatenate([x,x1],name='concat')
   x = Dense(256,activation='relu',name='features')(xc)
   x = Dense(real_targets.shape[0],name='bout')(x)
   x = MySwitch(galactic_targets.shape[0])([input_gal,x])
   out = Activation('softmax',name='out')(x)
   model=Model([input_timeseries,input_timeseries0, 
              input_timeseriese,input_meta,input_gal],out)
   return model
~~~

### Part2 Solution: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75131

**Source code: https://github.com/takashioya/plasticc**

- Use feATURE eXTRACTOR FOR tIME sERIES library: https://github.com/carpyncho/feets

- Ensemble: I used weighted average for ensemble. however, I gave different weights to each class, which gave us 0.006 improvement. **I calculated the weights using oof-predictions by hyperopt.**


- adversarial pseudo-labelling: **I selected the objects whose the prediction is very high and similar to train data and used them for training.** It didn't give a significant improvement for me, but I suppose it somewhat contributed to diversity in ensemble.

### Part3 Solution: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75222

- template fitting by sncosmo package https://www.kaggle.com/nyanpn/salt-2-feature-part-of-3rd-place-solution

- Pseudo Labelling: I also tried to use pseudo-labeling in the early stage of the competition. I found that using pseudo-label only in class90 gave me a big boost (0.005 ~ 0.03, depends on the model), but using all classes didn't work. I think it's related class99 (if class99 is similar to class52/62, pseudo-label in these classes contains a lot of false signals). class90+class42 gave a good result too, but its difference from class90-only was very small.


## 4th solution by Ahmet Erdem: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75011

**Source code:** https://github.com/aerdem4/kaggle-plasticc

- **Using normal values and log transformed values together on Neural Net:** Having both gives you the opportunity to do all four operations between the features (+, -, / *) because you can write log(xy) as log(x) + log(y). I guess hostgal_photoz was interacting with most of the features that I have.

**Log Ensemble:** Instead of averaging predictions, I have averaged the logarithm of the predictions because in the end we try to regress the log values. This worked better for me.

## 5th solution by SKZ lost in translation: 

https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75050

https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75040

**Partial source code:** https://github.com/jfpuget/Kaggle_PLAsTiCC

## 6th solution by Stefan Stefanov: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75061

a neural network consisting of 3 components:

- Meta Encoder - taking as input meta features hostgal photoz, hostgal photoz err, is galactic and summary stats for flux and flux_err as min/max/mean etc.

- Light Curve Encoder - bidirectional GRU taking as input grouped by day flux, flux_err, detected and time difference

- Spectroscopic Redshift Predictor : two fully-connected layers for predicting hostgal specz

Outputs of these 3 components are fed into two last fully-connected layers for predicting class probabilities.

## 8th solution by Jiwei Liu: https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75012

**Source code:** https://github.com/daxiongshu/Rapids_PLAsTiCC_2018

- Use cudf to replace pandas for this competition whenever I can. In general, it is at least 10x faster for csv reader and general groupby - aggregation with a single GPU than pandas on CPU. 

A key to successful non-linear stacking is to avoid overfitting. I have found two tricks for this:

- In level 1 base model, don't use the oof labels for early stopping. Instead, split the train data in that fold again to get its own validation set. This will degrade the level 1 model's score but it will help level 2 stacking.

- Make the 2nd level model simpler. For example, my 1st level lgb use depth of 7 and 3. In 2nd level lgb, the depth is always 1. The 1st level MLP has 4 layers but the 2nd level MLP only has 1 layer. And again, mysteriously, now it is my own TF MLP at 2nd level that is always better than the keras counterpart. In the end I just average the lgb/xgb and nn level 2 predictions with equal weights.

## 9th solution

https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75316

https://www.kaggle.com/meaninglesslives/a-slightly-better-nn-arch-and-some-tricks

## 11th solution

https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75174

## 12th solution

https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75237

## 13th solution

https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75134

## 14th solution

https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75054

https://github.com/btrotta/kaggle-plasticc

## 19th solution

https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75167

## 20th solution 

https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75262

## 21st solution

https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75156

https://www.kaggle.com/c/PLAsTiCC-2018/discussion/75140










