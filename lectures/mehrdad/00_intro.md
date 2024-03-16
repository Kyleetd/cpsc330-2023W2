**Types of Machine Learning**

- Supervised
  - the algorithm is trained on a labeled dataset, meaning the input data is paired with the correct output
  - Examples include linear regression, logistic regression, support vector machines (SVM), decision trees, and neural networks trained with labeled data.
- Unsupervised
  - algorithm is given a dataset without labeled responses. The goal is to find patterns or intrinsic structures in the data.

**_Lecture 2 - Terminology, Baselines, Decision Trees_**

- identify whether a given problem could be solved using supervised machine learning or not;

- differentiate between supervised and unsupervised machine learning;

In Supervised Learning, training data comprises a set of features (_X_) and their corresponding targets (_y_). We wish to find a **model function** _f_ that relates _X_ to _y_ then use that model function to predict the targets of new examples. Concerned with function approximation (i.e, finding a mapping from _X_ to _y_.)

In Unsupervised Learning, training data comprises a set of features (_X_) without any corresponding targets. Unsupervised learning can be used to group similar things together in _X_ or to provide a consice summary of the data. Concerned with precisely describing the data.

- explain machine learning terminology such as features, targets, predictions, training, and error;

Features: relevant characteristics to the problem. denoted as _X_.Inputs, predictors, explanatory variables, regressors, independent variables.

Target: The feature we want to redict. Denoted by _y_. Outputs, response variable, dependent variable, labels (if categorical).

Training: process of learning the mapping between features (_X_) and the target (_y_). Learning, fitting.

- differentiate between classification and regression problems;

**Classification**: predicting amount 2+ **discrete** classes. i.e., predicting presence of an illness, predicting if student got A+.

**Regression**: predicting a continuous value. i.e., predicting housing prices, student's grade.

- use `DummyClassifier` and `DummyRegressor` as baselines for machine learning problems;
- explain the `fit` and `predict` paradigm and use `score` method of ML models;

1. Read the data
2. Create _X_ and _y_
3. Create a classifier object
4. fit the classifier object
5. predict on new examples
6. score the model

`.fit()` a model on the training set only.

`.score` calls predict on _X_ and compares predictions with _y_ (true targets). Returns accuracy of classification. To get error, do 1 - classification_object.score(X, y)

```python
train_accuracy = model.score(X_train.values, y_train.values)
test_accuracy = model.score(X_test.values, y_test.values)
```

- broadly describe how decision tree prediction works;

Decision trees are models that make predictions by sequentially looking at features and checking whether they are above/below a threshhold.`max_depth` is a hyperparameter of `DecisionTreeClassifier`. `max_depth` controls model complexity.

When `fit` is called, a bunch of values get set, such as the fetaures to split on and split threshholds. There are called _parameters_.

The algorithm starts by looking at the entire dataset and finding the best feature to split the data into two groups. After the first split, each subset becomes its own smaller dataset. The algorithm then recursively applies the splitting process to each subset, creating a tree-like structure.

- use `DecisionTreeClassifier` and `DecisionTreeRegressor` to build decision trees using `scikit-learn`;

```python
model = DecisionTreeClassifier(max_depth = 1)
model.fit(X.values, y)
```

- visualize decision trees;
- explain the difference between parameters and hyperparameters;

Parameters are internal variables learned by the model during training to best fit the data, while hyperparameters are external configuration settings that govern the learning process and are set before training, such as the learning rate or the maximum tree depth in a DecisionTreeClassifier.

- explain the concept of decision boundaries;

Decision boundaries are the lines that separate different classes in classification. They are determined by the model based on the learned parameters and are used to make predictions on new data points by assigning them to the appropriate class based on which side of the boundary they fall.

- explain the relation between model complexity and decision boundaries.

Model complexity refers to the flexibility or capacity of a model to capture intricate patterns in the data. As model complexity increases, decision boundaries become more intricate and flexible, potentially capturing more complex relationships between features and classes. However, overly complex models can lead to overfitting, where decision boundaries fit the training data too closely and generalize poorly to unseen data.

**_Lecture 3 - Machine Learning Fundamentals_**

- explain how decision boundaries change with the `max_depth` hyperparameter;

The max_depth hyperparameter in decision tree algorithms controls the maximum depth of the tree, limiting the number of splits it can make. With a lower max_depth, decision boundaries tend to be simpler and more linear, as the tree is constrained to make fewer splits. Conversely, increasing max_depth allows the tree to create more complex decision boundaries, potentially capturing more intricate patterns in the data, which can lead to overfitting if not carefully tuned.

- explain the concept of generalization;

Generalization in machine learning refers to the ability of a model to perform well on unseen data, beyond the training set. A model that generalizes well can effectively capture underlying patterns in the data and make accurate predictions on new, unseen examples. Achieving good generalization involves finding the right balance between capturing complex patterns in the training data without memorizing it, avoiding overfitting, and ensuring that the model can adapt to new data without sacrificing performance. Cross-validation and hyperparameter tuning are common strategies to promote generalization in machine learning models.

- appropriately split a dataset into train and test sets using `train_test_split` function;

SHUFFLE DATA FIRST IN TRAIN_TEST_SPLIT.

```python
train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
X_train, y_train = train_df.drop(columns=["country"]), train_df
X_test, y_test= test_df.drop(columns=["country"]), train_df["country"]["country"]
```

- explain the difference between train, validation, test, and "deployment" data;

**Training Data**: This is the portion of the dataset used to train the model. The model learns from this data by adjusting its parameters to minimize the training error. `fit`, `score`, `predict`.

**Validation Data**: This is a separate portion of the dataset used to evaluate the performance of the model during training. It helps in tuning hyperparameters and preventing overfitting by providing an unbiased assessment of the model's performance on data not seen during training. Separate data for hyperparameter tuning. DATA WITH ACCESS TO TARGET VALUES. Unlike training data, we only use it for hyperparameter tuning (don't pass into `fit` function). When you break training data further it's called **validation split**. `score`, `predict`.

**Test Data**: This is another separate portion of the dataset used to evaluate the final performance of the trained model. It provides an independent assessment of how well the model generalizes to unseen data and gives an estimate of its real-world performance. DATA WITH ACCESS TO TARGET VALUES. Don't use for training nor hyperparameter tuning. Locked away until evaluation of model - we use it ONLY ONCE to evaluate bestperforming model on the validation set. `score` and `predict` ONCE.

**"Deployment" Data**: Deployment data refers to the data that the deployed model will encounter in the real-world scenario. DATA WITHOU ACCESS TO TARGET VALUES. `predict`.

- identify the difference between training error, validation error, and test error;

**training error**: The error calculated on the training dataset during the training phase. It measures how well the model fits the training data. A low training error indicates that the model has learned to represent the training data well.

**validation error**: The error or loss calculated on the validation dataset during the training phase. It provides an estimate of how well the model generalizes to unseen data. The goal is to minimize both the training error and the validation error simultaneously.

**test error**: The error or loss calculated on the test dataset after the model has been trained and validated. It measures the model's performance on new, unseen data and provides an estimate of its real-world performance. The test error helps to assess how well the model generalizes to data it has never seen before.

Usually: E<sub>train</sub> < E<sub>validation</sub> < E<sub>test</sub> < E<sub>deployment</sub>

- cross-validation

If dataset is small, you might end up with a small training and/or validation set. The split might be unlucky in that it doesn't align or represent the data. Cross-validation is the solution. Each fold gets a turn at being the validation set. `cross_validate` nor `cross_val_score` shuffles the data - that should be done prior to `train_test_split`. Each fold gives a score, take average of k scores.

- explain cross-validation and use `cross_val_score` and `cross_validate` to calculate cross-validation error;

```python
model = DecisionTreeClassifier(max_depth=4)
cv_scores = cross_val_score(model, X_train, y_train, cv=10)
cv_scores ## array of cv scores

mean_cv_score = np.mean(cv_scores)
std_cv_scores = np.std(cv_scores)
```

`cross_val_score` output is list of validation scores.

```python
scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
pd.DataFrame(scores) ## df of fit_time, score_time, test_score, and train_score for each of the 10 folds

pd.DataFrame(pd.DataFrame(scores).mean()) ## df of MEAN fit_time, score_time, test_score, and train_score
```

`cross_validate` evaluates how well model will generalize to unseen data. Note that the dataframe produced from cross_validate using X_train and y_train labels the validation score `test_score`. It should be called `val_score` since we are scoring on validation data, not test data.

- recognize overfitting and/or underfitting by looking at train and test scores;

Low train score + low test score = underfitting. High bias (the tendency to consistently predict the wrong thing) corresponds to underfit models.

High train score + low test score = overfitting. High variance (the tendency to learn random things irrespective to reality) corresponds to overfitting.

- explain why it is generally not possible to get a perfect test score (zero test error) on a supervised learning problem;
- describe the fundamental tradeoff between training score and the train-test gap;

**UNDERFITTING**
If your model is too simple (like the Dummy Classifier or Decision Tree w max_depth=1), it won't capture useful patterns in the training data. Both train and validation error will be high. This is underfitting: the gap between train and validation errors will be lower.

Underfit: E<sub>best</sub> < E<sub>train</sub> <= E<sub>validation</sub>

**OVERFITTING**
If your model is too complex (Decision Tree w max_depth=None), the model will learn unreliable patterns in order to classify every training example correct. Training error will be LOW but there will be a BIG GAP between training error and validation error. This is overfitting. Validation error does not necessarily decrease with training error. The mean train accuracy will be much higher than the mean cv-accuracy.

Overfit: E<sub>train</sub> < E<sub>best</sub> < E<sub>validation</sub>

Increasing complexity imrpoves the train score but DECREASES the cross-validation score. You must find the balance at depth ~ 5-6.

**_TRADE-OFF BETWEEN TRAIN SCORE AND TRAIN-TEST GAP_**:

As training error goes DOWN, the gap between the validaiton error and the training error goes UP.

- state the golden rule;

THE TEST DATA CANNOT INFLUENCE THE TRAINING PHASE IN ANY WAY.

- start to build a standard recipe for supervised learning: train/test split, hyperparameter tuning with cross-validation, test on test set.

1. Split data into Train and Test sets: X_train, y_train, X_test, y_test
2. Hyperparameter tuning using cross-validation in the train set (X_train and y_train)
3. Assess the best-performing model on test data (X_test and y_test) using **test error**. In general, pick model witht he minimum cross-validaiton error.
4. If test error is reasonable, deploy model on deployment (unseen) data. If the test error is compatable with the test error, this is a sign the model performs similarily on the train and test data. This increases confidence that model would perform similarily on deployment data.

Data splitting as a measn to approximate generalization error:

- train model on test split
- tune hyperparameters on validation split
- check generalization performance on test split

**_Lecture 5 - Preprocessing and sklearn pipelines_**

- explain motivation for preprocessing in supervised machine learning;

For example, kNN uses euclidian distance for classification. If features in the dataset have different ranges and values, variation amongst data in smaller-value columns will be dominated by columns with higher valeus. In other words, the euclindian distance is dominated by features with larger values. Thus, we must normalize the columns.

Use `scikis-learn`'s `StandardScaler` transformer. Normalizes by taking each value, subtracting the mean of the entire column, and dividng by the std of the column. Scaling is only needed for numeric data.

```python
scaler = StandardScaler()                   # create feature trasformer object
scaler.fit(X_train)                         # fitting the transformer on the train split
X_train_scaled = scaler.transform(X_train)  # transforming the train split
X_test_scaled = scaler.transform(X_test)    # transforming the test split
```

May also need to deal with missing data via **Imputation**. For categorical variables, use **One-hot encoding** (add c new binary columns where c = number fo unique categories in original column) or **Ordinal encoding**.

- identify when to implement feature transformations such as imputation, scaling, and one-hot encoding in a machine learning model development pipeline;

If some columns have NaN value, you cannot apply `fit` toa knn model. You must use `Simple Imputer` to impute missing values in categorical columns with the **most frequent** value. You can impute missing values in numeric columns with the **mean** or the **median**. Specifically, use mean unless you have outliers, then use median.

If you have numeric data and you are using kNN, scale (standardize) data. `StandardScaler` subtracts column mean and divides by column std for each value. After standardizing, the mean of each column will be 0 and the std will be 1 (fixed range).

```python
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)
X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

knn = KNeighboursRegressor()
knn.fit(X_train_scaled, y_train)
knn.score(X_train_scaled, y_train)
```

- use `sklearn` transformers for applying feature transformations on your dataset;

`sklearn` uses `fit` and `transform` paradigms for feature transformations. `fit` transformer to train split and then trasnform the train AND test split (apply same transformation on test split).

- discuss golden rule in the context of feature transformations;

During cross validation, within every fold the train and test data were all transformed from a transformer that was fit on the entire data. This means information about the test data is leaked during every fold. The solution is PIPELINING. Using a pipeline, `fit_transform` is only applied on train portion of each fold and `transform` is applied on the validation portion.

- use `sklearn.pipeline.Pipeline` and `sklearn.pipeline.make_pipeline` to build a preliminary machine learning pipeline.

```python
pipe = Pipeline(
  steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", KNeighborsRegressor()), ## last step is model
  ]
)
```

or

```python
pipe = make_pipeline(
  steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", KNeighborsRegressor()), ## last step is model
  ]
)

pipe.fit(X_train, y_train) ## Pass X_train, not imputed or scaled data
```

Once you have pipeline with an estimator as the last step, you can call `fit`, `predict`, and `score` on it.

**_Lecture 6 - sklearn Column Transformer and Test Features_**

- use `ColumnTransformer` to build all our transformations together into one object and use it with `sklearn` pipelines;

```python
ct = make_column_transformer(
  (StandardScaler(), numeric_feats),
  ("passthrough", passthrough_feats),
  (OneHotEncoder(), categorical_feats),
  ("drop", drop_feats),
)

pipe = make_pipeline(ct, SVC())
pipe.fit(X, y)
pipe.predict(X)
```

```python
ct = make_column_transformer(
  (
    make_pipeline(SimpleImputer(), StandardScaler()), ## Impute BEFORE scaling in mumeric columns
    numeric_feats
  ),
  ("passthrough", passthrough_feats),
  (OneHotEncoder(handle_unknown="ignore"), categorical_feats), ## handle_unknown creates a row with all zeros
  ("drop", drop_feats),
)

pipe = make_pipeline(ct, SVC())
pipe.fit(X, y)
pipe.predict(X)

pipe.cross_validate(pipe, X, y, return_train_score=True)
```

If we know all the categories beforehand, it MIGHT BE OKAY to specify them to OneHotEncoder to avoid ignoring unseen categories due to train-test split. This is technically breaking the golden rule, but is okay in cases where the categories are provinces in Canada for majors taught at UBC (fixed number of categories - won't differ in the deployment data).

- define `ColumnTransformer` where transformers contain more than one steps;

Most times some features are categorical, some are continuous, some are binary, and some are ordinal. Thus, we must apply different transformations on different columns - use `ColumnTransformer`.

- explain `handle_unknown="ignore"` hyperparameter of `scikit-learn`'s `OneHotEncoder`;
- explain `drop="if_binary"` argument of `OneHotEncoder`;

If you have a binary column, it's wasteful to split into 2 columns.Use drop-"if_binary" argument in OneHotEncoder to only create 1 column in such scenario.

- identify when it's appropriate to apply ordinal encoding vs one-hot encoding;
- explain strategies to deal with categorical variables with too many categories;

for One-hot encoding, use handle_unknown="ignore": if the encoder encounters a category in the test data that was not present in the training data, it will ignore that category during transformation and represent it with an all-zero vector.

- explain why text data needs a different treatment than categorical variables;

The feature is neither categorical nor ordinal (and clearly not numerical). Cannot use one-hot encoding or ordinal encoding since we do not have a fixed number of categories (each "category" is likely to only appear once and is thuss meaningless).

Natural language processing (NLP) can facilitate representing text data as a fixed number of features from which a model can find patterns.

- use `scikit-learn`'s `CountVectorizer` to encode text data;

`CountVectorizer`: each row is text. Each column is a word from the training data. Cells represent how often a word occurs the text.

```python
vec = CountVectorizer()
X_counts = vec.fit_transform(df["sms"]) ## pass a series to fit_transform
bow_df = pd.DataFrame(
  X_counter.toarray(), columns=vec.get_feature_names_out(), index=df["sms"]
)
```

You must define separate `CountVectorizer` transformers for EACH TEXT COLUMN unlike other transformers.

- explain different hyperparameters of `CountVectorizer`.

`binary`: whether to use presence/absense feature values or counter. `binary=True` only consideres the presence or absence of words instead of word counts.
`max_features`: only consider top `max_features` ordered by frequency. Controls the size of X (number of features).

- incorporate text features in a machine learning pipeline

```python
pipe = make_pipeline(CountVectorizer(), SVC())
pipe.fit(df["sms"], df["target"])
pipe.predict(df["sms"].tolist())
```

In scikit-learn's CountVectorizer, when it encounters a word in the test split that wasn't in the training data, it simply ignores that word.

**_Lecture 7 - Linear Models_**

- Explain the general intuition behind linear models;

Make a prediction using a linear function

1. Linear Regression (use Ridge instead of Linear Regression. Ridge can be used for datasets with nultiple features.). For regression.
2. Logistic Regression. For classification.
3. Linear SVM

- Explain how `predict` works for linear regression;

feature value: $x_1$\
coefficient / slope: $w_1$\
intercept: $b$

$$\hat{y}=w_1x_1 + b$$

i.e., if you have a 2-dimensional problem (2 features), the model will learn 3 parameters: one for each feature (weight) and the bian term. ONE COEFFICIENT PER FEATURE. Two important aspects of corefficients: sign and magnitiude. Larger magnitude = larger impact on prediction. Thus, if you do not scale the data features w smaller magnitudes are going to get larger coefficient magnitudes and features w large magnitudes are going to get coefficients w smaller magnitudes.

However, scaling makes values harder to interpret for humans.

High coefficient for a given feature moves prediction toward +1 class.
Low/negative coefficient for a feature moves prediciton toward -1 class.
If coefficient == 0, it's not useful in making prediction.

- Use `scikit-learn`'s `Ridge` model;

```python
pipe = make_pipeline(StandardScaler(), Ridge())
scores = cross_validate(pipe, X_train, y_train, return_train_score=True)
pd.DataFrame(scores)
```

- Demonstrate how the `alpha` hyperparameter of `Ridge` is related to the fundamental tradeoff;

Higher alpha = underfit (simpler)
Lower alpha = overfit (complex)

Alpha = 0 is the same as vanilla Linear Regression

- Explain the difference between linear regression and logistic regression;

Logistic regression is a linear model for **classification**. Learns weights for each feature and the bias FROM THE TRAINING DATA. Applies a threshhold to the raw output to determine if class if positive or negative.

- Use `scikit-learn`'s `LogisticRegression` model and `predict_proba` to get probability scores

```python
lr = LogisticRegressor()
scores = cross_validate(lr, X_train, y_train, return_train_score=True)
```

`LogisticRegression` hyperparameter **C**:\
Higher `C` = overfit\
Lower `C` = underfit

- Explain the advantages of getting probability scores instead of hard predictions during classification. Output is the probability of each class. Sum of probabilities is 1 for each observation.

`predict_proba` is a soft prediction, i.e., how confident the model is with a given prediction

- Broadly describe linear SVMs

Not the default SVM, but you can add `kernel="linear"` arg to SVM to create a linear SVM. `predict` method of linear SVM and logistic regression works the same: get a `_coef` for each fteaure and `intercept_` using a Linear SVM model. `fit` for linear SVM and logistic regression are different (so coefficients will be diff).

- Explain how can you interpret model predictions using coefficients learned by a linear model;
- Explain the advantages and limitations of linear classifiers.

Pros: Fast to train and predict. Scale to large datasets and works w sparse data. Easy to understand and interpret results. Performs well w large number of features.

Limitations: Only works if data is liearly separable (can draw a hyperplane between datapoints).

**_Lecture 8 - Hyperparameter Optimazation and Optimization Bias_**

- explain the need for hyperparameter optimization

Improve model's generalization performance. Picking good hyperparameters can help avoid underfitting or overfitting.searches for best hyperparameter values.

`GridSearchCV` searches for best hyperparameters. Can call `fit`, `predict`, and `score` on it. `GridSearchCV` finds best hyperparameters, then fits a new model on whole training set with these best parameters. All you need to do is `.score(X_test, y_test)`

- carry out hyperparameter optimization using `sklearn`'s `GridSearchCV` and `RandomizedSearchCV`

```python
pipe_svm = make_pipeline(preprocessor, SVC())

param_grid = {
    "columntransformer__countvectorizer__max_features": [100, 200, 400, 800, 1000, 2000],
    "svc__gamma": [0.001, 0.01, 0.1, 1.0, 10, 100],
    "svc__C": [0.001, 0.01, 0.1, 1.0, 10, 100],
}

# Create a grid search object
gs = GridSearchCV(pipe_svm,
                  param_grid = param_grid,
                  n_jobs=-1,
                  return_train_score=True
                 )

gs.fit(X_train, y_train)
```

- explain different hyperparameters of `GridSearchCV`

`n_jobs`: run in parallel

`n_iter`: only for Randomized Search. Only choose `n_iter` combinations randomly. Larger `n_iter` = more time. Each of the `n_iter` combos will have cross validation called on it.

- explain the importance of selecting a good range for the values.
- explain optimization bias

If our dataset is small and validation set is hit too many times, we suffer from optimzation bias (overfitting the validation set).

i.e., trying many different Decision Trees, we could chance accross one treee with low training error by chance. This tree would no perform weel on other data. Larger training set decreases this chance.

- identify and reason when to trust and not trust reported accuracies

If test score is much lower than cv score, this indicates optimization bias. This is why we need a test set.

**_Lecture 9 - Classification Metrics_**

`.score` by default returns accuracy which is
$$accuracy = \frac{correct\ predictions}{total\ examples}$$

This is only good when you have balanced data. Accuracy is misleading w imbalanced data.

3 other metrics based on confusion matrix:

1. recall: (# correctly predicted positives) / (# total positives)
   $$recall = \frac{TP}{TP+FN}$$

2. precision: (# correctly predicted positives) / (# positive predictions)
   $$precision = \frac{TP}{TP+FP}$$

3. f1-score: combines precision and recall. F1 is for a GIVEN THRESH-HOLD.

$$ f1 = 2 \times \frac{ precision \times recall}{precision + recall}$$

If the model makes more positive predictions, generally the recall goes up but the precision goes down (& vice-versa).

If you want to detect fraud and achieve at least 75% accuracy, then you can set the thresh-hold for fraud to be lower (i.e, predict_proba > 30% = fraud) - the model will predict fraud more and capture more of the total number of positives. (TO INCREASE RECALL, MAKE THRESH-HOLD EASIER). But there will more likely be false positives = DOWN PRECISION.

Decrease thresh-hold: Recall might improve. Precision might go down.
Increasing thresh-hold: Precision is likely to go up. Recall might go down.

Overall, we want to find maximal point on PR curse (closest to (1, 1)). We can do this by maximizing the AP (average precision) score (closest to 1). AP score is summary ACROSS THRESH-HOLDS. AP measures the quality of `predict_proba`

`class_weight = "balanced"`: For imbalanced data, increasing the weight of the smaller class will cause it to be predicted more (but will increase false predictions of it too).

**_Lecture 10 - Regression Metrics_**

- Carry out feature transformations on somewhat complicated dataset.

Note: not all numeric-looking columns are actually numerical; i.e., type of dwelling in a housing dataset can be represented as numbers. Don't blindly trust automated data-describing tools:

```python
numeric_looking_columns = X_train.select_dtypes(include=np.number).columns.tolist()
```

Ordinal encoding:

```python
ordinal_features_oth = [
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Functional",
    "Fence",
]
ordering_ordinal_oth = [
    ["NA", "No", "Mn", "Av", "Gd"],
    ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"],
]
```

- Visualize transformed features as a dataframe.
- Use `Ridge` and `RidgeCV`.

Use Ridge instead of Linear Regression in this course. Ridge uses hyperparameter `alpha`

```python
lr = make_pipeline(preprocessor, Ridge())
lr.fit(X_train, y_train);
lr_preds = lr.predict(X_test)

print("Smallest coefficient: ", lr.named_steps["ridge"].coef_.min())
print("Largest coefficient:", lr.named_steps["ridge"].coef_.max())

## Cross Validation with Ridge
pd.DataFrame(cross_validate(lr_pipe, X_train, y_train, cv=10, return_train_score=True))
```

Because it's so common to want to tune `alpha` with `Ridge`, sklearn provides a class called `RidgeCV`, which automatically tunes `alpha` based on cross-validation.

```python
alphas = 10.0 ** np.arange(-6, 6, 1)
ridgecv_pipe = make_pipeline(preprocessor, RidgeCV(alphas=alphas, cv=10))
ridgecv_pipe.fit(X_train, y_train);

best_alpha = ridgecv_pipe.named_steps["ridgecv"].alpha_

ridge_tuned = make_pipeline(preprocessor, Ridge(alpha=best_alpha)) ## use best alpha
ridge_tuned.fit(X_train, y_train)
ridge_preds = ridge_tuned.predict(X_test)
ridge_preds[:10] ## examine tuned model
```

- Explain how `alpha` hyperparameter of `Ridge` relates to the fundamental tradeoff.

Higher values of `alpha` means a more restricted model. The values of coefficients are likely to be smaller for higher values of `alpha` compared to lower values of alpha.

- Explain the effect of `alpha` on the magnitude of the learned coefficients.

General intuition: **larger `alpha` leads to smaller coefficients**.
**Smaller coefficients** mean the predictions are **less sensitive to changes in the data**. Hence **less chance of overfitting**. Too small of coefficients tends to push to underfitting.

Smaller `alpha` leads to bigger coefficients. With the best alpha found by the grid search, the coefficients are somewhere in between.

```python
param_grid = {"ridge__alpha": 10.0 ** np.arange(-5, 5, 1)}

pipe_ridge = make_pipeline(preprocessor, Ridge())

search = GridSearchCV(pipe_ridge, param_grid, return_train_score=True, n_jobs=-1)
search.fit(X_train, y_train)
train_scores = search.cv_results_["mean_train_score"]
cv_scores = search.cv_results_["mean_test_score"]
```

- Examine coefficients of transformed features.
- Appropriately select a scoring metric given a regression problem.
- Interpret and communicate the meanings of different scoring metrics on regression problems.
  - MSE, RMSE, $R^2$, MAPE

In regression, you can't just check for equality of classes like in the case of classification. We need a score that reflects how right/wrong a prediciton is.

**1. Mean Squared Error (MSE)**

```python
preds = ridge_tuned.predict(X_train)
np.mean((y_train - preds) ** 2)
```

Perfect predictions would have MSE=0. Downside: MSE units is the unit of the prediction _squared_. This makes it hard to interpret.

**2. Root Mean Squared Error (RMSE)**

```python
preds = ridge_tuned.predict(X_train)
np.mean((y_train - preds) ** 2)
```

RMSE is a more relatable metric than MSE (bc is has same units as target).

**3. $R^2$**

$$R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y_i})^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

$R^2$ measures the proportion of variability in $y$ that can be explained using $X$. The denominator measures the total variance in $y$. (The amount of variability that is left unexplained after performing regression).

NOTE: $\hat{y}$ is prediction. $y$ is Ground Truth. Sum of squared differences between $\hat{y}$ and $y$. $R^2$ = 0 = same as dummy. Negative means worse than dummy. Greater than 0 = better then dummy. Perfect if $R^2$ = 1. The maximum is 1 for perfect predictions. Negative values are very bad: "worse than DummyRegressor" (very bad). Independent of the scale of $y$. So the max is 1.

**4. MAPE**

Percent error: positive (predict too high) and negative (predict too low).

```python
np.abs(percent_errors) ## Get rid of negatives
```

And, like MSE, we can take the average over examples. This is called mean absolute percent error (MAPE).

```python
def my_mape(true, pred):
    return np.mean(np.abs((pred - true) / true))
```

i.e., if it returns 0.1, on average, we have around 10% error.

- Apply log-transform on the target values in a regression problem with `TransformedTargetRegressor`.

Does `.fit()` know we care about MAPE? No, it doesn't. Why are we minimizing MSE (or something similar) if we care about MAPE?? When minimizing MSE, the **expensive houses** will _dominate_ because they have the **biggest error**. Which is better for RMSE?

How can we get `.fit()` to think about MAPE? A common practice which tends to work is **log transforming the targets**. That is, transform $y\rightarrow \log(y)$.

When you have prices or count data, the target values are skewed. A common trick in such cases is applying a log transform on the target column to make it more normal and less skewed.

```python
ttr = TransformedTargetRegressor(
    Ridge(alpha=best_alpha), func=np.log1p, inverse_func=np.expm1
) # transformer for log transforming the target
ttr_pipe = make_pipeline(preprocessor, ttr)
```
