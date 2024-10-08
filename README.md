# Python-Scikit-Learn
<h1> Version Check </h1>
-scikit-learn 0.20.3
-Python 3.7
-Pandas 0.20.4
-NumPy 1.15
-Seaborn 0.8.1
-Matplotlib 2.2.0

# Course Outline
Intro to ML and scikit-learn
ML Workflow
Building simple ML Samples. Regression and Classification

# Intro
Types:
1) Classification: Is it a cat or a dog?
2) Regression:
3) Clustering: Logical data in a large dataset. 
4) Dimensionality reduction: Hugh number of atributes or features which don´t make sense. Give value to the data and use cases.

## 1) Classification
Before = Whale -> Rule-based classifier (Human Experts) -> Mammal
After = Breaths like a mammal, gives birth like a mammal -> ** ML-Based classifier (Corpus body of data) ** => Mammal

### ML-Based Classifier

- Training = Feed in a large corpu
- Prediction = Use it to classify new instances which it has not seen before

Training:
CORPUS -> 1° ML-based classifier -> Classification 
Classification -> Feedback Loss Function (IMPROVE MODEL PARAMETERS ) -> 2°ML-based Classifier

### ML-based binary classifier

Input: Feature vector (x variables)

ML-based classifier: Moves Like a fish, Looks like a fish ==> Predicted label ( FISH ) != Wrong result

Output: Label (y values or predictive values)

### Traditional ML Models
`Regresion Models: Linear, Lasso, Ridge, SVR`
`Classification models: Naive Bayes, SVMs, Decision trees, Random forests`
`Dimensionality Reduction: Manifold learning, factor analysis`
`Clustering: K-Means, DBSCAN, Spectral clustering`

### Starting ML Models. Basics
Have a fundamental algorithmic structure to solve problems
E.I Draw a line, create a curve

Model Parameters. Structures that learn from models.

1) Build a tree structure to classify instances
2) Fit a line or a curve on data to make predictions
3) Apply probabilities on input data to get output probabilities.

### Advanced Representation ML Models
Also used to solve classification, regression, clustering, and dimensionality reduction
BUT without experts. Learn significant features from the underlying data
<b>Deep learning models such as neural networks.<b>
#### What is a neural network?
Deep Learning: Algorithms that learn which features matter
Neural Networks: Most common of deep learning.
Neurons: Simple building blocks that actually "Learn"

Corpus -> Interconnected Layers of neurons -> ML-Based classfier
==Fish\Bunny -> Pixels | Edges | Corners | Object parts -> ML-Based classfier==

## SCIKIT_learn Ease of use
- Estimator API for consitent interface
- Estimators for all kind of models
- Create a model object
- Fit to training data
- Predict for new data
- Pipelines for complex operations

## SCIKIT_learn Comprehensive
- All common families of models supported
- Data pre-processing, cleaning, feature selection, and extracion
- Model validation and evaluation

$SCIKIT_learn Completeness$
-- Regression, classification, clustering, dinsionality reduction
-- Feature extraction and selection using statistical and dimensionality reduction
-- Data pre-processng
-- Data generation: Swiss rollsm S-Curves
-- Cross-Validation to evaluate models
-- Hyperparameter tuning

## SCIKIT_learn Efficiency
- Highly optimized implementations
- Built on SciPy, hence scikit prefix
- Iterporates with all common python libraries for data science

Common libraries:
~NumPy: Base n-dimansional array package
~Scipy: Fundamental library for scientific computing
~Matplotlib: Comprehensive 2D/3D plotting
 ~ Sympy: Symbolic math
 ~ Pandas: Data structured and analysis
