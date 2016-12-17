import graphlab
import json
import numpy as np
from math import sqrt
products = graphlab.SFrame('./data/amazon_baby_subset.gl/')
mtcars = graphlab.SFrame('./data/mtcars.csv')


with open('./data/important_words.json', 'r') as f: # Reads the list of most frequent words
    important_words = json.load(f)
important_words = [str(s) for s in important_words]

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

def get_numpy_data(data_sframe, features, label):
    data_sframe['intercept'] = 1
    features = ['intercept'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()
    return(feature_matrix, label_array)


def predict_probability(feature_matrix, coefficients):
    scores = np.dot(feature_matrix, coefficients)
    predictions = map(lambda x: 1 / (1 + np.exp(- x)), scores)
    return predictions


def feature_derivative(errors, feature):
    derivative = np.dot(errors, feature)
    return derivative


def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment == +1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))

    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]

    lp = np.sum((indicator - 1) * scores - logexp)
    return lp

def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients)
    for itr in xrange(max_iter):
        predictions = predict_probability(feature_matrix, coefficients)
        indicator = (sentiment == +1)
        errors = indicator - predictions
        for j in xrange(len(coefficients)):  # loop over each coefficient
            derivative = feature_derivative(errors, feature_matrix[:, j])
            coefficients[j] += step_size * derivative

        # Checking whether log likelihood is increasing
       # if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        #        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
         #   lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
          #  print 'iteration %*d: log likelihood of observed labels = %.8f' % \
           #       (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients

dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])
dummy_sentiment = np.array([-1, 1])
print(predict_probability(dummy_feature_matrix, dummy_coefficients))


