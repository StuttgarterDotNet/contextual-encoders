# Contextual Encoders
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
![Python: >= 3.7](https://img.shields.io/badge/python-^3.7-blue)
[![Documentation Status](https://readthedocs.org/projects/contextual-encoders/badge/?version=latest)](https://contextual-encoders.readthedocs.io/en/latest/?badge=latest)

Contextual Encoders is a library of [scikit-learn](https://scikit-learn.org/stable) compatible contextual variable encoders.

The documentation can be found here: [ReadTheDocs](https://contextual-encoders.readthedocs.io).

This package uses Poetry ([documentation](https://python-poetry.org/docs/)).

## Basic Usage



## What are contextual variables?
Contextual variables are numerical or categorical variables, that underlie a certain context or relationship.
Examples are the days of the week, that have a hidden graph structure:

<p align="center">
<img src="https://raw.githubusercontent.com/StuttgarterDotNet/contextual-encoders/main/docs/_static/weekdays.svg" alt="">
</p>

When encoding these categorical variables with a simple encoding strategy such as <em>One-Hot-Encoding</em>, the hidden structure will be neglected.
However, when the context can be specified, this additional information can be put it into the learning procedure to increase the performance of the learning model.
This is, where Contextual Encoders come into place.

## Principle
The step of encoding contextual variables is split up into four sub-steps:

1) Define the context
2) Define the comparer
3) Calculate the distance matrix
4) Map the distance matrix to euclidean vectors

Depending on the techniques that uses the encoding, step 4 can be optional.
For example, [Agglomerative Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) 
techniques do not require euclidean vectors, they can use the distance matrix directly.

So far, build-in support for tree- and graph-based context is provided.

## Notice
The [Preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) module from scikit-learn offers multiple encoders for categorical variables.
These encoders use simple techniques to encode categorical variables into numerical variables.
Additionally, the [Category Encoders](http://contrib.scikit-learn.org/category_encoders) package offers more sophisticated encoders for the same purpose.
This package is meant to be used as an extension to the previous two packages in the cases, when the context of a numerical or categorical variable can be specified.

This project is currently in the developer stage.
