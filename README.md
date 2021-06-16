# Contextual Encoders
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
![Python: >= 3.7](https://img.shields.io/badge/python-^3.7-blue)
[![Documentation Status](https://readthedocs.org/projects/contextual-encoders/badge/?version=latest)](https://contextual-encoders.readthedocs.io/en/latest/?badge=latest)
[![Python Tests](https://github.com/StuttgarterDotNet/contextual-encoders/actions/workflows/python.yml/badge.svg?branch=main)](https://github.com/StuttgarterDotNet/contextual-encoders/actions/workflows/python.yml)
![PyPi](https://img.shields.io/pypi/v/contextual-encoders?color=blue)

Contextual Encoders is a library of [scikit-learn](https://scikit-learn.org/stable) compatible contextual variable encoders.

The documentation can be found here: [ReadTheDocs](https://contextual-encoders.readthedocs.io).

This package uses Poetry ([documentation](https://python-poetry.org/docs/)).

## Installation
The library can be installed with `pip`

> pip install contextual-encoders

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
2) Define the measure
3) Calculate the (dis-) similarity matrix
4) Map the distance matrix to euclidean vectors

Setp 4. is optional and depends on the ML technique that uses the encoding.
For example, [Agglomerative Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) 
techniques do not require euclidean vectors, they can use a dissimilarity matrix directly.

## Basic Usage

The code below demonstrates the basic usage of the library.
Here, a simple dataset with 10 features is used.

```python
from contextual_encoders import ContextualEncoder, GraphContext, PathLengthMeasure
import numpy as np


# Create a sample dataset
x = np.array(["Fri", "Tue", "Fri", "Sat", "Mon", "Tue", "Wed", "Tue", "Fri", "Fri"])

# Step 1: Define the context
day = GraphContext("day")
day.add_concept("Mon", "Tue")
day.add_concept("Tue", "Wed")
day.add_concept("Wed", "Thur")
day.add_concept("Thur", "Fri")
day.add_concept("Fri", "Sat")
day.add_concept("Sat", "Sun")
day.add_concept("Sun", "Mon")

# Step 2: Define the measure
day_measure = PathLengthMeasure(day)

# Step 3+4: Calculate (Dis-) similarity Matrix
#           and map to euclidean vectors
encoder = ContextualEncoder(day_measure)
encoded_data = encoder.transform(x)

similarity_matrix = encoder.get_similarity_matrix()
dissimilarity_matrix = encoder.get_dissimilarity_matrix()
```

The output of the code is visualized below.
The graph-based structure can be clearly seen when the euclidean data points are plotted.
Note, that only five points can be seen, because the days "Thur" and "Sun" are missing in the dataset.

Similarity Matrix          |  Dissimilarity Matrix     |  Euclidean Data Points
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/StuttgarterDotNet/contextual-encoders/blob/main/docs/_static/readme_example_similarity_matrix.png?raw=true)  |  ![](https://github.com/StuttgarterDotNet/contextual-encoders/blob/main/docs/_static/readme_example_dissimilarity_matrix.png?raw=true)  | ![](https://github.com/StuttgarterDotNet/contextual-encoders/blob/main/docs/_static/readme_example_euclidean_data_points.png?raw=true)

More complicated examples can be found in the [documentation](https://contextual-encoders.readthedocs.io/en/latest/examples.html).

## Notice
The [Preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) module from scikit-learn offers multiple encoders for categorical variables.
These encoders use simple techniques to encode categorical variables into numerical variables.
Additionally, the [Category Encoders](http://contrib.scikit-learn.org/category_encoders) package offers more sophisticated encoders for the same purpose.
This package is meant to be used as an extension to the previous two packages in the cases, when the context of a numerical or categorical variable can be specified.

This project is currently in the developer stage.
