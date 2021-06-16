Examples
===============================================

In order to give you an easy introduction into the Contextual-Encoders library,
examples for the major applications of the library are provided in the following.

Single Attribute
+++++++++++++++++++++++++++++++++++++++++++++++

Within this example we want to describe the basic usage of the library.
As an example, we create a dataset consisting of a single attribute (column).
This attribute stores the days of the week. Hence, it is a categorical variable,
that has an underlying context - the cyclic structure of weekdays:

.. image:: https://raw.githubusercontent.com/StuttgarterDotNet/contextual-encoders/main/docs/_static/weekdays.svg
    :align: center

Let's start with importing all the types that we need for the example:

.. code:: python

    from contextual_encoders import ContextualEncoder, GraphContext, PathLengthMeasure
    import numpy as np

and create a dataset consisting of 10 features:

.. code:: python

    # Create a simple dataset
    x = np.array(
        [
            "Fri",
            "Tue",
            "Fri",
            "Sat",
            "Mon",
            "Tue",
            "Wed",
            "Tue",
            "Fri",
            "Fri"
        ]
    )

Next, we create the context. We can use the :class:`.GraphContext` type for that:

.. code:: python

    # Define the context
    day = GraphContext("day")
    day.add_concept("Mon", "Tue")
    day.add_concept("Tue", "Wed")
    day.add_concept("Wed", "Thur")
    day.add_concept("Thur", "Fri")
    day.add_concept("Fri", "Sat")
    day.add_concept("Sat", "Sun")
    day.add_concept("Sun", "Mon")

Having the context, we can then take a Measure and but the context into it.
For graph-based context we can use the :class:`.PathLengthMeasure` for example:

.. code:: python

    # Define the measure
    day_measure = PathLengthMeasure(day)

Now we are ready to encode the dataset. The easiest way is to just use the :class:`.ContextualEncoder` interface:

.. code:: python

    # Perform the encoding
    encoder = ContextualEncoder(day_measure)
    encoded_data = encoder.transform(x)

We can print the data and see what we get:

.. image:: https://github.com/StuttgarterDotNet/contextual-encoders/blob/main/docs/_static/readme_example_euclidean_data_points.png?raw=true
   :align: center

The graph-based structure can be clearly seen.
Note, that only five points can be seen, because the days "Thur" and "Sun" are missing in the dataset.


Beside the data points, also the similarity and dissimilarity matrices can be accessed:

.. code:: python

    similarity_matrix = encoder.get_similarity_matrix()
    dissimilarity_matrix = encoder.get_dissimilarity_matrix()

The matrices are visualized below:

|sim| |dissim|

.. |sim| image:: https://github.com/StuttgarterDotNet/contextual-encoders/blob/main/docs/_static/readme_example_similarity_matrix.png?raw=true
    :width: 49%
.. |dissim| image:: https://github.com/StuttgarterDotNet/contextual-encoders/blob/main/docs/_static/readme_example_dissimilarity_matrix.png?raw=true
    :width: 49%

Multiple Attributes
+++++++++++++++++++++++++++++++++++++++++++++++

The Contextual-Encoders library also allow to encode multiple attributes, i.e. multiple columns at once.
In this example, we simply add another column, named *Job* to the dataset. We can replace our
dataset from the example before with:

.. code:: python

    # Create a simple dataset with two columns
    x = np.array(
        [
            ["Fri", "Teacher"],
            ["Tue", "Student"],
            ["Fri", "Safety"],
            ["Sat", "Pilot"],
            ["Mon", "Police Man"],
            ["Tue", "Pilot"],
            ["Wed", "Student"],
            ["Tue", "Student"],
            ["Fri", "Education"],
            ["Fri", "Teacher"],
        ]
    )

Note, that we now have the jobs ``Teacher``, ``Student``, ``Pilot`` and ``Police Man``,
but also the concepts ``Safety`` and ``Education``. This can happen in situations,
where a specific job description does not fit, but a relationship to the same sector
can still be defined. An example for this could be a Safety Guard that is not a
Police Man but still within the same sector.

For this variable, a tree-based context fits well. We can create it using the :class:`.TreeContext` type:

.. code:: python

    from contextual_encoders import TreeContext

    ...

    # Define the context for jobs
    job = TreeContext("job")
    job.add_concept("Education")
    job.add_concept("Transportation")
    job.add_concept("Safety")
    job.add_concept("Teacher", "Education")
    job.add_concept("Student", "Education")
    job.add_concept("Police Man", "Safety")
    job.add_concept("Pilot", "Transportation")

A visualization of the tree context can be seen below:

.. image:: https://raw.githubusercontent.com/StuttgarterDotNet/contextual-encoders/main/docs/_static/jobs.svg
    :align: center

After creating the context, we can use a tree-based Measure, like the :class:`.WuPalmer` similarity measure:

.. code:: python

    from contextual_encoders import WuPalmer

    ...

    # Define the measure
    job_measure = WuPalmer(job)

and perform the encoding:

.. code:: python

    # Perform the encoding
    encoder = ContextualEncoder([day_measure, job_measure])
    encoded_data = encoder.transform(x)

The output is visualized below:

|sim2| |dissim2| |datapoints2|

.. |sim2| image:: https://github.com/StuttgarterDotNet/contextual-encoders/blob/main/docs/_static/multiple_attribute_example_similarity_matrix.png?raw=true
    :width: 32%
.. |dissim2| image:: https://github.com/StuttgarterDotNet/contextual-encoders/blob/main/docs/_static/multiple_attribute_example_dissimilarity_matrix.png?raw=true
    :width: 32%
.. |datapoints2| image:: https://github.com/StuttgarterDotNet/contextual-encoders/blob/main/docs/_static/multiple_attribute_example_euclidean_data_points.png?raw=true
    :width: 32%

Multiple Forms of Attributes
+++++++++++++++++++++++++++++++++++++++++++++++

Beside having multiple attributes, each attribute can potentially consist of more than one value itself.
One example is a person with two jobs from the dataset before.
Thus, a dataset could potentially look like:

.. code:: python

    # Create a simple dataset with two columns
    # and multiple forms
    x = np.array(
        [
            ["Fri", "Teacher"],
            ["Tue", "Student,Police Man"],
            ["Fri", "Safety"],
            ["Sat", "Pilot"],
            ["Mon", "Police Man"],
            ["Tue", "Pilot,Student"],
            ["Wed", "Student"],
            ["Tue", "Student"],
            ["Fri", "Education"],
            ["Fri", "Teacher,Pilot"]
        ]
    )

The optional parameters of the :class:`.ContextualEncoder` allows us to define such a behaviour.
Here, the forms are separated using the ``separator_token`` parameter.
When it comes to the calculation, it depends on the specified :class:`.Measure`,
if multiple forms can be handled. If a Measure has the property ``multiple_values``
with ``True``, it can directly compare attributes with multiple forms. If the
property is ``False``, a :class:`.Gatherer` is used to combine the pairwise
attribute form comparison values. Which Gatherer is used, can be specified
in the :class:`.ContextualEncoder`. In the following, we use the :class:`.SymMaxMeanGatherer`:

.. code:: python

    # Perform the encoding
    encoder = ContextualEncoder([day_measure, job_measure], gatherers="smm")
    encoded_data = encoder.transform(x)

Using Module Parameters
+++++++++++++++++++++++++++++++++++++++++++++++

All modules such as *Inverters* and *Reducers* have parameters. One example is the
parameter ``n_components`` of any :class:`.Reducer` type. It specifies the dimension
of the data points when converting a similarity or dissimilarity matrix to vectors.
Parameters are always set within the ``__init__`` function of the module.
In the following, we use the :class:`.MultidimensionalScalingReducer` and set the
``n_components`` parameter to 1 (instead of 2, which is the default):

.. code:: python

    from contextual_encoders import MultidimensionalScalingReducer

    ...

    # Perform the encoding
    encoder = ContextualEncoder(
        [day_measure, job_measure],
        reducer=MultidimensionalScalingReducer(n_components=1)
    )
    encoded_data = encoder.transform(x)

Defining own Modules
+++++++++++++++++++++++++++++++++++++++++++++++

The Contextual-Encoders library is designed to allow an easy extension.
This is needed, because the context can change from dataset to dataset.
The following types serve as base classes and every derived class of them
can be used to initialize the :class:`.ContextualEncoder` with custom behaviour.

- :class:`.Aggregator`: Used to combine similarity and dissimilarity matrices of multiple attributes to a single one.
- :class:`.Context`: Defines a general context type.
- :class:`.GraphBasedContext`: A context optimized for storing graphs.
- :class:`.Gatherer`: Used for combining form comparison values to an attribute comparison value.
- :class:`.Inverter`: Calculates similarity values given dissimilarity values and vice versa.
- :class:`.SimilarityMeasure`: Defines a general measure for calculating similarity comparison values based on a context.
- :class:`.DissimilarityMeasure`: Defines a general measure for calculating dissimilarity comparison values based on a context.
- :class:`.SimilarityMatrixReducer`: Used for converting similarity matrices to data points.
- :class:`.DissimilarityMatrixReducer`: Used for converting dissimilarity matrices to data points.

Given we have implemented custom types of these base classes, we can use them as follows:

.. code:: python

    # Perform the encoding
    encoder = ContextualEncoder(
        [day_measure, job_measure],
        gatherers=[MyGatherer1, MyGatherer2],
        aggregator=MyAggregator,
        inverters=[MyInverter1, MyInverter2],
        reducer=MyReducer)

    encoded_data = encoder.transform(x)

