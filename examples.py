import seaborn as sns
from contextual_encoders import (
    TreeContext,
    WuPalmer,
    ContextualEncoder,
    GraphContext,
    PathLengthMeasure,
    MultidimensionalScalingReducer,
)
import matplotlib.pyplot as plt
import numpy as np


# noinspection DuplicatedCode
def readme_example():
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

    # Step 3+4: Calculate Dissimilarity Matrix
    #           and map to euclidean vectors
    encoder = ContextualEncoder(day_measure)
    encoded_data = encoder.transform(x)

    print_data_points(encoded_data)

    print_matrix(encoder.get_similarity_matrix(), "Similarity Matrix")
    print_matrix(encoder.get_dissimilarity_matrix(), "Dissimilarity Matrix")

    return


# noinspection DuplicatedCode
def multiple_attributes_example():
    # Create a sample dataset with two columns
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
            ["Fri", "Teacher,Pilot"],
        ]
    )

    # Step 1: Define the context
    day = GraphContext("day")
    day.add_concept("Mon", "Tue")
    day.add_concept("Tue", "Wed")
    day.add_concept("Wed", "Thur")
    day.add_concept("Thur", "Fri")
    day.add_concept("Fri", "Sat")
    day.add_concept("Sat", "Sun")
    day.add_concept("Sun", "Mon")

    job = TreeContext("job")
    job.add_concept("Education")
    job.add_concept("Transportation")
    job.add_concept("Safety")
    job.add_concept("Teacher", "Education")
    job.add_concept("Student", "Education")
    job.add_concept("Police Man", "Safety")
    job.add_concept("Pilot", "Transportation")

    # Step 2: Define the measure
    day_measure = PathLengthMeasure(day)
    job_measure = WuPalmer(job)

    # Step 3+4: Calculate Dissimilarity Matrix
    #           and map to euclidean vectors
    encoder = ContextualEncoder(
        [day_measure, job_measure], reducer=MultidimensionalScalingReducer(n_components=1)
    )
    encoded_data = encoder.transform(x)

    print_data_points(encoded_data)

    print_matrix(encoder.get_similarity_matrix(), "Similarity Matrix")
    print_matrix(encoder.get_dissimilarity_matrix(), "Dissimilarity Matrix")

    return


# noinspection DuplicatedCode
def simple_example():
    # Load the tips dataset
    tips = sns.load_dataset("tips")

    # Select a subset of categorical variables
    data = tips["day"]

    subset = data.sample(frac=1, random_state=1).reset_index(drop=True).head(10)

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

    # Step 3+4: Calculate Distance Matrix
    #           and map to euclidean vectors
    encoder = ContextualEncoder(day_measure, inverters="sqrt")
    encoded_data = encoder.transform(subset)

    print_data_points(encoded_data)

    print_matrix(encoder.get_similarity_matrix(), "Similarities")
    print_matrix(encoder.get_dissimilarity_matrix(), "Dissimilarity")

    return


# noinspection DuplicatedCode
def advanced_example():
    # Load the tips dataset
    tips = sns.load_dataset("tips")

    # Select a subset of categorical variables
    data = tips[["sex", "smoker", "day", "time"]]

    subset = data.sample(frac=1, random_state=1).reset_index(drop=True).head(20)

    # Step 1: Define the context
    sex = TreeContext("sex")
    sex.add_concept("Female")
    sex.add_concept("Male")

    smoker = TreeContext("smoker")
    smoker.add_concept("No")
    smoker.add_concept("Yes")

    day = GraphContext("day")
    day.add_concept("Mon", "Tue")
    day.add_concept("Tue", "Wed")
    day.add_concept("Wed", "Thur")
    day.add_concept("Thur", "Fri")
    day.add_concept("Fri", "Sat")
    day.add_concept("Sat", "Sun")
    day.add_concept("Sun", "Mon")

    time = TreeContext("time")
    time.add_concept("Dinner")
    time.add_concept("Lunch")

    # Step 2: Define the measure
    sex_measure = WuPalmer(sex)
    smoker_measure = WuPalmer(smoker)
    day_measure = PathLengthMeasure(day)
    time_measure = WuPalmer(time)

    # Step 3+4: Calculate Distance Matrix
    #           and map to euclidean vectors
    encoder = ContextualEncoder(
        [sex_measure, smoker_measure, day_measure, time_measure],
        aggregator="mean",
        inverters=["lin", "lin", "sqrt", "lin"],
    )
    encoded_data = encoder.transform(subset)

    print_data_points(encoded_data)
    print_matrix(encoder.get_similarity_matrix(), "Similarity")
    print_matrix(encoder.get_dissimilarity_matrix(), "Dissimilarity")

    return


def print_matrix(matrix, title):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.get_cmap("Blues"))

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(i, j, str(round(matrix[j, i], 2)), va="center", ha="center")

    ax.set_title(title)

    plt.show()


def print_data_points(data_points):
    # colors = np.random.rand(len(data_points))
    colors = np.ones(len(data_points))

    plt.scatter(data_points[:, 0], data_points[:, 1], c=colors, alpha=0.5)
    plt.title("Euclidean Data Points")
    plt.show()


if __name__ == "__main__":
    # readme_example()
    # simple_example()
    # advanced_example()
    multiple_attributes_example()
