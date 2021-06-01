import seaborn as sns
from contextual_encoders import TreeContext, WuPalmerComparer, ContextualEncoder
import matplotlib.pyplot as plt
import numpy as np


def simple_example():

    # Load the tips dataset
    tips = sns.load_dataset("tips")

    # Select a subset of categorical variables
    data = tips["day"]

    # Step 1: Define the context
    day = TreeContext("day")
    day.add_concept("Mon")
    day.add_concept("Tue")
    day.add_concept("Wed")
    day.add_concept("Thur")
    day.add_concept("Fri")
    day.add_concept("Sat")
    day.add_concept("Sun")

    # Step 2: Define the comparer
    day_comparer = WuPalmerComparer(day)

    # Step 3+4: Calculate Distance Matrix
    #           and map to euclidean vectors
    encoder = ContextualEncoder(day_comparer, n_components=2)
    encoded_data = encoder.transform(data)

    print_data_points(encoded_data, "Day")


def advanced_example():

    # Load the tips dataset
    tips = sns.load_dataset("tips")

    # Select a subset of categorical variables
    data = tips[["sex", "smoker", "day", "time"]]

    # Step 1: Define the context
    sex = TreeContext("sex")
    sex.add_concept("Female")
    sex.add_concept("Male")

    smoker = TreeContext("smoker")
    smoker.add_concept("No")
    smoker.add_concept("Yes")

    day = TreeContext("day")
    day.add_concept("Mon")
    day.add_concept("Tue")
    day.add_concept("Wed")
    day.add_concept("Thur")
    day.add_concept("Fri")
    day.add_concept("Sat")
    day.add_concept("Sun")

    time = TreeContext("time")
    time.add_concept("Dinner")
    time.add_concept("Lunch")

    # Step 2: Define the comparer
    sex_comparer = WuPalmerComparer(sex)
    smoker_comparer = WuPalmerComparer(smoker)
    day_comparer = WuPalmerComparer(day)
    time_comparer = WuPalmerComparer(time)

    # Step 3+4: Calculate Distance Matrix
    #           and map to euclidean vectors
    encoder = ContextualEncoder(
        [sex_comparer, smoker_comparer, day_comparer, time_comparer]
    )
    encoder = ContextualEncoder(day_comparer, n_components=3)
    encoded_data = encoder.transform(data)

    print(encoded_data)


def print_matrix(matrix, title):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues)

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(i, j, str(round(matrix[j, i], 2)), va="center", ha="center")

    ax.set_title(title)

    plt.show()


def print_data_points(data_points, title):
    colors = np.random.rand(len(data_points))

    plt.scatter(data_points[:, 0], data_points[:, 1], c=colors, alpha=0.5)
    plt.title = title
    plt.show()


if __name__ == "__main__":
    simple_example()
