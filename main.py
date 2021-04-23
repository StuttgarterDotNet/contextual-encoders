import numpy as np
import matplotlib.pyplot as plt
from contextual_encoders import TreeContext, WuPalmerComparer, ContextualEncoder


def print_matrix(matrix, title):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues)

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(i, j, str(round(matrix[j, i], 2)), va='center', ha='center')

    ax.set_title(title)

    plt.show()


def print_data_points(data_points, title):
    colors = np.random.rand(len(data_points))

    plt.scatter(data_points[:, 0], data_points[:, 1], c=colors, alpha=0.5)
    plt.title=title
    plt.show()


def main():
    color = TreeContext('color')
    color.add_concept('dark')
    color.add_concept('light')
    color.add_concept('black', 'dark')
    color.add_concept('dark-blue', 'dark')
    color.add_concept('white', 'light')
    color.add_concept('light-blue', 'light')

    gender = TreeContext('gender')
    gender.add_concept('male')
    gender.add_concept('female')
    gender.add_concept('neutral')

    header = ['color', 'gender']
    data = np.array([
        ['dark,white', 'male'],
        ['dark', 'female'],
        ['light', 'female'],
        ['light-blue,dark-blue', 'neutral']
    ])

    color_comparer = WuPalmerComparer(color)
    gender_comparer = WuPalmerComparer(gender)

    encoder = ContextualEncoder([color_comparer, gender_comparer])
    data_points = encoder.transform(data)

    print_data_points(data_points, 'Data points')
    print_matrix(encoder.get_matrix(), 'Distance Matrix')

    return


def main_boston():
    bunch = load_boston()
    y = bunch.target
    x = pd.DataFrame(bunch.data, columns=bunch.feature_names)

    rad = TreeContext('rad')
    rad.add_concept('1.0')
    rad.add_concept('2.0')
    rad.add_concept('3.0')
    rad.add_concept('4.0')

    rad_comparer = WuPalmerComparer(rad)

    encoder = ContextualEncoder([rad_comparer], ['RAD'], 'id')
    data_points = encoder.transform(x)

    print_data_points(data_points, 'Data points')
    print_matrix(encoder.get_matrix(), 'Distance Matrix')


def main_old():
    color = TreeContext('color')
    color.add_concept('dark')
    color.add_concept('light')
    color.add_concept('black', 'dark')
    color.add_concept('dark-blue', 'dark')
    color.add_concept('white', 'light')
    color.add_concept('light-blue', 'light')

    gender = TreeContext('gender')
    gender.add_concept('male')
    gender.add_concept('female')
    gender.add_concept('neutral')

    header = ['color', 'gender']
    data = np.array([
        ['dark,white', 'male'],
        ['dark', 'female'],
        ['light', 'female'],
        ['light-blue,dark-blue', 'neutral']
    ])

    color_comparer = WuPalmerComparer(color, offset='depth', verbose=True)
    gender_comparer = WuPalmerComparer(gender, offset='depth', verbose=True)

    # color_comparer.import_from_file('color_comparer.json')
    # gender_comparer.import_from_file('gender_comparer.json')

    color_computer = SimilarityMatrixComputer(comparer=color_comparer)
    gender_computer = SimilarityMatrixComputer(comparer=gender_comparer)

    color_matrix = color_computer.compute(data[:, 0])
    gender_matrix = gender_computer.compute(data[:, 1])

    print_matrix(color_matrix, 'Color')
    print_matrix(gender_matrix, 'Gender')

    color_comparer.export_to_file('color_comparer.json')
    gender_comparer.export_to_file('gender_comparer.json')

    mean_aggregator = MeanAggregator()
    median_aggregator = MedianAggregator()

    mean = mean_aggregator.aggregate([color_matrix, gender_matrix])
    median = median_aggregator.aggregate([color_matrix, gender_matrix])

    print_matrix(mean, 'Mean')
    print_matrix(median, 'Median')

    reducer = MultidimensionalScalingReducer()
    data_points = reducer.reduce(mean)

    print_data_points(data_points, 'Reduced data points')

    return


if __name__ == "__main__":
    main()
