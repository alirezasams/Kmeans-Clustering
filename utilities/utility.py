import matplotlib.pyplot as plt


def visualize_2d(input_data, features):
    fig, ax = plt.subplots()
    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'pink', 4: 'black', 5: 'orange', 6: 'cyan',
              7: 'yellow', 8: 'brown', 9: 'purple', 10: 'white', 11: 'grey', 12: 'lightblue',
              13: 'lightgreen', 14: 'darkgrey'}
    ax.scatter(input_data[features[0]],
               input_data[features[1]],
               c=input_data['cluster'].apply(lambda x: colors[x]))

    fig_path = 'breast_cancer_data clusters.jpg'
    plt.savefig(fig_path)
