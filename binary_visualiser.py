import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    file_name = 'binary/X.csv'
    data = pd.read_csv(file_name)
    feature_count = len(data.columns)

    means = pd.read_csv(file_name, usecols=list(range(0, 256)), header=None)
    print(means)

    classes = pd.read_csv('binary/y.csv', header=None)
    print(classes)

    class_1_averages = []
    class_2_averages = []

    for column in means.columns:
        row = means[column]
        current_c1_average = 0
        current_c2_average = 0
        for i in range(len(row)):
            if i < 40:
                current_c1_average = current_c1_average + row[i]
            else:
                current_c2_average = current_c2_average + row[i]

        class_1_averages.append(current_c1_average/40)
        class_2_averages.append(current_c2_average/40)

    plt.title('Average channel values for each class in the binary feature file')
    l1, = plt.plot(class_1_averages, label='Book')
    l2, = plt.plot(class_2_averages, label='Plastic Case')
    plt.legend(handles=[l1, l2])
    plt.xlabel('Channel Number')
    plt.ylabel('Mean Value')
    plt.savefig('binary_visualisation')
