import matplotlib.pyplot as plt
import pandas as pd

# if __name__ == '__main__':
#     file_name = 'data_modified/train.csv'
#     data = pd.read_csv(file_name)
#     feature_count = len(data.columns)
#
#     means = pd.read_csv(file_name, usecols=list(range(0, 256)), header=None)
#     print(means)
#
#     classes = pd.read_csv('data_modified/color.csv', header=None)
#     print(classes)
#
#     class_1_averages = []
#     class_2_averages = []
#     class_3_averages = []
#     class_4_averages = []
#     class_5_averages = []
#
#     for column in means.columns:
#         row = means[column]
#         current_c1_average = 0
#         current_c2_average = 0
#         current_c3_average = 0
#         current_c4_average = 0
#         current_c5_average = 0
#
#         for i in range(len(row)):
#             if i < 40:
#                 current_c1_average = current_c1_average + row[i]
#             elif i < 80:
#                 current_c2_average = current_c2_average + row[i]
#             elif i < 120:
#                 current_c3_average = current_c3_average + row[i]
#             elif i < 160:
#                 current_c4_average = current_c4_average + row[i]
#             else:
#                 current_c5_average = current_c5_average + row[i]
#
#         class_1_averages.append(current_c1_average/40)
#         class_2_averages.append(current_c2_average/40)
#         class_3_averages.append(current_c3_average/40)
#         class_4_averages.append(current_c4_average/40)
#         class_5_averages.append(current_c5_average/40)
#
#     plt.title('Average channel values for each class in the mutliclass feature file')
#     l1, = plt.plot(class_1_averages, label='Air')
#     l2, = plt.plot(class_2_averages, label='Book')
#     l3, = plt.plot(class_3_averages, label='Hand')
#     l4, = plt.plot(class_4_averages, label='Knife')
#     l5, = plt.plot(class_5_averages, label='Plastic Case')
#     plt.legend(handles=[l1, l2, l3, l4, l5])
#     plt.xlabel('Channel Number')
#     plt.ylabel('Mean Value')
#     plt.savefig('multiclass_visualisation')
