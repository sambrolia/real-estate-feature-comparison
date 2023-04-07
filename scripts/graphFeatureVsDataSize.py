import matplotlib.pyplot as plt

dataset_sizes = [10000, 25000, 50000, 75000, 100000, 125000, 150000, 175000, "Full"]
street_name_importances = [0.2187, 0.3537, 0.2345, 0.3013, 0.2623, 0.2860, 0.2197, 0.2289, 0.2220]
use_code_importances = [0.4758, 0.3857, 0.4992, 0.3497, 0.3338, 0.3200, 0.2922, 0.2940, 0.2918]

x = list(range(len(dataset_sizes)))

plt.bar(x, street_name_importances, width=0.4, label='StreetName')
plt.bar([i + 0.4 for i in x], use_code_importances, width=0.4, label='use_code')

plt.xticks([i + 0.2 for i in x], dataset_sizes)
plt.xlabel('Dataset Size')
plt.ylabel('Importance')
plt.title('Comparison of Feature Importances for Different Dataset Sizes')
plt.legend()

plt.show()
