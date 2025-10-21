import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Load the MNIST data
data = pd.read_csv("./data/MNIST/mnist_train.csv")
data = np.array(data)

# Define the plot_number function
def plot_number(n, guessed_number):
    data_images = data[:, 1:]
    data_labels = data[:, 0]

    image = data_images[n].reshape(28, 28)

    plt.imshow(image, cmap="gray")
    plt.title(f"Actual Number: {data_labels[n]} | Guessed Number: {guessed_number}")
    plt.show()

# Ensure the function is not executed on import
if __name__ == "__main__":
    plot_number(616, 5)