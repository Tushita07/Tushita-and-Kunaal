# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from neural import NeuralNet
import pandas as pd

# each row is an (input, output) tuple
xor_data = [
    #   input     output    corresponding example
    ([0.0, 0.0],  [0.0]),  #[0, 0] => 0
    ([0.0, 1.0],  [1.0]),  #[0, 1] => 1
    ([1.0, 0.0],  [1.0]),  #[1, 1] => 1
    ([1.0, 1.0],  [0.0])   #[1, 0] => 0
]

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def read_wine_data():

    #names = pd.read_fwf('wine.names')
    contents = pd.read_csv("wine.data",
                           names=['winery', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                                  'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
                                  'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'])
    print(contents)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')

    read_wine_data()
    #
    # nn = NeuralNet(2, 1, 1) #changed to perceptron, 1 node vs. 5
    # nn.train(xor_data,iters = 2000, print_interval=50)
    #
    # print(nn.get_ho_weights())
    # print(nn.get_ih_weights())
    #
    # print()
    # print('Evaluate [0,1]')
    # print(nn.evaluate([0.0, 1.0]))
    #
    # print()
    # for triple in nn.test_with_expected(xor_data):
    #     print(triple)
    #
    # print()
    # for i in nn.test_with_expected(xor_data):
    #     print(f"desired: {i[1]}, actual: {i[2]}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
