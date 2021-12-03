
#kgx8720
#tse5029

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from neural import NeuralNet
import pandas as pd
from sklearn import preprocessing

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
    contents = pd.read_csv("wine.data", header = None)
                           # names=['winery', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                           #        'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
                           #        'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'])
    print(contents)
    return contents

def pre_process(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    #print(result)



    return result

# def train_wine_data(data):
#     nn = NeuralNet(13,3,1)
#
#     #print(data)
#     input_data = data.iloc[: ,1:]
#     input_data_list = input_data.values.tolist()
#     output_data = data.iloc[: ,0]
#     output_data_list = output_data.values.tolist()
#
#     wine_data = []
#
#
#     for row in range(0, len(input_data_list)):
#         wine_data.append((input_data_list[row],[output_data_list[row]]))
#
#     #print(wine_data[0])
#
#     nn.train(wine_data, iters=1000, print_interval=50)

def df_to_list(data):
    input_data = data.iloc[:, 1:]
    input_data_list = input_data.values.tolist()
    output_data = data.iloc[:, 0]
    output_data_list = output_data.values.tolist()

    wine_data = []

    for row in range(0, len(input_data_list)):
        wine_data.append((input_data_list[row], [output_data_list[row]]))

    return wine_data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')

    data = read_wine_data()
    proc_data = pre_process(data)

    train = proc_data.sample(frac = 0.8)
    test = proc_data.drop(train.index)

    nn = NeuralNet(13, 3, 1)
    nn.train(df_to_list((train)), iters=10000, print_interval=1000)

    #nn.test_with_expected(df_to_list(test))

    for i in nn.test_with_expected(df_to_list(test)):
        print(f"desired: {i[1]}, actual: {i[2]}")

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
