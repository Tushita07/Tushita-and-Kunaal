# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from neural import NeuralNet

# each row is an (input, output) tuple
xor_data = [
    #   input     output    corresponding example
    ([0.0, 0.0],  [0.0]),  #[0, 0] => 0
    ([0.0, 1.0],  [1.0]),  #[0, 1] => 1
    ([1.0, 0.0],  [1.0]),  #[1, 1] => 1
    ([1.0, 1.0],  [0.0])   #[1, 0] => 0
]


nn = NeuralNet(2, 5, 1)
nn.train(xor_data)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    nn = NeuralNet(2, 5, 1)
    nn.train(xor_data)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
