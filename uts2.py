#Nama : Alwan Rofiqi_21091397020
#Multi Neuron Batch Input 

#inisialisasi numpy
import numpy as np

#inisialisasi variabel dengan matriks 10 x 6 yang mana input layers feature 10 dan input perbachtnya 6
inputs = [[1, 8, 2, 1, 5, 6, 2, 3, 4, 1],
          [2, 1, 1, 3, 7, 10, 9, 2, 3, 1],
          [7, 1, 3, 4, 7, 1, 4, 2, 3, 10],
          [1, 3, 8, 2, 3, 9, 2, 7, 2, 4],
          [2, 4, 1, 2, 6, 3, 2, 4, 1, 2],
          [3, 2, 3, 1, 3, 2, 1, 3, 10,3]],
          
#panjang weights tergantung pada berapa banyaknya input layers yang telah ditentukan
#jumlah weights itu tergantung pada berapa banyak neuron yang ada 
weights1 =[[1, 2, 6, 1, 2, 3, 1, 3, 3, 4],
           [2, 3, 7, 5, 1, 11, 5, 1, 2, 1],
           [1, 2, 6, 8, 9, 10, 11, 3, 1, 3],
           [-2, 4, -10, 4, 1, 2, 2, 3, 1, 3],
           [-3, 5, 1, 4, 1, 6, 8, 10, 6, 11]]
#bias neuron
#jumlah batas panjang bias tergantung pada berapa banyak neuron yang ada
biases1 = [4, 1, 2, 6, 3]
#jumlah weights itu tergantung pada berapa banyak neuron pada layer1 
weights2 =[[3, 1, 3, 3, 4],
           [7, 8, 1, 6, 1],
           [2, 5, 2, 2, 4]]

#jumlah batas panjang bias tergantung pada berapa banyak neuron yang ada
biases2 = [4, 2, 6]
#output
layers1_outputs= np.dot(inputs, np.array(weights1).T) + biases1
#output 
layers2_outputs= np.dot(layers1_outputs, np.array(weights2).T) + biases2
#print output
print(layers2_outputs)