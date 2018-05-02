import numpy as np
import gzip
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

global train_images
train_images = np.zeros((60000,784))
global train_labels
train_labels = np.zeros((60000,1))
global train_labels_final
train_labels_final = np.zeros((60000,10))
global test_images
test_images = np.zeros((10000,784))
global test_labels
test_labels = np.zeros((10000,1))
global hidden_layer_weights , output_layer_weights, output_layer_weights_with_bias, modified_weights_hidden_to_output, \
modified_weights_input_to_hidden

global accuracy_count ,counter
global finding_accuracy
finding_accuracy = np.zeros((60000,1))
accuracy_count = counter = 0


Hidden_Layer_1 = 20
Hidden_Layer_2 = 50
Hidden_Layer_3 = 100

global Learning_rate
Learning_rate = 0.1
global Momentum
Momentum = 0.9

hidden_layer_weights = np.zeros((785,Hidden_Layer_1))
modified_weights_input_to_hidden = np.zeros((Hidden_Layer_1,785))
output_layer_weights_with_bias = np.zeros((Hidden_Layer_1+1,10))
output_layer_weights = np.zeros((Hidden_Layer_1,10))
modified_weights_hidden_to_output = np.zeros((10,Hidden_Layer_1+1))

############################### Pre-Processing #################################
class preProcessInput:


    def __init__(self):
        global train_labels,train_images,train_labels_final,hidden_layer_weights,output_layer_weights_with_bias,\
        modified_weights_hidden_to_output,modified_weights_input_to_hidden
        f = gzip.open("train-images.gz", 'rb')
        train_images = np.frombuffer(f.read(), np.uint8, offset=16)
        train_images = train_images.reshape(-1, 1, 28, 28)
        train_images = train_images.reshape(train_images.shape[0],784)
        train_images = train_images / np.float32(255)
        train_images = np.append(train_images,np.ones((60000,1)),axis = 1)

        f = gzip.open("train-labels.gz", 'rb')
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
        train_labels = train_labels.reshape(60000,1)

        train_labels_final = np.full((60000, 10), 0.1, dtype=float)

        for row in range(len(train_labels)):
            index = train_labels[row].astype(int)
            train_labels_final[row][index[0]] += 0.8
        '''
        f = gzip.open("test-images.gz", 'rb')
        self.test_images = np.frombuffer(f.read(), np.uint8, offset=16)
        self.test_images = self.test_images.reshape(-1, 1, 28, 28)
        self.test_images = self.test_images.reshape(self.test_images.shape[0],784)
        self.test_images = self.test_images / np.float32(255)
        self.test_images = np.append(self.test_images,np.ones((10000,1)),axis = 1)

        f = gzip.open("test-labels.gz", 'rb')
        self.test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
        self.test_labels = self.test_labels.reshape(10000,1)
        '''
        hidden_layer_weights = np.random.uniform(-0.05, 0.05, size=(785,Hidden_Layer_1))

        output_layer_weights_with_bias = np.random.uniform(-0.05, 0.05, size=(Hidden_Layer_1+1,10))
        output_layer_weights = output_layer_weights_with_bias[:20,:]
        modified_weights_hidden_to_output = np.zeros((10,Hidden_Layer_1+1))
        modified_weights_input_to_hidden = np.zeros((Hidden_Layer_1,785))

    def sigmoid (self,calc_sigma):
        #print np.around((1/(1 + np.exp(-1 * np.around(calc_sigma, decimals = 1)))), decimals = 1)
        return 1/(1 + np.exp(-calc_sigma))


    def matrix_multiplication_hidden_layer (self,row_test):
        return np.matmul(train_images[row_test:row_test+1],hidden_layer_weights)

    def matrix_multiplication_output_layer (self,step_1):
        return np.matmul(step_1,output_layer_weights_with_bias)

    def calc_error_term_output_to_hidden(self,row,y_final):
        step_1 = np.ones((1,10))
        step_1 = step_1 - y_final
        global counter,accuracy_count
        global finding_accuracy
        #global train_labels,train_labels_final
        step_2 = np.matmul(y_final,np.transpose(step_1))
        #print step_2

        step_3 = train_labels_final[row:row+1] - np.around(y_final,decimals = 1)
        #print step_3
        step_4 = np.matmul(step_2,step_3)
        #print step_4
        max_index = np.argmax(np.around(y_final,decimals = 1), axis = 1)
        finding_accuracy[row,:] = max_index
        #print self.train_labels[row]
        #if finding_accuracy[0] == 0:
        #    accuracy_count += 1
        #else:

        return step_4

    def calc_error_term_hidden_to_input(self,hidden_layer_calc,delta_k):
        step_1 = np.ones((1,20))
        step_2 = step_1 - hidden_layer_calc
        step_3 = np.matmul(hidden_layer_calc,np.transpose(step_2))
        step_4 = np.matmul(output_layer_weights,np.transpose(delta_k))
        step_5 = np.matmul(step_3,np.transpose(step_4))
        return step_5

    def update_weights_hidden_to_output(self,delta_k,hidden_layer_calc_after_bias):
        global modified_weights_hidden_to_output,output_layer_weights_with_bias
        step_1 = Learning_rate * (np.matmul(np.transpose(delta_k),hidden_layer_calc_after_bias))
        step_2 = Momentum * modified_weights_hidden_to_output
        modified_weights_hidden_to_output = step_1 + step_2
        output_layer_weights_with_bias +=  np.transpose(modified_weights_hidden_to_output)
        return output_layer_weights_with_bias

    def update_weights_input_to_hidden(self,row_test,delta_h,hidden_layer_calc):
        global modified_weights_input_to_hidden,hidden_layer_weights
        step_1 = Learning_rate * (np.matmul(np.transpose(delta_h),train_images[row_test:row_test+1]))
        step_2 = Momentum * modified_weights_input_to_hidden
        modified_weights_input_to_hidden = step_1 + step_2
        hidden_layer_weights +=  np.transpose(modified_weights_input_to_hidden)
        return hidden_layer_weights

def main():
    object = preProcessInput()
    for i in range (3):
        for row in range(len(train_images)):
                step_1 = object.matrix_multiplication_hidden_layer(row) #1*20
                step_1_without_bias = object.sigmoid(step_1) #1*20s
                step_1 = np.append(step_1_without_bias,np.ones((1,1)),axis = 1) #1*21
                step_2 = object.matrix_multiplication_output_layer(step_1)#1*10
                step_2 = object.sigmoid(step_2)#1*10
                step_3_delta_k = object.calc_error_term_output_to_hidden(row,step_2)
                if (argmax_index = np.argmax(np.around(step_3_delta_k,decimals = 1), axis = 1)) == train_labels(row):
                    continue
                    print "no update needed"
                else:
                    step_4 = object.calc_error_term_hidden_to_input(step_1_without_bias,step_3_delta_k);
                    step_5_update_weights_hidden_to_output = object.update_weights_hidden_to_output(step_3_delta_k,step_1)
                    step_6_update_weights_input_to_ohidden = object.update_weights_input_to_hidden(row,step_4,step_1_without_bias)

    finding_accuracy.reshape(60000,1)
    cfm = confusion_matrix(train_labels,finding_accuracy)
    diagonal_sum =  sum(np.diag(cfm))
    accuracy = (diagonal_sum/60000.00)*100
    print cfm
    print accuracy

main()
