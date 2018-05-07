import numpy as np
import gzip
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

global train_images, train_labels, train_labels_final, weights2, deltaO, deltaH, \
weights3,finding_accuracy_train,finding_accuracy_test,forward_phase_input_to_hidden, \
output_weight_update , input_weight_change

Learning_rate = 0.1
Momentum = 0.9
Hidden_layer = 20

forward_phase_input_to_hidden = np.zeros(Hidden_layer+1)
forward_phase_input_to_hidden[0] = 1
output_weight_update = np.zeros((Hidden_layer+1,10))
input_weight_change = np.zeros((785,Hidden_layer))
forward_phase_input_to_hidden_test = np.zeros(Hidden_layer+1)
forward_phase_input_to_hidden_test[0] = 1

class Perceptron:
    def __init__(self):
        f = gzip.open("train-images.gz", 'rb')
        self.train_images = np.frombuffer(f.read(), np.uint8, offset=16)
        self.train_images = self.train_images.reshape(-1, 1, 28, 28)
        self.train_images = self.train_images.reshape(self.train_images.shape[0],784)
        self.train_images = self.train_images / np.float32(255)
        self.train_images = np.append(self.train_images,np.ones((60000,1)),axis = 1)

        f = gzip.open("train-labels.gz", 'rb')
        self.train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
        self.train_labels = self.train_labels.reshape(60000,1)

        self.train_labels_final = np.full((60000, 10), 0.1, dtype=float)

        for row in range(len(self.train_labels)):
            index = self.train_labels[row].astype(int)
            self.train_labels_final[row][index[0]] += 0.8

        self.weights2 = np.random.uniform(-0.05, 0.05, size=(785,Hidden_layer))
        self.deltaO = np.zeros((Hidden_layer+1,10))
        self.deltaH = np.zeros((Hidden_layer,785))
        self.weights3 = np.random.uniform(-0.05, 0.05, size=(Hidden_layer+1,10))

        accuracy = 0

        self.finding_accuracy_train = np.zeros((60000,1))
        self.finding_accuracy_test = np.zeros((10000,1))

        f = gzip.open("test-images.gz", 'rb')
        self.test_images = np.frombuffer(f.read(), np.uint8, offset=16)
        self.test_images = self.test_images.reshape(-1, 1, 28, 28)
        self.test_images = self.test_images.reshape(self.test_images.shape[0],784)
        self.test_images = np.append(self.test_images,np.ones((10000,1)),axis = 1)


        f = gzip.open("test-labels.gz", 'rb')
        self.test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
        self.test_labels = self.test_labels.reshape(10000,1)

        self.test_labels_final = np.full((10000, 10), 0.1, dtype=float)

        for row in range(len(self.test_labels)):
            index = self.test_labels[row].astype(int)
            self.test_labels_final[row][index[0]] += 0.8

    def sigmoid(self,input):
        return (1/ (1 + np.exp(-input)))

    def matrix_mult_input_to_hidden(self,row):
        return np.dot(self.train_images[row][:],self.weights2)

    def matrix_mult_input_to_hidden_test(self,row,weights2):
        return np.dot(self.test_images[row][:],weights2)

    def matrix_mult_hidden_to_output(self,forward_phase_input_to_hidden):
        return np.dot(forward_phase_input_to_hidden,self.weights3)

    def matrix_mult_hidden_to_output_test(self,forward_phase_input_to_hidden,weights_3):
        return np.dot(forward_phase_input_to_hidden,weights_3)

    def output_error(self,row,output_sigmoid):
        return output_sigmoid * (1 - output_sigmoid) * (self.train_labels_final[row][:] - output_sigmoid)

    def hidden_error(self,forward_phase_input_to_hidden,output_error,):
        x = forward_phase_input_to_hidden[1:] * ( 1 - forward_phase_input_to_hidden[1:] )
        y =  np.dot(self.weights3[1:,:],np.transpose(output_error))
        hidden_error = x * y
        return hidden_error

    def hidden_to_output_weight_update(self,forward_phase_input_to_hidden,output_error):
        step_1 = Learning_rate * np.outer(forward_phase_input_to_hidden,output_error)
        step_2 = Momentum * self.deltaO
        newdeltaO = step_1 + step_2
        self.weights3 += newdeltaO
        self.deltaO = newdeltaO
        return self.weights3

    def  input_to_hidden_weight_update(self,row,hidden_error):
        step_1 = Learning_rate * np.outer(hidden_error,self.train_images[row][:])
        step_2 = Momentum * self.deltaH
        newdeltaH = step_1 + step_2
        self.weights2 += np.transpose(newdeltaH)
        self.deltaH = newdeltaH
        return self.weights2


def main():

    for i in range(50):
        global output_weight_update , input_weight_change
        class_object = Perceptron()
        for row in range(len(class_object.train_images)):
            input_hidden_mat_mul = class_object.matrix_mult_input_to_hidden(row)
            input_hidden_mat_mul_sigmoid = class_object.sigmoid(input_hidden_mat_mul)
            forward_phase_input_to_hidden[1:] = input_hidden_mat_mul_sigmoid
            hidden_output_mat_mul = class_object.matrix_mult_hidden_to_output(forward_phase_input_to_hidden)
            output_sigmoid = class_object.sigmoid(hidden_output_mat_mul)
            output_error = class_object.output_error(row,output_sigmoid)
            hidden_error = class_object.hidden_error(forward_phase_input_to_hidden,output_error)

            output_weight_update = class_object.hidden_to_output_weight_update(forward_phase_input_to_hidden,output_error)
            input_weight_change = class_object.input_to_hidden_weight_update(row,hidden_error)

            index = np.argmax(output_sigmoid,axis=0)
            class_object.finding_accuracy_train[row] =  index

        for row in range(len(class_object.test_images)):
            test_input_hidden = class_object.matrix_mult_input_to_hidden_test(row,input_weight_change)
            test_input_hidden_sigmoid = class_object.sigmoid(test_input_hidden)

            forward_phase_input_to_hidden_test[1:] = test_input_hidden_sigmoid

            test_hidden_output_mat_mul = class_object.matrix_mult_hidden_to_output_test(forward_phase_input_to_hidden_test,output_weight_update)
            test_output_sigmoid = class_object.sigmoid(test_hidden_output_mat_mul)

            index = np.argmax(test_output_sigmoid,axis=0)
            class_object.finding_accuracy_test[row] =  index
            '''
            print test_input_hidden.shape
            forward_phase_input_to_hidden_test[1:] = 1 / (1 + np.exp(-test_input_hidden))

            test_hidden_output_mat_mul = np.dot(forward_phase_input_to_hidden_test,class_object.output_weight_update)
            test_output_sigmoid = 1 / (1 + np.exp(-test_hidden_output_mat_mul))

            index = np.argmax(output_sigmoid,axis=0)
            class_object.finding_accuracy_test[row] =  index
            '''
    cfm_train = confusion_matrix(class_object.train_labels,class_object.finding_accuracy_train)
    diagonal_sum_train =  sum(np.diag(cfm_train))
    accuracy_train = (diagonal_sum_train/60000.00)*100
    print cfm_train
    print accuracy_train

    cfm_test = confusion_matrix(class_object.test_labels,class_object.finding_accuracy_test)
    diagonal_sum_test =  sum(np.diag(cfm_test))
    accuracy_test = (diagonal_sum_test/10000.00)*100
    print cfm_test
    print accuracy_test

    plt.plot(accuracy_train)
    plt.plot(accuracy_test)
    plt.ylabel("Accuracy in %")
    plt.xlabel("Epoch")

    image= "hidden100.png"
    plt.title("For 50 hidden units")
    plt.savefig(image)
    plt.show()

main()
