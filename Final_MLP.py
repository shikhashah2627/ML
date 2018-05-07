import numpy as np
import gzip
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

global train_images, train_labels, train_labels_final, weights2, deltaO, deltaH, \
weights3,finding_accuracy_train,finding_accuracy_test,forward_phase_input_to_hidden, \
output_weight_update , input_weight_change, accuracy_train, accuracy_test,\
train_images_quarter, train_images_half,train_labels_quarter, train_labels_half, \
train_labels_quarter_final, train_labels_half_final

main_size = 60000.00
main_size_quarter = 15000.00
main_size_half = 10000.00
Learning_rate = 0.1
Momentum = 0.5
Hidden_layer = 100

forward_phase_input_to_hidden = np.zeros(Hidden_layer+1)
forward_phase_input_to_hidden[0] = 1
output_weight_update = np.zeros((Hidden_layer+1,10))
input_weight_change = np.zeros((785,Hidden_layer))
forward_phase_input_to_hidden_test = np.zeros(Hidden_layer+1)
forward_phase_input_to_hidden_test[0] = 1
accuracy_train = np.zeros((50))
accuracy_test = np.zeros((50))

class Perceptron:
    def __init__(self):
        f = gzip.open("train-images.gz", 'rb')
        self.train_images = np.frombuffer(f.read(), np.uint8, offset=16)
        self.train_images = self.train_images.reshape(-1, 1, 28, 28)
        self.train_images = self.train_images.reshape(self.train_images.shape[0],784)
        self.train_images = self.train_images / np.float32(255)
        self.train_images = np.append(self.train_images,np.ones((60000,1)),axis = 1)
        #for experiment 3
        #self.train_images_quarter = self.train_images[:15000,:]
        self.train_images_half = self.train_images[:10000,:]


        f = gzip.open("train-labels.gz", 'rb')
        self.train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
        self.train_labels = self.train_labels.reshape(60000,1)
        #for experiment 3
        #self.train_labels_quarter = self.train_labels[:15000,:]
        self.train_labels_half = self.train_labels[:10000,:]

        #self.train_labels_final = np.full((60000, 10), 0.1, dtype=float)
        #self.train_labels_quarter_final = np.full((15000, 10), 0.1, dtype=float)
        self.train_labels_half_final = np.full((10000, 10), 0.1, dtype=float)

        for row in range(len(self.train_labels_half_final)):
            index = self.train_labels_half[row].astype(int)
            self.train_labels_half_final[row][index[0]] += 0.8

        #self.train_labels_quarter_final = self.train_labels_final[:15000,:]
        #self.train_labels_quarter_final = self.train_labels_final[:10000,:]

        self.weights2 = np.random.uniform(-0.05, 0.05, size=(785,Hidden_layer))
        self.deltaO = np.zeros((Hidden_layer+1,10))
        self.deltaH = np.zeros((Hidden_layer,785))
        self.weights3 = np.random.uniform(-0.05, 0.05, size=(Hidden_layer+1,10))

        accuracy = 0

        self.finding_accuracy_train = np.zeros((10000,1))
        #exeriment 3
        #self.finding_accuracy_train = np.zeros((15000,1))
        #self.finding_accuracy_train = np.zeros((10000,1))

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
        #return np.dot(self.train_images[row][:],self.weights2)
        #experiment 3
        #return np.dot(self.train_images_quarter[row][:],self.weights2)
        return np.dot(self.train_images_half[row][:],self.weights2)

    def matrix_mult_input_to_hidden_test(self,row,weights2):
        return np.dot(self.test_images[row][:],weights2)

    def matrix_mult_hidden_to_output(self,forward_phase_input_to_hidden):
        return np.dot(forward_phase_input_to_hidden,self.weights3)

    def matrix_mult_hidden_to_output_test(self,forward_phase_input_to_hidden,weights_3):
        return np.dot(forward_phase_input_to_hidden,weights_3)

    def output_error(self,row,output_sigmoid):
        #return output_sigmoid * (1 - output_sigmoid) * (self.train_labels_final[row][:] - output_sigmoid)
        #experiment 3
        #return output_sigmoid * (1 - output_sigmoid) * (self.train_labels_quarter_final[row][:] - output_sigmoid)
        return output_sigmoid * (1 - output_sigmoid) * (self.train_labels_half_final[row][:] - output_sigmoid)

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
        #step_1 = Learning_rate * np.outer(hidden_error,self.train_images[row][:])
        #experiment 3
        #step_1 = Learning_rate * np.outer(hidden_error,self.train_images_quarter[row][:])
        step_1 = Learning_rate * np.outer(hidden_error,self.train_images_half[row][:])

        step_2 = Momentum * self.deltaH
        newdeltaH = step_1 + step_2
        self.weights2 += np.transpose(newdeltaH)
        self.deltaH = newdeltaH
        return self.weights2


def main():

    for i in range(50):
        global output_weight_update , input_weight_change
        class_object = Perceptron()
        for row in range(len(class_object.train_images_half)):
            input_hidden_mat_mul = class_object.matrix_mult_input_to_hidden(row)
            input_hidden_mat_mul_sigmoid = class_object.sigmoid(input_hidden_mat_mul)
            forward_phase_input_to_hidden[1:] = input_hidden_mat_mul_sigmoid
            hidden_output_mat_mul = class_object.matrix_mult_hidden_to_output(forward_phase_input_to_hidden)
            output_sigmoid = class_object.sigmoid(hidden_output_mat_mul)
            output_error = class_object.output_error(row,output_sigmoid)
            hidden_error = class_object.hidden_error(forward_phase_input_to_hidden,output_error)
            # updating weights
            output_weight_update = class_object.hidden_to_output_weight_update(forward_phase_input_to_hidden,output_error)
            input_weight_change = class_object.input_to_hidden_weight_update(row,hidden_error)
            #useful for confusion_matrix
            index = np.argmax(output_sigmoid,axis=0)
            class_object.finding_accuracy_train[row] =  index
        print "entering test section :" , i

        for row in range(len(class_object.test_images)):
            #forward phase
            test_input_hidden = class_object.matrix_mult_input_to_hidden_test(row,input_weight_change)
            test_input_hidden_sigmoid = class_object.sigmoid(test_input_hidden)
            forward_phase_input_to_hidden_test[1:] = test_input_hidden_sigmoid
            test_hidden_output_mat_mul = class_object.matrix_mult_hidden_to_output_test(forward_phase_input_to_hidden_test,output_weight_update)
            test_output_sigmoid = class_object.sigmoid(test_hidden_output_mat_mul)

            index = np.argmax(test_output_sigmoid,axis=0)
            class_object.finding_accuracy_test[row] =  index

        #cfm_train = confusion_matrix(class_object.train_labels,class_object.finding_accuracy_train)
        #experiment 3
        #cfm_train = confusion_matrix(class_object.train_labels_quarter,class_object.finding_accuracy_train)
        cfm_train = confusion_matrix(class_object.train_labels_half,class_object.finding_accuracy_train)
        diagonal_sum_train =  sum(np.diag(cfm_train))
        #accuracy_train[i] = (diagonal_sum_train/main_size)*100
        #experiment 3
        accuracy_train[i] = (diagonal_sum_train/main_size_half)*100
        #accuracy_train[i] = (diagonal_sum_train/main_size_half)*100
        #print cfm_train
        #print class_object.accuracy_train

        cfm_test = confusion_matrix(class_object.test_labels,class_object.finding_accuracy_test)
        diagonal_sum_test =  sum(np.diag(cfm_test))
        accuracy_test[i] = (diagonal_sum_test/10000.00)*100
        #print cfm_test
        #print class_object.accuracy_test

    print cfm_test
    print accuracy_test

    print cfm_train
    print accuracy_train

    plt.plot(accuracy_train)
    plt.plot(accuracy_test)
    plt.ylabel("Accuracy in %")
    plt.xlabel("Epoch")

    image= "100_half.png"
    plt.title("For 100 hidden units _ 0.5")
    plt.savefig(image)
    plt.show()

main()
