import os
import time
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import socket
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import models
from preprocess import get_data

def getModel(model_name): 
    if(model_name == 'EEGTCNet'):
        # Train using EEGTCNet: https://arxiv.org/abs/2006.00622
        model = models.EEGTCNet(n_classes = 4)          
    elif(model_name == 'EEGNet'):
        # Train using EEGNet: https://arxiv.org/abs/1611.08024
        model = models.EEGNet_classifier(n_classes = 4) 
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model


def compare_labels_and_predictions(labels, inference_results, n_classes):
    # Count correct and incorrect predictions
    correct_counts = [0] * n_classes
    incorrect_counts = [0] * n_classes
    
    for true, pred in zip(labels, inference_results):
        if true == pred:
            correct_counts[true] += 1
        else:
            incorrect_counts[true] += 1

    # Plotting the comparison
    x = np.arange(n_classes)
    width = 0.35  # bar width

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, correct_counts, width, label='Correct', color='g')
    ax.bar(x + width/2, incorrect_counts, width, label='Incorrect', color='r')

    ax.set_ylabel('Count')
    ax.set_title('Predictions vs. True Labels')
    ax.set_xticks(x)
    ax.set_xticklabels(np.arange(n_classes))
    ax.legend()

    plt.show()
    # plt.savefig('./comparsion_res.png', dpi=300)


def Evaluation(model, dataset_conf, results_path, use_socket=False):
    # Initialize socket for communication (if use_socket is True)
    if use_socket:
        server_address = ('10.26.141.117', 65432)  # Replace with the target PC's IP address and port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(server_address)

    # Open the "Log" file to write the evaluation results
    log_write = open(results_path + "/log.txt", "a")
    
    # Get dataset parameters
    n_classes = dataset_conf.get('n_classes')
    n_sub = 3-1
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')

    # List to store inference results
    inference_results = []

    # Load the model for inference
    filepath = '/saved models/run-3/subject-3.h5'
    model.load_weights(results_path + filepath)
    
    # Load data
    _, _, _, X_test, _, y_test_onehot = get_data(data_path, n_sub, LOSO, isStandard)
    
    # Predict MI task using the loaded model
    y_pred = model.predict(X_test).argmax(axis=-1)
    
    # Calculate accuracy and K-score
    labels = y_test_onehot.argmax(axis=-1)
    
    # Store inference results in a list (including subject, accuracy, and kappa)
    inference_results = y_pred

    # Send the inference results through the socket
    if use_socket:
        try:
            sock.sendall(str(inference_results).encode('utf-8'))
            print("Inference results sent to the server.")
        except Exception as e:
            print(f"Error sending data to the server: {e}")

    # Close open files and socket connection     
    log_write.close()
    if use_socket:
        sock.close()

    return inference_results, labels


if __name__ == '__main__':
    # Set the path of the results folder
    data_path = "/home/artinx/workspace/COURSE/datasets/BCICIV_2a_mat/"
    results_path = os.getcwd() + "/eegnet_results"
    
    # Load the dataset configuration
    dataset_conf = { 
        'n_classes': 4, 'n_sub': 9, 'n_channels': 22, 'data_path': data_path,
        'isStandard': True, 'LOSO': False
    }

    # Load the model
    model = models.EEGNet_classifier(n_classes=4)
    
    # Evaluate the model
    inference_results, labels = Evaluation(model, dataset_conf, results_path)

    print(inference_results)
    print(labels)

    # Compare the labels and predictions
    compare_labels_and_predictions(labels, inference_results, n_classes=4)