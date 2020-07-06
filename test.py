import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn


def loadAndpreprocessTestData(test_dataset_path, test_label_csv):
    '''
    To load test data from disk and applies the VGG16 preprocessing : converts from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.
    Note : test_label_csv should have following column :
        'image_index' : having index of image
        'type'        : class of chart (label)  

    Takes path of test dataset and labels as csv file path as input and returns batches of tensor image data
    '''

    test_df = pd.read_csv(test_label_csv)
    test_df['image_filename'] = test_df['image_index'].apply(lambda index: str(index) + '.png')

    IMAGE_SIZE = 224 # VGG input size

    image_generator = ImageDataGenerator( preprocessing_function=tf.keras.applications.vgg16.preprocess_input  )

    test_data_gen = image_generator.flow_from_dataframe(
        test_df, 
        directory=test_dataset_path, 
        x_col='image_filename', 
        y_col='type', 
        target_size=(IMAGE_SIZE, IMAGE_SIZE), 
        color_mode='rgb', 
        class_mode='categorical', 
        batch_size=8, 
        seed=None, 
        validate_filenames=True
    )
    
    return test_data_gen

def loadTestData(test_dataset_path, test_label_csv):
    '''
    Takes path of test dataset and labels as csv file path and returns :
        X_test : array of image arrays
        y_test : list of labels of the images : each label is a one hot vector
    '''
    
    test_data_gen = loadAndpreprocessTestData(test_dataset_path, test_label_csv)
    X_test, y_test = test_data_gen[0]


    for idx in range(1, len(test_data_gen)):
        x, y = test_data_gen[idx]
        print(x.shape, y.shape)
        X_test = np.concatenate((X_test, x), axis=0)
        y_test = np.concatenate((y_test, y), axis=0)

    return X_test, y_test    

def plotConfusionMatrix(y_true, y_pred):

    # dict mapping predicted index to chart type
    # index_to_chart = { 0:'dot_line', 1: 'hbar_categorical', 2: 'line', 3: 'pie', 4: 'vbar_categorical'} 
    
    chart_types = ['dot_line', 'hbar_categorical', 'line', 'pie', 'vbar_categorical']
    confusionMatrix = confusion_matrix(y_true, y_pred)
    
    df_cm = pd.DataFrame(confusionMatrix, chart_types, chart_types)

    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    # plt.matshow(confusionMatrix)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

def evaluateTestData(model, X_test, y_test):
    '''
    Takes trained model, X_test, y_test and calculates loss, accuracy other metrics
    '''
    print("\n================== Evaluating Test Data ===================\n")

    evaluation = model.evaluate(X_test, y_test, verbose = 0)
    print("Evaluation on test data :")
    print("loss = {}, accuracy = {}".format(evaluation[0], evaluation[1]))

    prediction_softmax = model.predict(X_test)
    predictedClasses = np.argmax(prediction_softmax, axis=1)
    # print ("prediction =" )
    # print(predictedClasses)

    plotConfusionMatrix(np.argmax(y_test, axis=1), predictedClasses)

if __name__ == "__main__":

    # trying validation data as its labels are known
    test_dataset_path = "chart_images_dataset/charts/train_val"
    test_label_csv    = "val.csv"

    X_test, y_test = loadTestData(test_dataset_path, test_label_csv)

    # loading trained model
    # NOTE : Use "chart_image_transfer_learning.ipynb" notebook to create trained model before testing  
    trained_model_path = 'VGG_based_model.h5'
    model = tf.keras.models.load_model(trained_model_path)
    
    evaluateTestData(model, X_test, y_test)

