import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2

import models

def predict_camera(model_data_path):
    # default input size
    height = 480
    width = 640
    channels = 3
    batch_size = 1

    # create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    # get video stream
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Restore the training parameters from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        while True:
            ret, frame = cap.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = img.astype('float32')
            img = np.expand_dims(img, axis = 0)

            # Evalute the network for the given image
            pred = sess.run(net.get_output(), feed_dict={input_node: img})

            dst = pred[0,:,:,0]
            maxnum = dst.max()
            dst = dst / maxnum * 256
            dst = dst.astype('uint8')
            cv2.imshow('result', dst)
            if(cv2.waitKey(1) & 0xFF == ord('q')):
                break
                
        cap.release()
        cv2.destroyAllWindows()



def predict(model_data_path, image_path):

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Read image(OpenCV)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')
    img = np.expand_dims(img, axis = 0)
   
    # Read image (pillow)
    # img = Image.open(image_path)
    # img = img.resize([width,height], Image.ANTIALIAS)
    # img = np.array(img).astype('float32')
    # img = np.expand_dims(np.asarray(img), axis = 0)
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
        # Plot result (matplotlib)
        # fig = plt.figure()
        # ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        # fig.colorbar(ii)
        # plt.show()

        # show result (OpenCV)
        output = pred[0,:,:,0]
        maxnum = output.max()
        output = output / maxnum * 256
        output = output.astype('uint8')
        cv2.imshow('result', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    # pred = predict(args.model_path, args.image_paths)
    predict_camera(args.model_path)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



