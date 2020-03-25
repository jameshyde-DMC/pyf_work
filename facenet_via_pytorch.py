#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
created Wednesday March 25, 2020
by James Hyde
modified from w_pytorch.py, refactored with functions to reduce redundancy

using pyTorch implementation of FaceNet
https://github.com/timesler/facenet-pytorch
https://pypi.org/project/facenet-pytorch/

***NOTE***
though they don't need to be imported, facenet_pytorch requires the following packages also be pip installed:

torch
torchvision

"""
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import timeit
import matplotlib.pyplot as plt
from scipy.stats import levene, f, ttest_ind

# image/video-processing related functions

def embed_array(network, image, output_image_size, save_processed = None):
    """
    :param image: opened image object
    :return: array of embedding for one single image
    """
    # NN to find face and crop to it
    mtcnn = MTCNN(image_size=output_image_size, margin=0)
    # find face and crop to it, also process as needed and save out if desired
    img_cropped = mtcnn(image, save_path = save_processed)
    # this caused some errors with the fake video
    try:
        # use NN to output embedding for this image
        img_embedding = network(img_cropped.unsqueeze(0))
        # "detach" this embedding from the torch object
        embed_arr = img_embedding.detach()
        # convert to numpy array
        embed_arr = np.array(embed_arr)
        return embed_arr
    except:
        return None



def video_frame_embeddings(network, filepath, output_images_size):
    """
    :param network: Neural Net model to use to create embeddings
    :param filepath: file name and location of the video to parse
    :param output_images_size: size of cropped/processed face image to use
    :return: tuple of array of individual frames' face embeddings and time elapsed in processing
    """
    vidpath = filepath
    start = timeit.default_timer()
    count = -1
    # This loads the video into opencv
    cap = cv2.VideoCapture(vidpath)
    # This clause will remain true until there are no more frames
    while (cap.isOpened()):
        # This reads in the frame and tells you if there is a valid frame to read (ret)
        ret, frame = cap.read()
        if ret is True:
            # add one to count of frames
            count += 1
            # use custom function (above) to output vector for face in this frame
            new_embed = embed_array(network, frame, output_images_size)
            # if this is the first frame, instantiate array
            if count == 0:
                holding = new_embed
            else:
                try:
                    # add this vector to existing array of vectors
                    holding = np.concatenate((holding, new_embed), axis=0)
                except:
                    print('Frame ' + str(count) + ' did not work for some reason. Its embedding is:')
                    print(new_embed)

            # This is about frame delay or something of the sort, don't remember full specifics, but you can look it up
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # This releases all the frames from memory and gets rid of any shown window
    cap.release()
    cv2.destroyAllWindows()
    # stop the timer
    time_real = timeit.default_timer() - start
    # return array of embeddings and time elapsed
    return (holding, time_real)


#### Distance-related functions

def cos_sim(vec1, vec2):
    """
    takes in two vectors and returns cosine similarity of them
    """
    cs = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cs

def l2_calc(vec1, vec2):
    dist = np.linalg.norm(vec1 - vec2)
    return dist

def l2_square(vec1, vec2):
    dist = np.square(np.linalg.norm(vec1 - vec2))
    return dist

def vector_dist(array_of_vecs, type, ref_point):
    """
    :param array_of_vecs: array of vectors from which to calculate mean vector
    :param type: distance metric to use
    :param ref_point: whether should be external reference vector or mean of array_of_vecs
    :return: distance metric of choice of each face embedding vector, compared to the chosen reference vector

    """
    if ref_point == 'mean':
        meanv = np.mean(array_of_vecs, axis = 0)
        refv = meanv
    else:
        refv = ref_point
    if type == "cos_sim":
        all_cos_sims = np.apply_along_axis(cos_sim, 1, array_of_vecs, vec2 = refv)
        return all_cos_sims
    if type == "l2":
        all_l2 = np.apply_along_axis(l2_calc, 1, array_of_vecs, vec2=refv)
        return all_l2
    if type == "squarel2":
        all_l2_square = np.apply_along_axis(l2_square, 1, array_of_vecs, vec2=refv)
        return all_l2_square

##### Plot-related functions

def plot_two_hists(data1, lab1, data2, lab2, title):
    """
    takes in two one-d vectors of data, a label for each, and a title, and plots two overlaid histograms
    """
    fig, ax = plt.subplots()
    ax.hist(data1, alpha=0.7, label=lab1)
    ax.hist(data2, alpha=0.7, label=lab2)
    ax.legend()
    fig.suptitle(title)
    fig.show



### SINGLE IMAGE EMBEDDING

# original image
img = Image.open('/Users/jameshyde/PYF_media/Obama_image.jpg')

# instantiate pretrained model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# set size of processed image to be embedded
im_size = 160

# get embedding for original image using custom function
single_image_embed = embed_array(resnet, img, im_size, '/Users/jameshyde/PYF_media/processed_image.jpg')

# save single vector to .npy file
with open('/Users/jameshyde/PYF_media/faceimage_vector.npy', 'wb') as f:
    np.save(f, single_image_embed)


### IMAGE EMBEDDING for REAL VIDEO

video = '/Users/jameshyde/PYF_media/Obama_real_trim.mp4'
real_vecs, real_time = video_frame_embeddings(resnet, video, im_size)

# elapsed time in minutes
real_time/60

# took 7.79 min for 35 seconds of video which yielded 1061 vectors
real_vecs.shape

# save out array as numpy file
with open('/Users/jameshyde/PYF_media/real_face_vectors.npy', 'wb') as f:
    np.save(f, real_vecs)

### IMAGE EMBEDDING for DEEPFAKE VIDEO

# Path to deepfake video
fvideo = '/Users/jameshyde/PYF_media/Obama_deepfake_trim.mp4'

fake_vecs, fake_time = video_frame_embeddings(resnet, fvideo, im_size)

# elapsed time in minutes
fake_time/60

fake_vecs.shape
# only resulted in 811 vectors for the same amount of time, 35 seconds -- different frame rate, 24 vs 30

# save out array as numpy file
with open('/Users/jameshyde/PYF_media/deepfake_face_vectors.npy', 'wb') as f:
    np.save(f, fake_vecs)



### measurements of variance using Euclidian Distance ###

# reshape single image's embedding to be directly comparable to vectors produced by np.apply_along_axis
ref_image = single_image_embed.reshape(512,)

# vector of euclidean distances for each frame of a real video, as compared to mean vector from that video
ed_real_mean = vector_dist(real_vecs, 'l2', 'mean')

# vector of euclidean distances for each frame of a deepfake video, as compared to mean vector from that video
ed_fake_mean = vector_dist(fake_vecs, 'l2', 'mean')

# plot these two vectors of distances as overlaid histograms
plot_two_hists(ed_real_mean, 'real', ed_fake_mean, 'fake', 'Euclidian between Frames of a Video and the Mean Vector')

# repeat above using reference image instead of mean vector
ed_real_ref = vector_dist(real_vecs, 'l2', ref_image)
ed_fake_ref = vector_dist(fake_vecs, 'l2', ref_image)
plot_two_hists(ed_real_ref, 'real', ed_fake_ref, 'fake', 'Euclidian between Frames of a Video and a Reference Photo')

# variance of each vector of distances
np.var(ed_real_mean)
np.var(ed_fake_mean)

# Brown-Forsythe test of equal variance for two non-normal distributions
levene(ed_real_mean, ed_fake_mean, center = 'median')
# p-value of 0.0358

# since variances are not equal, Welch's t-test of equal means for that case
ttest_ind(ed_real_mean, ed_fake_mean, equal_var=False)
# p-value of 2.38 * 10^-47

np.mean(ed_real_mean)
np.mean(ed_fake_mean)


### Create other video embeddings

video = '/Users/jameshyde/PYF_media/part_cordin.mp4'
out_vecs, out_time = video_frame_embeddings(resnet, video, im_size)

out_vecs.shape
out_time/60

with open('/Users/jameshyde/PYF_media/bp_spill_vecs.npy', 'wb') as f:
    np.save(f, out_vecs)
