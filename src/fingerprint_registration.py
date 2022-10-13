"""
Unicamp - 10-04-2022
Author: André Igor Nóbrega da Silva
email: a203758@dac.unicamp.br
Image registration module as a part of end-to-end latent identification system. This is a Msc. project developed in Unicamp
in partnership with Griaule.

Input: list with pairs latent/reference pairs to be registered, input image folder, files containing minutia locations
Output: list with registration score and status, output image folder
"""

import sys
import time

from PIL import Image
import cv2 as cv

from utils import *
from registration_methods import *
from posprocessing import *

DEBUG = True

class InputParameterError(Exception):
    "Raised when input parameter is not as expected"
    pass

def main():
    # input parameters
    if len(sys.argv) != 5:
        message = 'Error in fingerprint_registration.py:\n '
        message += '<P1> <P2> <P3>\n'
        message += 'P1: comparisons .txt file\n'
        message += 'P2: input folder with all fingerprint images\n'
        message += 'P3: input folder with all minutia files\n'
        message += 'P4: output folder with registered images and score li   st\n'
        raise InputParameterError(message)
    tic = time.time()

    # reading input_parameters
    input_comparisons_file = open(sys.argv[1], 'r').readlines()
    input_image_folder     = sys.argv[2]
    input_mnts_folder      = sys.argv[3]

    # creating output directories and result list file
    experiment_directory        = '../outputs/' + sys.argv[4]
    registered_images_directory = create_registered_images_directory(experiment_directory)
    debug_images_directory      = create_debug_images_directory(experiment_directory)
    register_result_list        = create_register_results_list(experiment_directory)


    # Starting register algorithm
    for i, line in enumerate(input_comparisons_file):
        print("Progress: {}/{} \r".format(i + 1, len(input_comparisons_file)))
        # reading input pair
        latent, reference = line.strip().split(' ')
        
        # reading input mnt files
        latent_mnt_file    = input_mnts_folder + '/' + latent.replace('png', 'mnt')
        reference_mnt_file = input_mnts_folder + '/' + reference.replace('png', 'mnt')

        latent_mnts    = read_mnt_file(latent_mnt_file)
        reference_mnts = read_mnt_file(reference_mnt_file)

        # reading input images
        reference_img = np.array(Image.open(input_image_folder + reference))
        latent_img    = np.array(Image.open(input_image_folder + latent))


        # calculating parameters of similarity transformation from minutia points
        tic = time.time()
        #scale, theta, tx, ty, inliers = brute_force_ransac_similarity_transformation(latent_mnts, reference_mnts)
        scale, theta, tx, ty, inliers = descriptor_based_ransac_similarity_transformation(latent_mnts, reference_mnts)
        # scale, theta, tx, ty, inliers = clustering_rigid_transformation(latent_mnts, reference_mnts)
  


        
        if scale == -1:
            print('Algorithm failed in step {}'.format(i+1))
            continue

        # warping latent image
        H = create_similarity_transformation_matrix(scale, theta, tx, ty)
        # warped_latent = warp_from_similarity_transformation(H, latent_img) # this is probably better! Check with Falcao
        warped_latent, warped_reference, t = warpTwoImages(reference_img, latent_img, H)

        # cropping registered images
        bounding_box                      = extract_minimum_bounding_box(warped_latent)
        cropped_latent, cropped_reference = crop_registered_images(warped_latent, warped_reference, bounding_box)

        # saving cropped images
        cv.imwrite(registered_images_directory + '/' + latent + f'_{i + 1}', cropped_latent)
        cv.imwrite(registered_images_directory + '/' + reference + f'_{i + 1}', cropped_reference)

        if DEBUG:
            save_dir = debug_images_directory + f'/debug_{i + 1}.png'
            debug_plot_correspondence_images(warped_latent, warped_reference,bounding_box, scale, theta, tx, ty, t, inliers, save_dir, latent_mnts, reference_mnts)

    toc = time.time()
    print('Fingerprint registration algorithm finished with sucess in {:.2f}s'.format(toc - tic))

    

if __name__ == '__main__':
    main()