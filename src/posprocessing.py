import numpy as np
import matplotlib.pyplot as plt
import cv2 
import matplotlib.colors as mcolors
from random import sample

def create_similarity_transformation_matrix(scale, theta, tx, ty):
    H = np.array([[scale * np.cos(theta), -scale * np.sin(theta), tx],
                  [scale * np.sin(theta),  scale * np.cos(theta), ty],
                  [0                    ,  0                    , 1]])
    return H

def warp_from_similarity_transformation(similarity_transformation_matrix, image):
    """
    Warps a given image using the parameters of a similarity transformation.
    Maintains the original image shape
    """
    warped_image = cv2.warpPerspective(image, similarity_transformation_matrix, (1024,1024), borderValue=(0,0,0))
    return warped_image

# DEBUG METHODS

def crop_registered_images(warped_latent, reference_image, bounding_box):
    """
    Crops warped latent and reference image. We assume that both images are in the same coordinate system.
    First we extract the minimum bounding box of warped_latent and use that to crop both images.
    """
    xmin, xmax, ymin, ymax = bounding_box
    cropped_latent    = warped_latent[xmin:xmax, ymin:ymax]
    cropped_reference = reference_image[xmin:xmax, ymin:ymax]

    return cropped_latent, cropped_reference



def extract_minimum_bounding_box(image):
    binarized_image              = np.where(image > 200, 255, 0)
    x_coordinates, y_coordinates = np.where(binarized_image == 255)
    xmin, xmax     = np.min(x_coordinates), np.max(x_coordinates)
    ymin, ymax     = np.min(y_coordinates), np.max(y_coordinates)
    
    return (xmin, xmax, ymin, ymax)

def warpTwoImages(img1, img2, H):
    """
    Applies the homografy matrix H. Author: Ilan
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2 = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2), axis=0)

    [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1, 0, t[0]],
                   [0, 1, t[1]],
                   [0, 0,   1]])

    thr = 16
    if (xmax-xmin) * (ymax-ymin) > h1 * w1 * thr or (xmax-xmin) * (ymax-ymin) > thr * h2 * w2:
        raise ValueError("Warping exceeds expected image dimension limit")

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin), borderValue=(0,0,0))
    result2 = np.ones_like(result) * 0
    result2[t[1]:h1+t[1], t[0]:w1+t[0]] = img1
    return result, result2, t


def debug_plot_correspondence_images(warped_latent, warped_reference, bounding_box, scale, theta, tx, ty, t, inliers, save_dir):
    """
    Plots latent and reference image on the same coordinate system. Also plots the correspodent homologous points
    """

    # Inverting both images for better visualization
    warped_latent    = np.where(warped_latent > 200, 0, 255)
    warped_reference = np.where(warped_reference > 200, 0, 255)


    # x_latent, y_latent       = zip(*latent_mnts)
    # x_reference, y_reference = zip(*reference_mnts)

    # x_latent = list(x_latent)
    # y_latent = list(y_latent)
    # x_reference = list(x_reference)
    # y_reference = list(y_reference)

    # Unpacking inliers from latent and from reference
    x_inliers_latent, y_inliers_latent       = zip(*[item[0] for item in inliers])
    x_inliers_reference, y_inliers_reference = zip(*[item[1] for item in inliers])

    x_inliers_latent = list(x_inliers_latent)
    y_inliers_latent = list(y_inliers_latent)
    x_inliers_reference = list(x_inliers_reference)
    y_inliers_reference = list(y_inliers_reference)

    # Creating image
    fig, ax = plt.subplots()
    w,h = warped_latent.shape

    # Color scheme to superimpose images
    debug_image = np.zeros((w,h,3))
    debug_image[:,:,0] = warped_reference
    debug_image[:,:,1] = warped_reference
    debug_image[:,:,2] = warped_reference

    xmin, xmax, ymin, ymax = bounding_box

    debug_image[xmin:xmax, ymin:ymax,0] = warped_latent[xmin:xmax, ymin:ymax] * 0.75 + warped_reference[xmin:xmax, ymin:ymax] * 0.25
    debug_image[xmin:xmax, ymin:ymax,1] = (warped_latent[xmin:xmax, ymin:ymax] + warped_reference[xmin:xmax, ymin:ymax])/2
    debug_image[xmin:xmax, ymin:ymax,2] = warped_reference[xmin:xmax, ymin:ymax]* 0.75 + warped_latent[xmin:xmax, ymin:ymax] * 0.25

    debug_image = debug_image.astype(np.uint8)
    ax.imshow(debug_image)

    # for i in range(len(x_latent)):
    #     aux_x = scale * np.cos(theta) * x_latent[i] - scale * np.sin(theta) * y_latent[i] + tx + t[0]
    #     aux_y = scale * np.sin(theta) * x_latent[i] + scale * np.cos(theta) * y_latent[i] + ty + t[1]
    #     x_latent[i] = (aux_x)
    #     y_latent[i] = (aux_y)
    
    # for i in range(len(x_reference)):
    #     aux_x = x_reference[i] + t[0]
    #     aux_y = y_reference[i] + t[1]
    #     x_reference[i] = (aux_x)
    #     y_reference[i] = (aux_y)
    
    # Transforming inliers according to the found transformation
    for i in range(len(x_inliers_latent)):
        aux_x = scale * np.cos(theta) * x_inliers_latent[i] - scale * np.sin(theta) * y_inliers_latent[i] + tx + t[0]
        aux_y = scale * np.sin(theta) * x_inliers_latent[i] + scale * np.cos(theta) * y_inliers_latent[i] + ty + t[1]
        x_inliers_latent[i] = (aux_x)
        y_inliers_latent[i] = (aux_y)
    
    for i in range(len(x_inliers_reference)):
        aux_x = x_inliers_reference[i] + t[0]
        aux_y = y_inliers_reference[i] + t[1]
        x_inliers_reference[i] = (aux_x)
        y_inliers_reference[i] = (aux_y)

    available_colors = mcolors.CSS4_COLORS
    # match_colors     = sample(list(available_colors.values()), len(x_inliers_latent))

    # Plotting final image
    #ax.scatter(x_latent, y_latent, color = 'orange', s = 20, marker = 'o')
    ax.scatter(x_inliers_latent, y_inliers_latent, color = 'blue', s = 20, marker = '*')
    #ax.scatter(x_reference, y_reference, color = 'red', s = 20, marker = 'o')
    ax.scatter(x_inliers_reference, y_inliers_reference, color = 'red', s = 20, marker = '*')
    ax.axis('off')

    plt.savefig(save_dir)