from itertools import combinations
from random import shuffle

import numpy as np
from utils import _plot_rotation_histogram, _plot_translation_histogram


DEBUG = False

class MaxIterReached(Exception):
    "Raised when reaches max number of iterations"
    pass

def ransac_similarity_transformation(latent_pts, reference_pts):
    """
    Finds the parameters of a similarit transformation between the latent and reference points. If there is no correspondence between
    the two points, it return 0 in all parameters and an error flag. Otherwise, returns the found parameters s,tx, ty and theta.
    
    Input: latent minutia, reference minutia - candidate to homologous points
    Output: parameters of a similarity transformation tx, ty, theta and an status flag
    
    For further reference, please check section 5.2.2 of the book Theory and Applications of Image Registration - Goshtasby 2017
    """
    max_pixel_distance_tolerance          = 4
    number_of_iterations                  = 0
    minimum_fraction_of_homologous_points = 0.25
    MAX_ITER                              = get_max_ransac_iter_from_confidence(0.99, len(reference_pts))
    epsilon                               = 0.5

    # selecting pairs of points from latent and from reference
    latent_pairs    = [comb for comb in combinations(latent_pts, 2)]
    reference_pairs = [comb for comb in combinations(reference_pts, 2)]

    # randomly shuffling points and lines

    shuffle(latent_pts)
    shuffle(reference_pts)
    shuffle(latent_pairs)
    shuffle(reference_pairs)

    for (l1,l2) in latent_pairs:
        for (R1,R2) in reference_pairs:
            if number_of_iterations%10000 == 0:
                print('Ransac progress: iteration {}/{}'.format(number_of_iterations+1, MAX_ITER))
            # HIPOTHESIS STAGE
            # determining the scale parameter
            latent_distance    = np.sqrt((l1[0] - l2[0])*(l1[0] - l2[0]) + (l1[1] - l2[1])*(l1[1] - l2[1])) # euc distance of latent
            reference_distance = np.sqrt((R1[0] - R2[0])*(R1[0] - R2[0]) + (R1[1] - R2[1])*(R1[1] - R2[1])) # euc distance of ref
            scale              = reference_distance/latent_distance
            
            # incrementing iterations
            number_of_iterations += 1

            if (1 - epsilon) > scale or scale > (1 + epsilon): # failed distance condition
                    continue # go to next iteration

            # determining the angle between the two lines
            latent_vector    = (l2[0] - l1[0], l2[1] - l1[1])
            reference_vector = (R2[0] - R1[0], R2[1] - R1[1])
            theta = _float_angle_between_vectors(latent_vector, reference_vector)

            # getting mid-point of l1l2 and R1R2; x' = (x1 + x2)/2; y' = (y1 + y2)/2
            latent_mid_point    = ((l2[0] + l1[0])/2 , (l2[1] + l1[1])/2)  
            reference_mid_point = ((R2[0] + R1[0])/2 , (R2[1] + R1[1])/2)

            # finding tx and ty from homologous points (mid latent and mid reference)
            tx = reference_mid_point[0] - scale * (latent_mid_point[0] * np.cos(theta) - latent_mid_point[1] * np.sin(theta))
            ty = reference_mid_point[1] - scale * (latent_mid_point[0] * np.sin(theta) + latent_mid_point[1] * np.cos(theta))

            # VERIFICATION STAGE
            number_of_homologous_points   = 0
            inliers = []

            # print('#################################')
            for (x,y) in latent_pts:
                # finding the coordinates of transformed latent point
                X_hat = scale * x * np.cos(theta) - scale * y * np.sin(theta) + tx
                Y_hat = scale * x * np.sin(theta) + scale * y * np.cos(theta) + ty
                
                # finding the best correspondance in the reference minutias
                distances    = np.zeros((len(reference_pts,)))
                for i, (X,Y) in enumerate(reference_pts):
                    distances[i] = np.sqrt((X_hat - X)*(X_hat - X) + (Y_hat - Y)*(Y_hat - Y))
                
                best_correspondent_distance = np.min(distances)
                index = np.argmin(distances)

                if best_correspondent_distance < max_pixel_distance_tolerance:
                    inliers.append(((x, y),(reference_pts[(index)][0], reference_pts[(index)][1])))
                    number_of_homologous_points += 1
                
                if number_of_homologous_points/len(latent_pts) > minimum_fraction_of_homologous_points:
                    return scale, theta, tx, ty, inliers
            
            if number_of_iterations > MAX_ITER:
                print('Ransac failed! Max number of iterations reached')
                return -1, -1, -1, -1, -1, -1, -1, -1 # algorithm failed

def get_max_ransac_iter_from_confidence(confidence, number_of_points_in_reference):
    """
    Check section 5.2.2 of the book Theory and Applications of Image Registration - Goshtasby 2017 to understand this expression
    """
    max_iter = np.log(1 - confidence)/(np.log(1 - 1/(4 * number_of_points_in_reference * number_of_points_in_reference)))
    return int(max_iter) + 1

def clustering_rigid_transformation(latent_pts, reference_pts):
    """
    Finds the parameters of a rigid transformation between the latent and reference points. If there is no correspondence between
    the two points, it return 0 in all parameters and an error flag. Otherwise, returns the found parameters tx, ty and theta.
    
    Input: latent minutia, reference minutia - candidate to homologous points
    Output: parameters of a rigid transformation tx, ty, theta and an status flag
    
    For further reference, please check section 5.2.1 of the book Theory and Applications of Image Registration - Goshtasby 2017
    """
    
    # first estimate the rotational difference between the two sets of points
    theta = _estimate_rotational_difference(latent_pts, reference_pts)
    tx, ty        = _estimate_translational_difference(latent_pts, reference_pts, theta)

    return (2, tx, ty)

def _estimate_rotational_difference(latent_pts, reference_pts):
    """
    Determining rotational diﬀerence between two point sets by clustering
    """
    rotation_histogram = np.zeros((181,))
    epsilon            = 0.01

    latent_pairs    = [comb for comb in combinations(latent_pts, 2)]
    reference_pairs = [comb for comb in combinations(reference_pts, 2)]

    # randomly shuffling points and lines
    shuffle(latent_pts)
    shuffle(reference_pts)
    shuffle(latent_pairs)
    shuffle(reference_pairs)


    iter = 0
    MAX_ITER = len(latent_pts) * len(reference_pts) * len(reference_pts) # assumes that half of the mnts in latent appear in reference

    try:
        for latent_pair in latent_pairs:
            for reference_pair in reference_pairs:
                l1, l2 = latent_pair
                R1, R2 = reference_pair

                # calculating ratio between distances of points
                latent_distance    = np.sqrt((l1[0] - l2[0])*(l1[0] - l2[0]) + (l1[1] - l2[1])*(l1[1] - l2[1])) # euc distance of latent
                reference_distance = np.sqrt((R1[0] - R2[0])*(R1[0] - R2[0]) + (R1[1] - R2[1])*(R1[1] - R2[1])) # euc distance of ref
                distance_ratio     = latent_distance/reference_distance

                if (1 - epsilon) > distance_ratio or distance_ratio > (1 + epsilon): # failed distance condition
                    continue # go to next iteration
                    
                # calculating angle between the latent vector and the reference vector
                latent_vector    = (l2[0] - l1[0], l2[1] - l1[1])
                reference_vector = (R2[0] - R1[0], R2[1] - R1[1])
                theta = _integer_angle_between_vectors(latent_vector, reference_vector)

                # incrementing histogram at the calculated theta
                rotation_histogram[theta] += 1

                # checking if reached max number of iterations
                iter = iter + 1
                if iter >= MAX_ITER:
                    raise MaxIterReached('Max number of iterations achieved')

    except MaxIterReached:
        pass

    
    if DEBUG:
        _plot_rotation_histogram(rotation_histogram)

    estimated_theta = np.argmax(rotation_histogram)

    return estimated_theta

def _estimate_translational_difference(latent_pts, reference_pts, theta):
    """
    Determining translational diﬀerence between two point sets by clustering
    """
    d = 512
    translation_histogram = np.zeros((2*d + 1, 2*d + 1))
    iter = 0

    MAX_ITER = len(latent_pts) * len(reference_pts) / 2

    theta = theta / 180 * np.pi # deg to rad
    try:
        for (x,y) in latent_pts:
            for (X,Y) in reference_pts:
                tx = X - x * np.cos(theta) + y * np.sin(theta)
                ty = Y - x * np.sin(theta) - y * np.cos(theta)

                if (-d <= tx) and (tx < d) and (-d <= ty) and (ty < d):
                    translation_histogram[int(tx) + d, int(ty) + d] += 1
                else:
                    continue # go to next iteration
                
                # checking if reached max number of iterations
                iter = iter + 1
                if iter >= MAX_ITER:
                    raise MaxIterReached('Max number of iterations achieved')

    except MaxIterReached:
        pass

    

    # getting the max coordinates from the histogram
    max_coords = np.where(translation_histogram == np.amax(translation_histogram))
    tx, ty     = max_coords[0][0], max_coords[1][0]
    tx, ty     = (tx - d, ty - d)
    
    if DEBUG:
            _plot_translation_histogram(translation_histogram)

    return tx, ty



def _integer_angle_between_vectors(latent_vector, reference_vector):
    unit_latent_vector    = latent_vector / np.linalg.norm(latent_vector)
    unit_reference_vector = reference_vector / np.linalg.norm(reference_vector)

    dot_product = np.dot(unit_latent_vector, unit_reference_vector)

    angle = np.arccos(np.round(dot_product, 4))
    angle = int(angle/np.pi * 180)

    return angle

def _float_angle_between_vectors(latent_vector, reference_vector):
    unit_latent_vector    = latent_vector / np.linalg.norm(latent_vector)
    unit_reference_vector = reference_vector / np.linalg.norm(reference_vector)

    dot_product = np.dot(unit_latent_vector, unit_reference_vector)

    angle = np.arccos(np.round(dot_product, 4))

    return angle





    
