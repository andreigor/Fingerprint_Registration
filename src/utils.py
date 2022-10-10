import os
import numpy as np
import matplotlib.pyplot as plt

def create_debug_images_directory(experiment_directory):
    """
    Creates the debug directory inside the experiment directory.
    Returns the debug directory path
    """
    debug_images_directory = experiment_directory + '/debug_images'
    os.makedirs(debug_images_directory, exist_ok=True)
    return debug_images_directory

def create_registered_images_directory(experiment_directory):
    """
    Creates the registered images directory inside the experiment directory.
    Returns the registered images directory path
    """
    registered_images_directory = experiment_directory + '/registered_images'
    os.makedirs(registered_images_directory, exist_ok=True)
    return registered_images_directory

def create_register_results_list(experiment_directory):
    """
    Creates the register results list inside the directory experiment.
    Returns the file object in writing mode
    """
    register_result_list = open(experiment_directory + '/register_results.txt', 'w')
    return register_result_list

def read_mnt_file(mnt_file):
    with open(mnt_file, 'r') as fp:
        mnts = fp.readlines()[2:]
    mnts = [tuple([int(i) for i in item.strip().split(' ')[:2]]) for item in mnts]
    return mnts


def _plot_rotation_histogram(rotation_histogram):
    fig, ax = plt.subplots(figsize = (15,6))
    ax.plot(rotation_histogram, linewidth = 2.8)
    ax.set_title('Rotation angle histogram', fontsize = 18, fontweight = 'bold')
    ax.set_xlabel(r'$\theta$', fontsize = 16)
    ax.set_ylabel('Distribution', fontsize = 16, fontweight = 'bold')
    ax.tick_params(axis = 'both', labelsize = 15)

    theta_max = np.argmax(rotation_histogram)
    anchored_text = AnchoredText(r'$\theta_{max} = $' + f'{theta_max}', loc=2)
    ax.add_artist(anchored_text)
    ax.grid(True)
    plt.show()

def _plot_translation_histogram(translation_histogram):
        
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    d = 512
    # Make data.
    X = np.arange(-d, d + 1, 1)
    Y = np.arange(-d, d + 1, 1)
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, translation_histogram, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


