from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
from algorithm import inpainter


def get_corrupted_image(image, corrupted_pct):

    """
    :param image: the 3-dimensional numpy RGB array that represents an image
    :param corrupted_pct: percentage of the image to be corrupted
    :return: the corrupted image where corrupted_pct of total pixels are black
    """

    rng = np.random.default_rng(seed=42) # we reset the seed each time in order to have the same transformation A

    n_pixels = np.prod(image.shape[:2]) # the total number of pixels

    n_corrupted_pixels = int(n_pixels * corrupted_pct) # the number of corrupted pixels

    corrupted_pixels = np.array([0]*n_corrupted_pixels + [1]* (n_pixels - n_corrupted_pixels))

    rng.shuffle(corrupted_pixels)


    # reshape the pixels in a 2D array and then do it for all the three layers

    corrupted_pixels = corrupted_pixels.reshape(image.shape[:2])

    corrupted_pixels_3d = np.repeat(corrupted_pixels[:, :, None], repeats = 3, axis=2)

    # lastly we corrupt the image

    corrupted_image = np.multiply(image, corrupted_pixels_3d)

    return corrupted_image

def transform_tensor(Y, which_axis):

    """
    :param Y: the 3-dimensional numpy RGB array that represents an image
    :param which_axis: if 0 we are transforming Y to a 2D array of size (N, 3*M), else of size (3*N, M) i.e. the transpose
    :return: the transformed 2D numpy array
    """

    if which_axis == 0:

        return Y.reshape(Y.shape[0], Y.shape[1]*3)

    else:

        return Y.reshape(Y.shape[0] * 3 , Y.shape[1])

def un_transform_tensor(Y, which_axis):

    """
    Same parameters as in transform_tensor but now we reverse the transformation
    """

    if which_axis == 0:

        return Y.reshape(Y.shape[0], Y.shape[1] // 3, 3 )

    else:

        return Y.reshape(Y.shape[0] // 3, Y.shape[1], 3)

def nuclear_norm(Y):

    """
    :param Y: the 3-dimensional numpy RGB array that represents an image
    :return: the nuclear norm
    """

    # the nuclear norm can only be calculated on a 2D matrix so we need to transform the array on both axis

    return np.linalg.norm(transform_tensor(Y, 0), ord = 'nuc') + np.linalg.norm(transform_tensor(Y, 1), ord = 'nuc')

def prox_nuclear_norm(Y, gamma):

    """
    :param Y: the array created using the transform_tensor function
    :param gamma: the gamma parameter attached to the function f and g
    :return: the proximal of the nuclear norm
    """

    U, S, V_t = np.linalg.svd(Y, full_matrices=False)

    return (U * np.maximum(S-gamma, 0)) @ V_t




def sensitivity_analysis(which_analysis):

    """
    This function performs sensitivity analysis on the percentage of erased pixels, gamma and sigma
    """

    figura = np.array(Image.open('artu.jpg'))

    figura = figura.astype(np.float32) / 255.0


    if which_analysis == 'erased':

        pct_list = np.linspace(0.1, 1, 10)

        n_iters_list = []

        time_list= []

        for pct in pct_list:

            print(pct)

            corr_image = get_corrupted_image(figura, pct)

            t0 = time.time()

            _, n_its = inpainter(corr_image, 100, 1e-3, 1, 1, pct)

            time_list.append(time.time() - t0)

            n_iters_list.append(n_its)


        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,6))

        ax[0].plot(pct_list, n_iters_list, color="orange", linestyle = 'solid')
        ax[0].plot(pct_list, [100] * 10, color="black", linestyle = '--')
        ax[1].plot(pct_list, time_list, color="orange", linestyle = 'solid')


        ax[0].set_xlabel('% erased pixels')
        ax[1].set_xlabel('% erased pixels')
        ax[0].set_ylabel('Iterations')
        ax[1].set_ylabel('Time (s)')
        plt.show()


    elif which_analysis == 'gamma':

        gammas = np.linspace(0.1, 1.9, 19)

        n_iters_list = []

        time_list = []

        for gamma in gammas:
            corr_image = get_corrupted_image(figura, 0.5)

            t0 = time.time()

            _, n_its = inpainter(corr_image, 100, 1e-3, gamma, 1, 0.5)

            time_list.append(time.time() - t0)

            n_iters_list.append(n_its)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

        ax[0].plot(gammas, n_iters_list, color="orange", linestyle = 'solid')
        ax[0].plot(gammas, [100] * 19, color="black", linestyle = '--')
        ax[1].plot(gammas, time_list, color="orange", linestyle = 'solid')

        ax[0].set_xlabel(r'$\gamma$')
        ax[1].set_xlabel(r'$\gamma$')
        ax[0].set_ylabel('Iterations')
        ax[1].set_ylabel('Time (s)')
        plt.tight_layout()
        plt.show()


    elif which_analysis == 'sigma':

        sigmas = [0.0625, 0.25, 0.5, 1, 4, 16]

        sigmas = [0,0,0,0,0,0]

        restored_images = []

        corr_image = get_corrupted_image(figura, 0.5)

        for sigma in sigmas:

            print(sigma)

            restored_image, _ = inpainter(corr_image, 100, 1e-3, 1, sigma, 0.5)

            restored_images.append(restored_image)


        fig, axs = plt.subplots(nrows = 2, ncols = 4,  figsize=(12, 6))

        axs[0, 0].title.set_text("Original")
        axs[0, 0].imshow(figura)
        axs[0, 0].set_axis_off()

        axs[0, 1].title.set_text("Corrupt")
        axs[0, 1].imshow(corr_image)
        axs[0, 1].set_axis_off()

        axs[0, 2].title.set_text(r"$\sigma=0.0625$")
        axs[0, 2].imshow(restored_images[0])
        axs[0, 2].set_axis_off()

        axs[0, 3].title.set_text(r"$\sigma=0.25$")
        axs[0, 3].imshow(restored_images[1])
        axs[0, 3].set_axis_off()

        axs[1, 0].title.set_text(r"$\sigma=0.5$")
        axs[1, 0].imshow(restored_images[2])
        axs[1, 0].set_axis_off()


        axs[1, 1].title.set_text(r"$\sigma=1$")
        axs[1, 1].imshow(restored_images[3])
        axs[1, 1].set_axis_off()


        axs[1, 2].title.set_text(r"$\sigma=4$")
        axs[1, 2].imshow(restored_images[4])
        axs[1, 2].set_axis_off()


        axs[1, 3].title.set_text(r"$\sigma=16$")
        axs[1, 3].imshow(restored_images[5])
        axs[1, 3].set_axis_off()


        plt.show()
