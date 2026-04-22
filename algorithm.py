from utils import *

def inpainter(corrupted_image, max_it, tol, gamma, sigma, corrupted_pct):

    """
    :param corrupted_image:
    :param max_it: maximum number of iterations
    :param tol: tolerance
    :param gamma: the gamma parameter attached to the function f and g
    :param sigma: the sigma parameter that multiplies the function f and g
    :param corrupted_pct: percentage of corrupted pixels
    :return: the reconstructed image
    """

    proxf = lambda Y: un_transform_tensor(prox_nuclear_norm(transform_tensor(Y, 0), gamma*sigma), 0)

    proxg = lambda Y: un_transform_tensor(prox_nuclear_norm(transform_tensor(Y, 1), gamma * sigma), 1)

    Y_prev = corrupted_image

    n_it = max_it

    reconstructed_image = corrupted_image

    def T(Y):

        Z = proxg(Y)

        return Y - Z + proxf(2*Z - Y - gamma * get_corrupted_image(Z-corrupted_image, corrupted_pct))


    for i in range(max_it):

        T_Y = T(Y_prev)


        # if Y_prev is the zero vector we are not doing anything so we skip the step

        if np.linalg.norm(Y_prev) == 0.0:

            continue

        if np.linalg.norm(T_Y - Y_prev) / np.linalg.norm(Y_prev) < tol:

            n_it = i

            reconstructed_image = T_Y

            break

        Y_prev = T_Y


    return reconstructed_image, n_it


