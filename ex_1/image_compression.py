from numpy import diag,zeros
from numpy.linalg import svd, norm
from matplotlib.pyplot import *
import scipy.misc as misc

IMAGE_SIZE = 512

def main():
    image = misc.ascent()  # image in size 512 * 512
    k = 0
    y_ratio = []
    y_distance = []

    while k < IMAGE_SIZE :

        # Use the SVD to zero all but the k largest singular-values
        # Reconstruct the rank-k approximation matrix

        compressed_image, s = compress_svd(image, k)

        if k in [16, 32, 64, 128, 256]:
            imshow(compressed_image, cmap='gray')
            title("k="+str(k))
            show()

        # For each k calculate the compression ratio

        image_rank = np.linalg.matrix_rank(image)
        ratio = compressing_ratio(IMAGE_SIZE, k, image_rank)
        y_ratio += [ratio]

        # For each k calculate the Frobenius distance between the original and the reconstructed
        # images. Use numpy.linalg.norm.

        distance = frobenius_distance(image, compressed_image)
        y_distance += [distance]
        k += 1


    # graph of the Frobenius distance a function of k

    plot(y_distance)
    title("Frobenius distance a function of k")
    ylabel("Frobenius distance")
    xlabel("k")
    show()

    # graph of  the compression ratio a function of k
    plot(y_ratio)
    title("compression ratio a function of k")
    ylabel("compression ratio")
    xlabel("k")
    show()


def frobenius_distance(x , y):
    """
    calculate the Frobenius distance between the original and reconstructed
    image
    :param x: original image
    :param y: reconstructed image
    :return:
    """
    return norm(x-y, 'fro')


def compressing_ratio(n, k, r):
    """
    calculate the compression ratio
    :param n:
    :param k:
    :param r:
    :return:
    """

    return (((2*k*n) + k)/((2*n*r) + r))


def compress_svd(image, k):
    """
    select k singular vectors and values
    Instead of storing m×n values for the original image,
    we store k(m+n)+k values
    :param image:
    :param k:
    :return: compressed_image matrix
             s array of singular values
    """
    u,s,v = svd(image, full_matrices=False)
    compressed_image = np.dot(u[:,:k], np.dot(diag(s[:k]), v[:k,:]))
    return compressed_image, s







main()