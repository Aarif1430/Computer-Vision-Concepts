from imports.imports import *

img = cv2.imread('../data/Lenna.jpg')
#
# for i, col in enumerate(['b', 'g', 'r']):
#     hist = cv2.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(hist, color=col)
#     plt.xlim([0, 256])
# plt.savefig('data/lenna_rgb.png')
# plt.show()

################### Add noise to Image ##########################
# Convert to gray scale

def sp_noise(image, prob):
    """
    Add salt and pepper noise to image
    prob: Probability of the noise
    """
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def plot_gaussian_2d():
    x = np.linspace(-10, 10, num=100)
    y = np.linspace(-10, 10, num=100)

    x, y = np.meshgrid(x, y)

    z = np.exp(-0.1 * x ** 2 - 0.1 * y ** 2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.jet)
    plt.savefig('../data/gauss_2d.png')
    plt.show()


mean = 0.0
var = 0.1
sigma = var**0.5
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('../data/gray_img.png', img_gray)
gaussian = np.random.normal(mean, sigma, img_gray.shape)
sns.distplot(gaussian, fit=stats.norm, kde=True)
plt.savefig('../data/gauss_pdf.png')
# Add noise to the image
new_img = img_gray + gaussian
noise_img = sp_noise(img_gray, 0.05)
cv2.imwrite('../data/noisy_img.png', noise_img)
plot_gaussian_2d()
