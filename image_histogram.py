from imports import *

img = cv2.imread('data/Lenna.jpg')
#
# for i, col in enumerate(['b', 'g', 'r']):
#     hist = cv2.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(hist, color=col)
#     plt.xlim([0, 256])
# plt.savefig('data/lenna_rgb.png')
# plt.show()

################### Add noise to Image ##########################
# Convert to gray scale
mean = 0.0
std = 1.0
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gaussian = np.random.normal(mean, std, img_gray.shape)
sns.distplot(gaussian, fit=stats.norm, kde=True)
plt.savefig('data/gauss_pdf.png')
plt.show()