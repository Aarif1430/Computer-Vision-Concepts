# Computer Vision concepts 
## Image Histogram
- Distribution of different gray levels in an image and frequency of these levels.
```python
from matplotlib import pyplot as plt
import cv2
img = cv2.imread('data/Lenna.jpg')

for i, col in enumerate(['b', 'g', 'r']):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
```
Original Image             |  RGB Distribution
:-------------------------:|:-------------------------:
![](/data/Lenna.jpg)  |  ![](/data/lenna_rgb.png)

## Image Noise

- I(x,y) = The true pixel values
- n(x,y) = The noise at pixel (x,y)

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{I}(x,y)&space;=&space;I(x,y)&space;&plus;&space;n(x,y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{I}(x,y)&space;=&space;I(x,y)&space;&plus;&space;n(x,y)" title="\hat{I}(x,y) = I(x,y) + n(x,y)" /></a>
```python
mean = 0.0
std = 1.0
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gaussian = np.random.normal(mean, std, img_gray.shape)
sns.distplot(gaussian, fit=stats.norm, kde=True)
plt.show()
```

The pdf of gaussian noise:

<img src="https://latex.codecogs.com/gif.latex?\bg_white&space;n(x,y)=&space;e^{\frac{-n^2}{2\sigma^2}}" title="n(x,y)= e^{\frac{-n^2}{2\sigma^2}}" />

![](/data/gauss_pdf.png)

```python
rdn = random.random()
if rdn < prob:
    output[i][j] = 0
elif rdn > thres:
    output[i][j] = 255
else:
    output[i][j] = image[i][j]
```

Original Image             |  Noisy Image
:-------------------------:|:-------------------------:
![](/data/gray_img.png)  |  ![](/data/noisy_img.png)
