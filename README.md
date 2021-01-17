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

