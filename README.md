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
  
  <img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\large&space;g(x,y)&space;=&space;e^{\frac&space;{-(x^2&space;&plus;&space;y^2)}{2&space;\sigma&space;^2}}" title="\large g(x,y) = e^{\frac {-(x^2 + y^2)}{2 \sigma ^2}}" />
  
  ![](/data/gauss_pdf.png) ![](/data/gauss_2d.png)  

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

## Image Derivatives & Averages

- ### Derivative in two dimensions

    <img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\large&space;\begin{array}{ll}&space;\text&space;{&space;Given&space;function&space;}&space;&&space;f(x,&space;y)&space;\\&space;\text&space;{&space;Gradient&space;vector&space;}&space;&&space;\nabla&space;f(x,&space;y)=\left[\begin{array}{l}&space;\frac{\partial&space;f(x,&space;y)}{\partial&space;x}&space;\\&space;\\&space;\frac{\partial&space;f(x,&space;y)}{\partial&space;y}&space;\end{array}\right]=\left[\begin{array}{l}&space;f_{x}&space;\\&space;f_{y}&space;\end{array}\right]&space;\\&space;\\&space;\text&space;{&space;Gradient&space;magnitude&space;}&space;&&space;|\nabla&space;f(x,&space;y)|=\sqrt{f_{x}^{2}&plus;f_{y}^{2}}&space;\\&space;\\&space;\text&space;{&space;Gradient&space;direction&space;}&space;&&space;\theta=\tan&space;^{-1}&space;\frac{f_{x}}{f_{y}}&space;\end{array}" title="\large \begin{array}{ll} \text { Given function } & f(x, y) \\ \text { Gradient vector } & \nabla f(x, y)=\left[\begin{array}{l} \frac{\partial f(x, y)}{\partial x} \\ \\ \frac{\partial f(x, y)}{\partial y} \end{array}\right]=\left[\begin{array}{l} f_{x} \\ f_{y} \end{array}\right] \\ \\ \text { Gradient magnitude } & |\nabla f(x, y)|=\sqrt{f_{x}^{2}+f_{y}^{2}} \\ \\ \text { Gradient direction } & \theta=\tan ^{-1} \frac{f_{x}}{f_{y}} \end{array}" />

- ### Correlation

    <img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\large&space;\begin{aligned}&space;&f&space;\otimes&space;h=\sum_{k}&space;\sum_{i}&space;f(k,&space;l)&space;h(i&plus;k,&space;j&plus;l)\\&space;&\begin{array}{l}&space;f=\text&space;{&space;Image&space;}&space;\\&space;h=\text&space;{&space;Kernel&space;}&space;\end{array}&space;\end{aligned}" title="\large \begin{aligned} &f \otimes h=\sum_{k} \sum_{i} f(k, l) h(i+k, j+l)\\ &\begin{array}{l} f=\text { Image } \\ h=\text { Kernel } \end{array} \end{aligned}" />

- ### Convolution
  
    <img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\large&space;f^{*}&space;h=\sum_{k}&space;\sum_{l}&space;f(k,&space;l)&space;h(i-k,&space;j-l)" title="\large f^{*} h=\sum_{k} \sum_{l} f(k, l) h(i-k, j-l)" />

    ![](/data/convolution.png)
  
- ## Convolution from scratch using numpy
  ```python
  import numpy as np
  def convolution_2d(image, kernel):
      flipped_kernel = np.fliplr(np.flipud(kernel))
      padded_image = np.zeros((image.shape[0]+2, image.shape[1]+2))
      padded_image[1:-1,1:-1] = image
      result = np.zeros(image.shape)
      x = kernel.shape[0]
      y = kernel.shape[1]
      for r in range(image.shape[0]-x):
              for c in range(image.shape[1]-y):
                  rc = kernel * padded_image[r:x+r, c:y+c]
                  result[r, c] = rc.sum()
      return result
  kernel = (1/25)*np.ones((15,15))
  print(kernel.shape)
  img = cv2.imread('../data/gray_img.png')
  img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  plt.imshow(convol2d(img_gray, kernel))
  plt.savefig('../data/blurred_image')
  ```
  
   ![](/data/blurred_image.png)

## Marr Hildreth Edge Detector

- ### Gaussian Smoothing
  
  - ### Laplacian 
    <img src="https://latex.codecogs.com/gif.latex?\Delta&space;^2&space;S&space;=&space;\frac&space;{\partial}{\partial{x^2}}(S)&space;&plus;&space;\frac&space;{\partial}{\partial{y^2}}(S)" title="\Delta ^2 S = \frac {\partial}{\partial{x^2}}(S) + \frac {\partial}{\partial{y^2}}(S)" />
  
  - ### Laplacian of Gaussian
    <img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\large&space;\Delta&space;^2&space;S&space;=&space;\Delta&space;^2&space;(g&space;*&space;I)&space;=&space;(\Delta&space;^2&space;g)*I" title="\large \Delta ^2 S = \Delta ^2 (g * I) = (\Delta ^2 g)*I" />
    <br>
    <img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;\large&space;\Delta&space;^2&space;g&space;=&space;\frac&space;{1}{\sqrt{2\pi}\sigma^3}[2&space;-&space;\frac&space;{x^2&space;&plus;&space;y^2}{\sigma^2}]e^{\frac{x^2&space;&plus;&space;y^2}{2\sigma^2}}" title="\large \Delta ^2 g = \frac {1}{\sqrt{2\pi}\sigma^3}[2 - \frac {x^2 + y^2}{\sigma^2}]e^{\frac{x^2 + y^2}{2\sigma^2}}" />

  - ### Separability
    ![](/data/Separability.png)

  ## Laplacian Gaussian Edge Detector
  - Computer LOG
  - Find Zero_Crossings from each row
  - Find slope of Zero-Crossing
  - Apply threshold to slope and mark edges
  
## Canny Edge Detector
  ### Steps
  - Smooth image with gaussian filter
    
    <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\fn_jvn&space;S&space;=&space;I&space;*&space;g(x,y)&space;=&space;g(x,y)*I;&space;\thickspace&space;where,&space;g(x,y)&space;=&space;\frac&space;{1}{\sqrt{2\pi\sigma}}&space;e&space;^-({\frac&space;{x^2&plus;y^2}{2\sigma^2}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;S&space;=&space;I&space;*&space;g(x,y)&space;=&space;g(x,y)*I;&space;\thickspace&space;where,&space;g(x,y)&space;=&space;\frac&space;{1}{\sqrt{2\pi\sigma}}&space;e&space;^-({\frac&space;{x^2&plus;y^2}{2\sigma^2}})" title="S = I * g(x,y) = g(x,y)*I; \thickspace where, g(x,y) = \frac {1}{\sqrt{2\pi\sigma}} e ^-({\frac {x^2+y^2}{2\sigma^2}})" /></a>
    
  - Compute derivative of filtered image
    
    <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\fn_jvn&space;\nabla&space;=&space;\nabla&space;(g*I)&space;=&space;(\nabla&space;g)&space;*&space;I" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;\nabla&space;=&space;\nabla&space;(g*I)&space;=&space;(\nabla&space;g)&space;*&space;I" title="\nabla = \nabla (g*I) = (\nabla g) * I" /></a>

  - Find magnitude and orientation of gradient
  
    <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\fn_jvn&space;magnitude&space;=&space;\sqrt&space;{{S_x}^2&space;&plus;&space;{S_y}^2}&space;\\&space;direction&space;=&space;\theta&space;=&space;\tan^{-1}&space;\frac{S_y}{S_x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\fn_jvn&space;magnitude&space;=&space;\sqrt&space;{{S_x}^2&space;&plus;&space;{S_y}^2}&space;\\&space;direction&space;=&space;\theta&space;=&space;\tan^{-1}&space;\frac{S_y}{S_x}" title="magnitude = \sqrt {{S_x}^2 + {S_y}^2} \\ direction = \theta = \tan^{-1} \frac{S_y}{S_x}" /></a>  

  - Apply "Non-maximum Suppression"
  - Apply "Hysteresis Threshold"
  ### Dervative of a gaussian filter w.r.t x and y
  ```python
    def plot_gaussian_derivative(der):
        x = np.linspace(-10, 10, num=100)
        y = np.linspace(-10, 10, num=100)
    
        x, y = np.meshgrid(x, y)
    
        z = np.exp(-0.1 * x ** 2 - 0.1 * y ** 2)
        dzdx = -0.2*x*np.exp(-0.1 * x ** 2 - 0.1 * y ** 2)
        dzdy = -0.2 * y * np.exp(-0.1 * x ** 2 - 0.1 * y ** 2)
        z = dzdx if der == 'dx' else dzdy
    
        print(x[:-1].shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap=cm.jet)
        plt.savefig('../data/gauss_2d_der.png')
        plt.show()
```

  Derivative w.r.t x              |  Derivative w.r.t y
  :-------------------------:|:-------------------------:
  ![](/data/gauss_2d_dx.png)  |  ![](/data/gauss_2d_dy.png)
  
  