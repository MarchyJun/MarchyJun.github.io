

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from io import BytesIO
import base64
from PIL import Image
#import cv2
import plotly.graph_objects as go
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import preprocessing
import plotly.offline as pyo

%matplotlib inline
```

# 1. Image Compression with SVD


```python
org_img_uint = io.imread('marchisio.jpg')
org_img_uint.dtype, org_img_uint.shape
```




    (dtype('uint8'), (672, 516, 3))



Original image is .jpg file with (672, 516) size matrix for each R,G,B. To apply svd, we have to change a type of the image from uint to float.


```python
def uint2float(uint_array):
    """
    uint array -> float array
    """
    info = np.iinfo(uint_array.dtype)
    float_array = uint_array.astype(np.float64)/info.max
    return float_array
```


```python
def get_RGB(img):
    """
    3 dimension array of img -> 2 dimension R, G, B array
    """
    [R, G, B] = [img[:,:,i] for i in [0,1,2]]
    return [R, G, B]
```


```python
org_img = uint2float(org_img_uint)
[org_r, org_g, org_b] = get_RGB(org_img)
```


```python
def display_img(img, r, g, b):
    """
    show plots of img, r, g, b
    """
    plt.figure(figsize = (13,10))
    ax1 = plt.subplot(1,4,1)
    ax1.imshow(img)
    ax1.set_yticks([])
    ax1.set_xticks([])
    

    cmaps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues]
    for i,rgb in enumerate([r, g, b]):
        ax = plt.subplot(1,4,i+2)
        ax.imshow(rgb, cmap = cmaps[i])
        ax.set_yticks([])
        ax.set_xticks([])

    plt.show()
```


```python
print('Original Image : ')
display_img(org_img, org_r, org_g, org_b)
```

    Original Image : 


![Image](/assets/images/MatrixDecomposition_2_ImageCompression_files/MatrixDecomposition_2_ImageCompression_8_1.png)


These are plots of an original image and R, G, B.  

Next, we have to apply svd to each R, G, B array, and reconstruct R, G, B only with p% eigen values. 


```python
def float2uint(float_array):
    """
    float array -> uint array
    """
    uint_array = float_array*255
    uint_array = uint_array.astype('uint8')
    return uint_array
```


```python
def svd4rgb(rgb_array):
    """
    return compressed R, G, B image with p_eigen% eigen values of input R, G, B image
    """
    [u, s, v] = np.linalg.svd(rgb_array)
    return [u, s, v]
```


```python
def svd4img(img, p_eigen):
    """
    return compressed image with p_eigen% eigen values of input image
    """
    original_shape = img.shape
    R, G, B = get_RGB(img)
    n_total = min(original_shape[:2])
    k = int(round(n_total * p_eigen / 100))
    
    compressed_rgb_list = []
    for rgb in [R, G, B]:
        [u,s,v] = svd4rgb(rgb)
        u_k = u[:,:k]
        s_k = np.diag(s[:k])
        v_k = v[:k,:]
        compressed_rgb = np.dot(np.dot(u_k, s_k), v_k)
        compressed_rgb_list.append(compressed_rgb)
   
    original_shape = img.shape
    compressed_img = np.zeros(original_shape)
    
    for i in range(3):
        compressed_img[:,:,i] = compressed_rgb_list[i]
        
    return compressed_img
```


```python
compressed_img = svd4img(org_img, 10)
compressed_img_uint = float2uint(compressed_img)
comp_r, comp_g, comp_b = get_RGB(compressed_img_uint)

print('Compressed Image with 10% eigen values')
display_img(compressed_img_uint, comp_r, comp_g, comp_b)
```

    Compressed Image with 10% eigen values



![Image](/assets/images/MatrixDecomposition_2_ImageCompression_files/MatrixDecomposition_2_ImageCompression_14_1.png)


These are plots of images only with 10% eigenvalues.

Following plots are results with 1%, 2%, 5%, 10%, 20%, 40%, 60%, 80% eigen values.


```python
plt.figure(figsize = (13,10))
for i,p in enumerate([1, 2, 5, 10, 20, 40, 60, 80]):
    compressed_img = svd4img(org_img, p)
    compressed_img_uint = float2uint(compressed_img)
    ax = plt.subplot(2,4,i+1)
    ax.imshow(compressed_img_uint)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.title('{}% are used'.format(p))
```


![Image](/assets/images/MatrixDecomposition_2_ImageCompression_files/MatrixDecomposition_2_ImageCompression_17_0.png)


# 2. Some Information after SVD 

## 2.1. Restored Information

We can get restored information by comparing the sum of eigenvalues of a compressed image and an original image. 


```python
def calculate_restored_info(img, p_eigen):
    """
    return restored information of compressed img when use p_eigen% eigen values 
    """    
    (m, n, d) = img.shape
    n_eigen = min(m, n)
    k_eigen = int(round(n_eigen * p_eigen / 100))

    [r, g, b] = get_RGB(img)

    [u_red, s_red, v_red] = svd4rgb(r)
    [u_green, s_green, v_green] = svd4rgb(g)
    [u_blue, s_blue, v_blue] = svd4rgb(b)

    sum_sigma = 0
    sum_k_sigma = 0

    for sigma in [s_red, s_green, s_blue]:
        sum_sigma += np.sum(sigma)
        sum_k_sigma += np.sum(sigma[: k_eigen])

    return (sum_k_sigma / sum_sigma) * 100

def plot_restored_info(img):
    """
    plot restored information of img
    """   
    n_eigen = min(img.shape[:2])
    p_eigen = np.arange(0, 101, 2)
    p_eigen[0] = 1
    
    fig = go.Figure()
    restored_info_array = np.zeros(len(p_eigen))

    for i,p in enumerate(p_eigen):
        restored_info = calculate_restored_info(img, p)
        restored_info_array[i] = restored_info
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x = p_eigen,
            y = restored_info_array,
            line = dict(color = 'black'),
            showlegend = False
        ))

    
    fig.update_layout(
                      xaxis_title = 'Percentage of used eigen value',
                      yaxis_title = 'Restored information',
                      width=400, height=300,
                      margin=dict(l=20, r=20, t=20, b=20)
                      )
    
    x_tickvals = np.arange(0, 101, 10)
    x_tickvals[0] = 1
    fig.update_xaxes(tickmode = 'array',
                     tickvals = x_tickvals,
                     ticktext=['{}%'.format(i) for i in x_tickvals]
                     )
    
    y_tickvals = np.arange(10,101,10)
    fig.update_yaxes(tickmode = 'array',
                     tickvals = y_tickvals,
                     ticktext = ['{}%'.format(i) for i in y_tickvals])
    fig.show()

```


```python
plot_restored_info(org_img)
```
![Image](/assets/images/MatrixDecomposition_2_ImageCompression_files/restored_information.png)


   

## 2.2. Compression Ratio

Compression ratio shows how the original image is compressed. The higher the value, the more compressed the image is.


```python
def calculate_compression_ratio(img, p_eigen):
    """
    return compression ratio between compressed img with p_eigen% eigen values and img 
    """  
    (m, n, d) = img.shape
    n_eigen = min(m, n)
    k_eigen = int(round(n_eigen * p_eigen / 100))
    compression_ratio = (m*n)/(k_eigen*(m + n)+k_eigen)

    return compression_ratio

def plot_compression_ratio(img):
    """
    plot compression ratio of img
    """
    n_eigen = min(img.shape[:2])
    p_eigen = np.arange(0, 101, 2)
    p_eigen[0] = 1

    compression_ratio_array = np.zeros(len(p_eigen))
    
    for i,p in enumerate(p_eigen):
        compression_ratio = calculate_compression_ratio(img, p)
        compression_ratio_array[i] = compression_ratio

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = p_eigen,
        y = compression_ratio_array,
        line = dict(color = 'black'),
        showlegend = False
        ))
    
    fig.update_layout(
                      xaxis_title = 'Percentage of used eigen value',
                      yaxis_title = 'Compression ratio',
                      width=400, height=300,
                      margin=dict(l=20, r=20, t=20, b=20)
                      )
    x_tickvals = np.arange(0, 101, 10)
    x_tickvals[0] = 1
    fig.update_xaxes(tickmode = 'array',
                     tickvals = x_tickvals,
                     ticktext=['{}%'.format(i) for i in x_tickvals]
                     )
    
    fig.show()
```


```python
plot_compression_ratio(org_img)
```
![Image](/assets/images/MatrixDecomposition_2_ImageCompression_files/compression_ratio.png)



## 2.3 Frobenius Norm

Frobenius norm is a norm between matrix of compressed image and original image.


```python
def calculate_frobenius_norm(rgb_array):
    """
    return frobenius norm between matrix of compressed img and original img
    """
    [u,s,v] = svd4rgb(rgb_array)
    
    n_eigen = min(rgb_array.shape)
    p_eigen = np.arange(0, 101, 2)
    p_eigen[0] = 1
    k_eigen = np.round(n_eigen * p_eigen / 100).astype('int')
    
    frobenius_norm_array = np.zeros(len(k_eigen))
    
    for i,k in enumerate(k_eigen):
        u_k = u[:,:k]
        s_k = np.diag(s[:k])
        v_k = v[:k,:]
    
        compressed_rgb = np.dot(np.dot(u_k,s_k), v_k)
        error_matrix = np.subtract(rgb_array, compressed_rgb)
        frobenius_norm = np.linalg.norm(error_matrix,'fro')
        frobenius_norm_array[i] = frobenius_norm
        
    return frobenius_norm_array

def plot_frobenius_norm(r_frobenius_norm, g_frobenius_norm, b_frobenius_norm, rank):
    """
    plot frobenius norm of img
    """
    n_eigen = rank
    p_eigen = np.arange(0, 101, 2)
    p_eigen[0] = 1
    k_eigen = np.round(n_eigen * p_eigen / 100).astype('int')
    
    fig = go.Figure()
    
    for c,frobenius_norm in zip(['red','green','blue'], [r_frobenius_norm, g_frobenius_norm, b_frobenius_norm]):
        fig.add_trace(go.Scatter(
            x = k_eigen,
            y = frobenius_norm,
            line = dict(color = c),
            showlegend = False
        ))

    fig.update_layout(
                      xaxis_title = 'The number of used eigen value',
                      yaxis_title = 'Frobenius norm',
                      width=400, height=300,
                      margin=dict(l=20, r=20, t=20, b=20)
                     )
    
    x_tickvals = np.arange(0,n_eigen,50)
    x_tickvals[0] = 1
    x_tickvals = np.append(x_tickvals, n_eigen)
    fig.update_xaxes(tickmode = 'array',
                     tickvals = x_tickvals
                     )  
    
    fig.show()
```


```python
r_frobenius_norm = calculate_frobenius_norm(org_r)
g_frobenius_norm = calculate_frobenius_norm(org_g)
b_frobenius_norm = calculate_frobenius_norm(org_b)

plot_frobenius_norm(r_frobenius_norm, g_frobenius_norm, b_frobenius_norm, 516)
```
![Image](/assets/images/MatrixDecomposition_2_ImageCompression_files/frobenius_norm.png)




# 3. Optimal Number of Eigen Value

By using above information, I want to find optimal number of eigen value for compression. I set up 2 criteria.        
- 1. I think restored information is most important. I will choose the number of eigenvalue that have 75 ~ 85% restored information.
- 2. Compare compression ratio and frobenius norm within above range of eigenvalue. Since the bigger compression ratio is better, and the smaller frobenius norm is better, so I will choose the number of eigenvalue that have biggest value of compression ratio - frobenius norm.  

## 3.1. 80~90% Restored Information


```python
p_eigen = np.arange(1,101)
p_eigen

restored_info_array = np.zeros(len(p_eigen))

for i, p in enumerate(p_eigen):
    restored_info = calculate_restored_info(org_img, p)
    restored_info_array[i] = restored_info

boolen = (restored_info_array >=75) & (restored_info_array<=85)

n_eigen = min(org_img.shape[:2])
p_eigen_range = p_eigen[boolen]

k_eigen_range = np.arange(np.round(n_eigen * p_eigen_range / 100)[0], 
                          np.round(n_eigen * p_eigen_range / 100)[-1] + 1).astype('int')
k_eigen_range
```




    array([ 57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
            70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,
            83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
            96,  97,  98,  99, 100, 101, 102, 103])



57 ~ 103 eigenvalues have 75 ~ 85% restored information. This range of eigenvalues is a set of candidates of optimal number of eigenvalue.

## 3.2. Compression Ratio and Frobenius Norm


```python
def calculate_frobenius_norm_k(img, k_eigen):
    (m, n, d) = img.shape
    n_eigen = min(m, n)

    [r, g, b] = get_RGB(img)

    [u_r, s_r, v_r] = svd4rgb(r)
    [u_g, s_g, v_g] = svd4rgb(g)
    [u_b, s_b, v_b] = svd4rgb(b)

    u_r_k = u_r[:, : k_eigen]
    s_r_k = np.diag(s_r[: k_eigen])
    v_r_k = v_r[: k_eigen, :]
    compressed_r = np.dot(np.dot(u_r_k, s_r_k), v_r_k)
    error_matrix_r = np.subtract(r, compressed_r)
    frobenius_norm_r = np.linalg.norm(error_matrix_r, 'fro')

    u_g_k = u_g[:, : k_eigen]
    s_g_k = np.diag(s_g[: k_eigen])
    v_g_k = v_g[: k_eigen, :]
    compressed_g = np.dot(np.dot(u_g_k, s_g_k), v_g_k)
    error_matrix_g = np.subtract(g, compressed_g)
    frobenius_norm_g = np.linalg.norm(error_matrix_g, 'fro')

    u_b_k = u_b[:, : k_eigen]
    s_b_k = np.diag(s_b[: k_eigen])
    v_b_k = v_b[: k_eigen, :]
    compressed_b = np.dot(np.dot(u_b_k, s_b_k), v_b_k)
    error_matrix_b = np.subtract(b, compressed_b)
    frobenius_norm_b = np.linalg.norm(error_matrix_b, 'fro')

    return np.mean([frobenius_norm_r, frobenius_norm_g, frobenius_norm_b])

```


```python
compression_ratio_array = np.zeros(len(k_eigen_range))
frobenius_norm_array = np.zeros(len(k_eigen_range))

for i,k in enumerate(k_eigen_range):
    compression_ratio = calculate_compression_ratio(org_img, k)
    compression_ratio_array[i] = compression_ratio
    
    frobenius_norm = calculate_frobenius_norm_k(org_img, k)
    frobenius_norm_array[i] = frobenius_norm

compression_ratio_array_scaled = preprocessing.scale(compression_ratio_array)
frobenius_norm_array_scaled = preprocessing.scale(-frobenius_norm_array)

print('Compression ratio (scaled)')
print('mean : {}, std: {}'.format(np.mean(compression_ratio_array_scaled),np.std(compression_ratio_array_scaled) ))
print('---------------------------------------')
print('Frobenius norm (scaled)')
print('mean : {}, std: {}'.format(np.mean(frobenius_norm_array_scaled),np.std(frobenius_norm_array_scaled) ))
print('---------------------------------------')
```

    Compression ratio (scaled)
    mean : -3.826726169984582e-16, std: 0.9999999999999998
    ---------------------------------------
    Frobenius norm (scaled)
    mean : -8.078644136634117e-16, std: 0.9999999999999999
    ---------------------------------------


I got compression ratio and frobenius norm for each number of eigenvalues within above range. Since compression ratio and frobenius norm have different scale, I standardized them.


```python
def show_comp_frob_target(k_eigen_range, compression_ratio_array_scaled, frobenius_norm_array_scaled):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
            go.Scatter(
                x=k_eigen_range,
                y= compression_ratio_array_scaled,
                line=dict(color='blue'),
                showlegend=True,
                name='Compression ratio'
            )
        )

    fig.add_trace(
        go.Scatter(
            x=k_eigen_range,
            y= frobenius_norm_array_scaled,
            line=dict(color='green'),
            showlegend=True,
            name='- Frobenius norm',
        ),
        secondary_y=True
    )

    fig.update_layout(
            xaxis=dict(
                title='The number of used eigen values'
            ),

            yaxis1=dict(
                title='Compression ratio (scaled)',
                titlefont=dict(color='blue')
            ),
            yaxis2=dict(
                title=' - Frobenius norm (scaled)',
                titlefont=dict(color='green')
            ),
            width=400, height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(x=0, y=1.38)
        )


    total = compression_ratio_array_scaled + frobenius_norm_array_scaled

    fig.add_trace(
            go.Scatter(
                x=k_eigen_range,
                y= total,
                line=dict(color='red'),
                showlegend=True,
                name='Compression ratio - Frobenius norm'
            )
    )

    fig.show()
    print('The number of optimal eigen value : {}'.format(k_eigen_range[np.argmax(total)]))

```


```python
show_comp_frob_target(k_eigen_range, compression_ratio_array_scaled, frobenius_norm_array_scaled)
```
![Image](/assets/images/MatrixDecomposition_2_ImageCompression_files/comp_frob_target.png)



## 3.3. Merge All Procedures

I merged all procedures : to find optimal number of eigenvalue based on above criteria and to show result of image compression with the optimal number.


```python
def calculate_restored_info_k(img, k_eigen):
    (m, n, d) = img.shape
    n_eigen = min(m, n)

    [r, g, b] = get_RGB(img)

    [u_red, s_red, v_red] = svd4rgb(r)
    [u_green, s_green, v_green] = svd4rgb(g)
    [u_blue, s_blue, v_blue] = svd4rgb(b)

    sum_sigma = 0
    sum_k_sigma = 0

    for sigma in [s_red, s_green, s_blue]:
        sum_sigma += np.sum(sigma)
        sum_k_sigma += np.sum(sigma[: k_eigen])

    return (sum_k_sigma / sum_sigma) * 100
```


```python
def calculate_compression_ratio_k(img, k_eigen):
    (m, n, d) = img.shape
    n_eigen = min(m, n)
    compression_ratio = (m*n)/(k_eigen*(m + n)+k_eigen)

    return compression_ratio
```


```python
def calculate_optimal_k(org_img):
    n_eigen = min(org_img.shape[:2])
    p_eigen = np.arange(1,101)

    restored_info_array = np.zeros(len(p_eigen))

    for i, p in enumerate(p_eigen):
        #compression_ratio = calculate_compression_ratio(org_img, p)
        #compression_ratio_array[i] = compression_ratio

        restored_info = calculate_restored_info(org_img, p)
        restored_info_array[i] = restored_info
        
    boolen = (restored_info_array >=75) & (restored_info_array<=85)
    p_eigen_range = p_eigen[boolen]
    k_eigen_range = np.arange(np.round(n_eigen * p_eigen_range / 100)[0], 
                              np.round(n_eigen * p_eigen_range / 100)[-1] + 1).astype('int')
    
    compression_ratio_array = np.zeros(len(k_eigen_range))
    frobenius_norm_array = np.zeros(len(k_eigen_range))

    for i,k in enumerate(k_eigen_range):
        compression_ratio = calculate_compression_ratio(org_img, k)
        compression_ratio_array[i] = compression_ratio

        frobenius_norm = calculate_frobenius_norm_k(org_img, k)
        frobenius_norm_array[i] = frobenius_norm
        
    compression_ratio_array_scaled = preprocessing.scale(compression_ratio_array)
    frobenius_norm_array_scaled = preprocessing.scale(-frobenius_norm_array)
    total = compression_ratio_array_scaled + frobenius_norm_array_scaled
    optimal_k = k_eigen_range[np.argmax(total)]
    return optimal_k, compression_ratio_array_scaled, frobenius_norm_array_scaled, total, restored_info_array, k_eigen_range
        

```


```python
def compression_from_k(org_img, optimal_k):
    [r,g,b] = get_RGB(org_img)
    
    [u_r, s_r, v_r] = svd4rgb(r)
    [u_g, s_g, v_g] = svd4rgb(g)
    [u_b, s_b, v_b] = svd4rgb(b)
    
    original_shape = org_img.shape
    img_reconst = np.zeros(original_shape)
        
    compressed_rgb_list = []
    for rgb in [r, g, b]:
        [u,s,v] = svd4rgb(rgb)
        u_k = u[:,:optimal_k]
        s_k = np.diag(s[:optimal_k])
        v_k = v[:optimal_k,:]
        compressed_rgb = np.dot(np.dot(u_k, s_k), v_k)
        compressed_rgb_list.append(compressed_rgb)
   
    compressed_img = np.zeros(original_shape)
    
    for i in range(3):
        compressed_img[:,:,i] = compressed_rgb_list[i]
        
    return compressed_img
```


```python
def optimal_compression(org_img, show_graph = True, show_image = True):
    (optimal_k, compression_ratio_array_scaled, 
     frobenius_norm_array_scaled, total, restored_info_array, k_eigen_range) = calculate_optimal_k(org_img)
    
    p_eigen = np.arange(1,101)
    
    # show graph
    if show_graph:
        boolen = (restored_info_array >=75) & (restored_info_array<=85)


        first_y = restored_info_array[boolen][0]
        last_y = restored_info_array[boolen][-1]


        first_where = np.where(restored_info_array == first_y)
        last_where = np.where(restored_info_array == last_y)

        first_x = p_eigen[first_where][0]
        last_x = p_eigen[last_where][0]

        fig = make_subplots(rows=1, cols = 2 , column_widths=[0.4, 0.6],
                            specs=[[{},{"secondary_y": True}]])

        p_eigen = np.arange(1,101)

        # restored information plot
        fig.add_trace(
            go.Scatter(
                    x = p_eigen,
                    y = restored_info_array,
                    showlegend = False,
                    line = dict(color = 'gray'),
                    opacity = 0.5
            ),
            row = 1, col = 1
        )    

        x_tickvals = np.arange(0, 101, 10)
        x_tickvals[0] = 1
        fig.update_xaxes(tickmode='array',
                         tickvals=x_tickvals,
                         ticktext=['{}%'.format(i) for i in x_tickvals],
                         row=1, col=1
                         )

        y_tickvals = np.arange(10, 101, 10)

        fig.update_yaxes(tickmode='array',
                         tickvals=y_tickvals,
                         ticktext=['{}%'.format(i) for i in y_tickvals],
                         row=1, col=1
                         )
        fig.update_layout(height = 400)
        

        # show 80~90% box
        fig.add_trace(
            go.Scatter(
                x = [first_x,first_x],
                y = [first_y, last_y],
                line = dict(color = 'black'),
                showlegend = False
            )
        )

        fig.add_trace(
            go.Scatter(
                x = [last_x,last_x],
                y = [first_y,last_y],
                line = dict(color = 'black'),
                showlegend = False
            )
        )

        fig.add_trace(
            go.Scatter(
                x = [first_x,last_x],
                y = [first_y,first_y],
                line = dict(color = 'black'),
                showlegend = False
            )
        )

        fig.add_trace(
            go.Scatter(
                x = [first_x,last_x],
                y = [last_y,last_y],
                line = dict(color = 'black'),
                showlegend = False
            )
        )




        # compression ratio and frobenius norm plot
        fig.add_trace(
                go.Scatter(
                    x=k_eigen_range,
                    y= compression_ratio_array_scaled,
                    line=dict(color='blue'),
                    showlegend=True,
                    name='Compression ratio'
                ),
                row = 1, col = 2
            )

        fig.add_trace(
            go.Scatter(
                x=k_eigen_range,
                y= frobenius_norm_array_scaled,
                line=dict(color='green'),
                showlegend=True,
                name='- Frobenius norm',
            ),
            secondary_y=True,
            row = 1, col = 2
        )

        total = compression_ratio_array_scaled + frobenius_norm_array_scaled

        fig.add_trace(
                go.Scatter(
                    x=k_eigen_range,
                    y= total,
                    line=dict(color='red'),
                    showlegend=True,
                    name='Compression ratio - Frobenius norm'
                ),
                row = 1, col = 2
        )

        # setting axis
        fig.update_layout(
                xaxis1=dict(
                    title='Percentage of used eigen values'
                ),

                xaxis2=dict(
                    title='The number of used eigen values'
                ),


                yaxis1=dict(
                    title='Restored information'
                ),

                yaxis2=dict(
                    title='Compression ratio (scaled)',
                    titlefont=dict(color='blue')
                ),
                yaxis3=dict(
                    title=' - Frobenius norm (scaled)',
                    titlefont=dict(color='green')
                ),

                legend=dict(x=0.5, y=1.38),
                height=300, width = 1000,
                margin=dict(l=20, t=20)

            )
        
        fig.update_xaxes(tickmode='array',
                         tickvals= np.arange(k_eigen_range[0],k_eigen_range[-1], 20),
                         row=1, col=2
                         )


        fig.show()

    print('Total number of eigen value : {}'.format(min(org_img.shape[:2])))
    print('The number of optimal eigen value : {}'.format(optimal_k))
    print('')
    #show image
    if show_image:
        compressed_img = compression_from_k(org_img, optimal_k)
        compressed_img_uint = float2uint(compressed_img)

        org_r, org_g, org_b = get_RGB(org_img)  
        comp_r, comp_g, comp_b = get_RGB(compressed_img)
        
        print('- Original Image : ')
        plt.figure(figsize = (13,10))
        ax1 = plt.subplot(1,4,1)
        ax1.imshow(org_img)
        ax1.set_yticks([])
        ax1.set_xticks([])

        cmaps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues]
        for i,rgb in enumerate([org_r, org_g, org_b]):
            ax = plt.subplot(1,4,i+2)
            ax.imshow(rgb, cmap = cmaps[i])
            ax.set_yticks([])
            ax.set_xticks([])

        plt.show()
        
        print('- Compressed Image with {} eigen values'.format(optimal_k))
        plt.figure(figsize = (13,10))
        ax = plt.subplot(1,4,1)
        ax.imshow(compressed_img_uint)
        ax.set_yticks([])
        ax.set_xticks([])

        for i,rgb in enumerate([comp_r, comp_g, comp_b]):
            ax = plt.subplot(1,4,i+2)
            ax.imshow(rgb, cmap = cmaps[i])
            ax.set_yticks([])
            ax.set_xticks([])
    
        plt.show()
        
        optimal_restored_info = calculate_restored_info_k(org_img, optimal_k)
        optimal_compression_ratio = calculate_compression_ratio_k(org_img, optimal_k)
        optimal_frobenius_norm = calculate_frobenius_norm_k(org_img, optimal_k)
        
        print('Restored information : {}%'.format(round(optimal_restored_info,2)))
        print('Compression ratio : {}'.format(round(optimal_compression_ratio,2)))
        print('Frobenius norm : {}'.format(round(optimal_frobenius_norm,2)))
    
```


```python
optimal_compression(org_img,show_graph=True,show_image=True)
```

![Image](/assets/images/MatrixDecomposition_2_ImageCompression_files/optimal_compression.png)

# 4. Make Web Application with Dash

I made a simple web application of image compression with Dash and Heroku.

![Image](/assets/images/MatrixDecomposition_2_ImageCompression_files/dash.png)

http://my-image-compression-app.herokuapp.com/
