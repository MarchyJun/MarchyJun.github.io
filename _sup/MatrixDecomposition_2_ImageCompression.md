

```python
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



![png](MatrixDecomposition_2_ImageCompression_files/MatrixDecomposition_2_ImageCompression_8_1.png)


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



![png](MatrixDecomposition_2_ImageCompression_files/MatrixDecomposition_2_ImageCompression_13_1.png)


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


![png](MatrixDecomposition_2_ImageCompression_files/MatrixDecomposition_2_ImageCompression_16_0.png)


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


<div>
        
        
            <div id="bb65d201-f5d7-4bac-9a0e-736f68cadf95" class="plotly-graph-div" style="height:300px; width:400px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("bb65d201-f5d7-4bac-9a0e-736f68cadf95")) {
                    Plotly.newPlot(
                        'bb65d201-f5d7-4bac-9a0e-736f68cadf95',
                        [{"line": {"color": "black"}, "showlegend": false, "type": "scatter", "x": [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100], "y": [38.72800434801821, 48.64225864555799, 59.86507076414238, 65.9142594877179, 70.24829011827819, 73.86228661041059, 76.5730559271184, 78.87432822046352, 81.07745353442554, 82.84671195182341, 84.42715822713176, 85.97137170130974, 87.23500738882416, 88.38765645868054, 89.43637484532967, 90.48527848822206, 91.35661959730272, 92.15945484952188, 92.9663975746454, 93.63468066702882, 94.24782418417439, 94.86669694823121, 95.37771710335609, 95.84539887509116, 96.31856783420366, 96.70976243437545, 97.06828104893181, 97.42609304034336, 97.71825394469296, 97.98436960847295, 98.2478438884394, 98.46144075188357, 98.65279868231532, 98.83959645256591, 98.99103033563821, 99.12611921711172, 99.25776043136445, 99.363478665097, 99.45748344217878, 99.54124566122586, 99.62225732453979, 99.68683825069333, 99.74364933070439, 99.7982300092063, 99.84136685950253, 99.8786791020472, 99.91399323125349, 99.94116050134448, 99.96423570013775, 99.98530177221129, 100.0]}],
                        {"height": 300, "margin": {"b": 20, "l": 20, "r": 20, "t": 20}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "width": 400, "xaxis": {"tickmode": "array", "ticktext": ["1%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], "tickvals": [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "title": {"text": "Percentage of used eigen value"}}, "yaxis": {"tickmode": "array", "ticktext": ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], "tickvals": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "title": {"text": "Restored information"}}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('bb65d201-f5d7-4bac-9a0e-736f68cadf95');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


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


<div>
        
        
            <div id="c499e76c-eb67-4407-b559-133888e61bcf" class="plotly-graph-div" style="height:300px; width:400px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("c499e76c-eb67-4407-b559-133888e61bcf")) {
                    Plotly.newPlot(
                        'c499e76c-eb67-4407-b559-133888e61bcf',
                        [{"line": {"color": "black"}, "showlegend": false, "type": "scatter", "x": [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100], "y": [58.32666105971405, 29.163330529857024, 13.887300252312867, 9.407525977373233, 7.113007446306591, 5.608332794203274, 4.703762988686616, 4.050462573591253, 3.513654280705665, 3.1358419924577445, 2.8313913135783517, 2.558186888583949, 2.351881494343308, 2.1763679499893303, 2.0252312867956266, 1.8815051954746467, 1.7674745775670924, 1.666476030277544, 1.5679209962288723, 1.4879250270335216, 1.4156956567891759, 1.3439322824818905, 1.2847282171743182, 1.2305202755213933, 1.175940747171654, 1.1303616484440706, 1.0881839749946651, 1.0452806641525814, 1.0091117830400353, 0.9753622250788302, 0.9407525977373233, 0.911354079058032, 0.8837372887835462, 0.8552296343066575, 0.8308641176597442, 0.8078484911317735, 0.7839604981144361, 0.7634379719857859, 0.7439625135167608, 0.7254559833297767, 0.7061339111345526, 0.6894404380580856, 0.6735180260936957, 0.6568317686904734, 0.6423641085871591, 0.6285200545227807, 0.6139648532601478, 0.6013057841207634, 0.5891581925223641, 0.5763504057283997, 0.5651808242220353]}],
                        {"height": 300, "margin": {"b": 20, "l": 20, "r": 20, "t": 20}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "width": 400, "xaxis": {"tickmode": "array", "ticktext": ["1%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], "tickvals": [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "title": {"text": "Percentage of used eigen value"}}, "yaxis": {"title": {"text": "Compression ratio"}}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('c499e76c-eb67-4407-b559-133888e61bcf');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


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


<div>
        
        
            <div id="1c62184d-9957-43d1-974a-fce85aaed84d" class="plotly-graph-div" style="height:300px; width:400px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("1c62184d-9957-43d1-974a-fce85aaed84d")) {
                    Plotly.newPlot(
                        '1c62184d-9957-43d1-974a-fce85aaed84d',
                        [{"line": {"color": "red"}, "showlegend": false, "type": "scatter", "x": [5, 10, 21, 31, 41, 52, 62, 72, 83, 93, 103, 114, 124, 134, 144, 155, 165, 175, 186, 196, 206, 217, 227, 237, 248, 258, 268, 279, 289, 299, 310, 320, 330, 341, 351, 361, 372, 382, 392, 402, 413, 423, 433, 444, 454, 464, 475, 485, 495, 506, 516], "y": [91.0756980815575, 64.36237705074441, 42.015682598951486, 33.53904987567485, 28.21281850116283, 24.328301171815898, 21.53858948245428, 19.361198309217396, 17.35020210932844, 15.76737062695549, 14.35882506374491, 12.991586190247569, 11.888301952997374, 10.890472141073639, 9.981314450518692, 9.07095739789643, 8.31077462874294, 7.611012272909489, 6.901305967768841, 6.313941440428554, 5.773397956992962, 5.221444180392682, 4.756954017415829, 4.328697665941026, 3.887123103359944, 3.519597953924112, 3.179198099775906, 2.8350582310539747, 2.5491016637779977, 2.284339359472959, 2.0207564203035213, 1.8028743927721087, 1.6050644870025006, 1.412518691996304, 1.2538263166169858, 1.109329991327992, 0.9679109124621993, 0.8520617103032391, 0.7471600854688385, 0.6520354570269011, 0.5586294912891346, 0.48323412459716375, 0.41505667631948595, 0.3483564759786768, 0.2921327834135385, 0.24240672430148724, 0.19078909070832656, 0.14933336601563424, 0.10965119063655694, 0.06467025812351468, 1.417808959276739e-12]}, {"line": {"color": "green"}, "showlegend": false, "type": "scatter", "x": [5, 10, 21, 31, 41, 52, 62, 72, 83, 93, 103, 114, 124, 134, 144, 155, 165, 175, 186, 196, 206, 217, 227, 237, 248, 258, 268, 279, 289, 299, 310, 320, 330, 341, 351, 361, 372, 382, 392, 402, 413, 423, 433, 444, 454, 464, 475, 485, 495, 506, 516], "y": [91.51577956685868, 66.42590014426213, 45.85213917556248, 36.52931542388457, 30.707728668085803, 26.335809231164724, 23.31694276998832, 20.874935768385622, 18.611849106287742, 16.815614134579562, 15.24843757897346, 13.761010440874562, 12.556786524874084, 11.461797050535113, 10.469500923700743, 9.493713248427008, 8.67832816115821, 7.912564785490552, 7.143617407279913, 6.511628348661045, 5.9278244788516385, 5.328896355458006, 4.839575277457121, 4.388109944422412, 3.9252905494648687, 3.540251316812599, 3.1793934119173017, 2.8176381665505397, 2.520794839612197, 2.247016776062828, 1.97274759235335, 1.7510294147070746, 1.5509772401298434, 1.3554565155201925, 1.1951905314343272, 1.0506211041857534, 0.9085797831356371, 0.7957356553046274, 0.6942279831314845, 0.602616083375575, 0.5124857209467139, 0.4389126201201415, 0.37358476640977484, 0.30959001708480366, 0.2580871904354702, 0.21220846778423172, 0.16656248335394666, 0.12918966332561913, 0.09390867507090586, 0.05495199837950762, 5.680560246433807e-13]}, {"line": {"color": "blue"}, "showlegend": false, "type": "scatter", "x": [5, 10, 21, 31, 41, 52, 62, 72, 83, 93, 103, 114, 124, 134, 144, 155, 165, 175, 186, 196, 206, 217, 227, 237, 248, 258, 268, 279, 289, 299, 310, 320, 330, 341, 351, 361, 372, 382, 392, 402, 413, 423, 433, 444, 454, 464, 475, 485, 495, 506, 516], "y": [82.20939316641928, 60.629660324306904, 42.223704416065246, 33.88087585194144, 28.730520834711943, 24.922434740838863, 22.205355552351897, 19.943644578850446, 17.818023941649475, 16.152072314995394, 14.691504682760709, 13.29922884598856, 12.166580429025021, 11.128824424030585, 10.19783965184176, 9.259361050733524, 8.482234149563048, 7.764838909127532, 7.039942752835406, 6.431122067809249, 5.869673138809814, 5.29943597135703, 4.828892805033575, 4.396990203653346, 3.9500729076051493, 3.57414491150747, 3.22545324363158, 2.869373301241483, 2.5802581203362998, 2.3117779763417343, 2.03972414612439, 1.8183949567886513, 1.6201591097719732, 1.423690123497401, 1.2633662155587317, 1.1212083809101152, 0.980443128858946, 0.8652543992290211, 0.762025062405037, 0.6677814222294853, 0.5748288873786689, 0.4990017289672144, 0.43047859722216736, 0.36102104633234333, 0.3045957902863413, 0.25288713334681007, 0.2021758117772421, 0.1584727473120032, 0.11719217182004706, 0.07024388404567697, 2.466160955989795e-12]}],
                        {"height": 300, "margin": {"b": 20, "l": 20, "r": 20, "t": 20}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "width": 400, "xaxis": {"tickmode": "array", "tickvals": [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 516], "title": {"text": "The number of used eigen value"}}, "yaxis": {"title": {"text": "Frobenius norm"}}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('1c62184d-9957-43d1-974a-fce85aaed84d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


# 3. Optimal Number of Eigen Value

By using above information, I want to find optimal number of eigen value for compression. I set up 2 criteria.        
- 1. I think restored information is most important. I will choose the number of eigenvalue that have 80 ~ 90% restored information.
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



57 ~ 103 eigenvalues have 80 ~ 90% restored information. This range of eigen value is a set of candidates of optimal number of eigenvalue.

## 3.2. Compression Ratio and Frobenius Norm


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


<div>
        
        
            <div id="8649f349-5fc1-456e-94b9-d25b70db9bc4" class="plotly-graph-div" style="height:300px; width:400px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("8649f349-5fc1-456e-94b9-d25b70db9bc4")) {
                    Plotly.newPlot(
                        '8649f349-5fc1-456e-94b9-d25b70db9bc4',
                        [{"line": {"color": "blue"}, "name": "Compression ratio", "showlegend": true, "type": "scatter", "x": [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103], "y": [2.063105914037101, 1.9334843116337102, 1.808126577730431, 1.6630351102320555, 1.546347792455584, 1.4333069533596272, 1.3237442939281614, 1.2175017150855278, 1.1144305565068533, 0.9947350175122635, 0.8981598596047796, 0.8043361306975079, 0.7131478997932503, 0.6244856586647335, 0.5382458831408208, 0.437818273579104, 0.35657047274004844, 0.2774495776821202, 0.2003731501967492, 0.1252629581064121, 0.052044710401222355, -0.019352187858564785, -0.10271856612072773, -0.17034020466267616, -0.23634410065577424, -0.3007876208570728, -0.36372545133404177, -0.42520975223879487, -0.49714218263670185, -0.5556007831665329, -0.6127574104106206, -0.6686550811074824, -0.7233349376279421, -0.7768363489647703, -0.8395354497399307, -0.8905747177393596, -0.940550667655467, -0.9894961856145403, -1.0374428154520015, -1.084420826504866, -1.139556695764517, -1.1845044152696658, -1.228572531653384, -1.271786614773852, -1.3141712529207983, -1.355750099467992, -1.3965459168900043]}, {"line": {"color": "green"}, "name": "- Frobenius norm", "showlegend": true, "type": "scatter", "x": [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103], "xaxis": "x", "y": [-1.9062411048280852, -1.798995866493739, -1.6934528270466869, -1.5903895277143762, -1.4909214085249918, -1.3932406801790098, -1.2975519551132517, -1.2027250166846446, -1.1112772737400995, -1.0206997181220103, -0.9326247821905177, -0.8460400363589751, -0.7604393212536064, -0.6763087202158319, -0.593728613303243, -0.5119389279738523, -0.4317993129371953, -0.35357006656950873, -0.2754097345765655, -0.19916328730858937, -0.1242046583481458, -0.04961949506286035, 0.023817174689224167, 0.09626390882751203, 0.1677756940111993, 0.23804983758794254, 0.30770653699998096, 0.375922100877822, 0.4436614711964659, 0.5106048101997298, 0.5761934840002995, 0.6411536002292249, 0.7049948681871364, 0.7680125564193143, 0.8304443486695515, 0.8922858735360275, 0.9538499389732752, 1.0142475146546188, 1.0735896238037068, 1.1322866901549677, 1.1906204185769478, 1.2480833275987127, 1.3048691981860445, 1.360099976134989, 1.4148765680163116, 1.4689026800339435, 1.5220301329807984], "yaxis": "y2"}, {"line": {"color": "red"}, "name": "Compression ratio - Frobenius norm", "showlegend": true, "type": "scatter", "x": [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103], "y": [0.1568648092090157, 0.13448844513997105, 0.11467375068374408, 0.07264558251767927, 0.05542638393059218, 0.040066273180617396, 0.026192338814909677, 0.014776698400883248, 0.003153282766753751, -0.02596470060974687, -0.03446492258573808, -0.0417039056614672, -0.04729142146035603, -0.05182306155109839, -0.05548273016242222, -0.0741206543947483, -0.07522884019714687, -0.07612048888738854, -0.07503658437981628, -0.07390032920217726, -0.07215994794692344, -0.06897168292142514, -0.07890139143150357, -0.07407629583516413, -0.06856840664457495, -0.06273778326913024, -0.05601891433406081, -0.04928765136097285, -0.05348071144023597, -0.04499597296680313, -0.036563926410321135, -0.027501480878257545, -0.018340069440805684, -0.008823792545455955, -0.009091101070379137, 0.001711155796667918, 0.01329927131780817, 0.024751329040078485, 0.03614680835170536, 0.0478658636501017, 0.05106372281243088, 0.06357891232904689, 0.07629666653266054, 0.0883133613611371, 0.10070531509551328, 0.11315258056595145, 0.12548421609079408]}],
                        {"height": 300, "legend": {"x": 0, "y": 1.38}, "margin": {"b": 20, "l": 20, "r": 20, "t": 20}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "width": 400, "xaxis": {"anchor": "y", "domain": [0.0, 0.94], "title": {"text": "The number of used eigen values"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"font": {"color": "blue"}, "text": "Compression ratio (scaled)"}}, "yaxis2": {"anchor": "x", "overlaying": "y", "side": "right", "title": {"font": {"color": "green"}, "text": " - Frobenius norm (scaled)"}}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('8649f349-5fc1-456e-94b9-d25b70db9bc4');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


    The number of optimal eigen value : 57


## 3.3. Merge All Procedures

I merged all procedures : to find optimal number of eigenvalue based on above criteria and to show result of image compression with the optimal number.


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


<div>
        
        
            <div id="a61a1ee3-25fd-4c36-8a42-b170df18ec58" class="plotly-graph-div" style="height:300px; width:1000px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("a61a1ee3-25fd-4c36-8a42-b170df18ec58")) {
                    Plotly.newPlot(
                        'a61a1ee3-25fd-4c36-8a42-b170df18ec58',
                        [{"line": {"color": "gray"}, "opacity": 0.5, "showlegend": false, "type": "scatter", "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100], "xaxis": "x", "y": [38.72800434801821, 48.64225864555799, 54.70513423764327, 59.86507076414238, 63.182335118065325, 65.9142594877179, 68.24317889081962, 70.24829011827819, 72.00011210019395, 73.86228661041059, 75.27882126008282, 76.5730559271184, 77.76528230705676, 78.87432822046352, 79.91152052949778, 81.07745353442554, 81.98715275632878, 82.84671195182341, 83.66049373854267, 84.42715822713176, 85.15323595828984, 85.97137170130974, 86.61895698509888, 87.23500738882416, 87.82469370375968, 88.38765645868054, 88.92440801427472, 89.43637484532967, 90.01979906912787, 90.48527848822206, 90.93087327523665, 91.35661959730272, 91.76611686535155, 92.15945484952188, 92.6093809027278, 92.9663975746454, 93.30784320146338, 93.63468066702882, 93.94776061154953, 94.24782418417439, 94.59357711926688, 94.86669694823121, 95.1273187230332, 95.37771710335609, 95.61657395062565, 95.84539887509116, 96.10889778742371, 96.31856783420366, 96.5186615921876, 96.70976243437545, 96.89289664956034, 97.06828104893181, 97.23564840268418, 97.42609304034336, 97.57604986715783, 97.71825394469296, 97.85428787135328, 97.98436960847295, 98.1080150297077, 98.2478438884394, 98.35762875846406, 98.46144075188357, 98.55972936466213, 98.65279868231532, 98.74049810500908, 98.83959645256591, 98.91716333049855, 98.99103033563821, 99.06049958207144, 99.12611921711172, 99.187993033263, 99.25776043136445, 99.31206053362227, 99.363478665097, 99.41187369142229, 99.45748344217878, 99.50052974619639, 99.54124566122586, 99.58693435995981, 99.62225732453979, 99.65556752076914, 99.68683825069333, 99.71609444253315, 99.74364933070439, 99.77446638083387, 99.7982300092063, 99.82052133888956, 99.84136685950253, 99.8606445198475, 99.8786791020472, 99.89855734268598, 99.91399323125349, 99.92814804511502, 99.94116050134448, 99.95317486356227, 99.96423570013775, 99.9762682311871, 99.98530177221129, 99.9933250187985, 100.0], "yaxis": "y"}, {"line": {"color": "black"}, "showlegend": false, "type": "scatter", "x": [11, 11], "y": [75.27882126008282, 84.42715822713176]}, {"line": {"color": "black"}, "showlegend": false, "type": "scatter", "x": [20, 20], "y": [75.27882126008282, 84.42715822713176]}, {"line": {"color": "black"}, "showlegend": false, "type": "scatter", "x": [11, 20], "y": [75.27882126008282, 75.27882126008282]}, {"line": {"color": "black"}, "showlegend": false, "type": "scatter", "x": [11, 20], "y": [84.42715822713176, 84.42715822713176]}, {"line": {"color": "blue"}, "name": "Compression ratio", "showlegend": true, "type": "scatter", "x": [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103], "xaxis": "x2", "y": [2.063105914037101, 1.9334843116337102, 1.808126577730431, 1.6630351102320555, 1.546347792455584, 1.4333069533596272, 1.3237442939281614, 1.2175017150855278, 1.1144305565068533, 0.9947350175122635, 0.8981598596047796, 0.8043361306975079, 0.7131478997932503, 0.6244856586647335, 0.5382458831408208, 0.437818273579104, 0.35657047274004844, 0.2774495776821202, 0.2003731501967492, 0.1252629581064121, 0.052044710401222355, -0.019352187858564785, -0.10271856612072773, -0.17034020466267616, -0.23634410065577424, -0.3007876208570728, -0.36372545133404177, -0.42520975223879487, -0.49714218263670185, -0.5556007831665329, -0.6127574104106206, -0.6686550811074824, -0.7233349376279421, -0.7768363489647703, -0.8395354497399307, -0.8905747177393596, -0.940550667655467, -0.9894961856145403, -1.0374428154520015, -1.084420826504866, -1.139556695764517, -1.1845044152696658, -1.228572531653384, -1.271786614773852, -1.3141712529207983, -1.355750099467992, -1.3965459168900043], "yaxis": "y2"}, {"line": {"color": "green"}, "name": "- Frobenius norm", "showlegend": true, "type": "scatter", "x": [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103], "xaxis": "x2", "y": [-1.9062411048280852, -1.798995866493739, -1.6934528270466869, -1.5903895277143762, -1.4909214085249918, -1.3932406801790098, -1.2975519551132517, -1.2027250166846446, -1.1112772737400995, -1.0206997181220103, -0.9326247821905177, -0.8460400363589751, -0.7604393212536064, -0.6763087202158319, -0.593728613303243, -0.5119389279738523, -0.4317993129371953, -0.35357006656950873, -0.2754097345765655, -0.19916328730858937, -0.1242046583481458, -0.04961949506286035, 0.023817174689224167, 0.09626390882751203, 0.1677756940111993, 0.23804983758794254, 0.30770653699998096, 0.375922100877822, 0.4436614711964659, 0.5106048101997298, 0.5761934840002995, 0.6411536002292249, 0.7049948681871364, 0.7680125564193143, 0.8304443486695515, 0.8922858735360275, 0.9538499389732752, 1.0142475146546188, 1.0735896238037068, 1.1322866901549677, 1.1906204185769478, 1.2480833275987127, 1.3048691981860445, 1.360099976134989, 1.4148765680163116, 1.4689026800339435, 1.5220301329807984], "yaxis": "y3"}, {"line": {"color": "red"}, "name": "Compression ratio - Frobenius norm", "showlegend": true, "type": "scatter", "x": [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103], "xaxis": "x2", "y": [0.1568648092090157, 0.13448844513997105, 0.11467375068374408, 0.07264558251767927, 0.05542638393059218, 0.040066273180617396, 0.026192338814909677, 0.014776698400883248, 0.003153282766753751, -0.02596470060974687, -0.03446492258573808, -0.0417039056614672, -0.04729142146035603, -0.05182306155109839, -0.05548273016242222, -0.0741206543947483, -0.07522884019714687, -0.07612048888738854, -0.07503658437981628, -0.07390032920217726, -0.07215994794692344, -0.06897168292142514, -0.07890139143150357, -0.07407629583516413, -0.06856840664457495, -0.06273778326913024, -0.05601891433406081, -0.04928765136097285, -0.05348071144023597, -0.04499597296680313, -0.036563926410321135, -0.027501480878257545, -0.018340069440805684, -0.008823792545455955, -0.009091101070379137, 0.001711155796667918, 0.01329927131780817, 0.024751329040078485, 0.03614680835170536, 0.0478658636501017, 0.05106372281243088, 0.06357891232904689, 0.07629666653266054, 0.0883133613611371, 0.10070531509551328, 0.11315258056595145, 0.12548421609079408], "yaxis": "y2"}],
                        {"height": 300, "legend": {"x": 0.5, "y": 1.38}, "margin": {"l": 20, "t": 20}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "width": 1000, "xaxis": {"anchor": "y", "domain": [0.0, 0.296], "tickmode": "array", "ticktext": ["1%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], "tickvals": [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "title": {"text": "Percentage of used eigen values"}}, "xaxis2": {"anchor": "y2", "domain": [0.496, 0.94], "tickmode": "array", "tickvals": [57, 77, 97], "title": {"text": "The number of used eigen values"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "tickmode": "array", "ticktext": ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], "tickvals": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "title": {"text": "Restored information"}}, "yaxis2": {"anchor": "x2", "domain": [0.0, 1.0], "title": {"font": {"color": "blue"}, "text": "Compression ratio (scaled)"}}, "yaxis3": {"anchor": "x2", "overlaying": "y2", "side": "right", "title": {"font": {"color": "green"}, "text": " - Frobenius norm (scaled)"}}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('a61a1ee3-25fd-4c36-8a42-b170df18ec58');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


    Total number of eigen value : 516
    The number of optimal eigen value : 57
    
    - Original Image : 



![png](MatrixDecomposition_2_ImageCompression_files/MatrixDecomposition_2_ImageCompression_43_2.png)


    - Compressed Image with 57 eigen values



![png](MatrixDecomposition_2_ImageCompression_files/MatrixDecomposition_2_ImageCompression_43_4.png)


    Restored information : 75.28%
    Compression ratio : 5.12
    Frobenius norm : 23.69


# 4. Make Web Application with Dash

I made a simple web application of image compression with Dash and Heroku.

![title](dash.png)

http://my-image-compression-app.herokuapp.com/
