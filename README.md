# Road Segmentation Project

## System Breakdown
<img src="./images/system breakdown.png" alt="drawing" width="720"/>

## Architecture Diagram
<img src="./images/architecture diagram.png" alt="drawing" width='720'/>

## API Structure

### get_lbp_image

* Input: U8 Array sobel_array[L,W]
* Output: U8 Array lbp_array[L,W]
* Method: create a L x W sized array and fill with the LBP of respective pixel

### get_bfs_kernel

* Input: U8 Array lbp_array[L,W]
* Output: U8 Array kernel_queue[2]
* Parameters: 
+ U8 kernel_size
+ U8 Array kernel_center[2]
+ U8 bfs_stride
* Method: create a double ended queue where running the BFS algorithm clockwise

### compute_lbp_histogram

* Input: U8 Array lbp_array[L,W]
* Output: U8 Array lbp_hist[256]
* Parameters: 
+ U8 Array lbp_kernel_center[2]
+ U8 lbp_kernel_size
* Method: create a double ended queue where running the BFS algorithm
    clockwise

### get_top_n_modes

* Input: U8 Array lbp_hist[256]
* Output: U8 Array top_n_indices[N,2]
* Parameters: 
+ U8 N
* Method: get the top N mode of the histogram and it's value

### euclidean_distance_top_modes_weighted

* Input: U8 Array top_n_indices1[N,2], U8 Array top_n_indices2[N,2]
* Output: float distances
* Parameters: 
* Method: calculate the weighted Euclidean distance based on the top N modes of two histograms.

### Self made LBP code snippet

```
    neighboring_lbp_k_dist = ((-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0))
    center = sobel_array[kernel_center]
    lbp_val = 0
    for idx, (dy, dx) in enumerate(neighboring_lbp_k_dist):
        to_bitshift = 1 if center < sobel_array[kernel_center[0] - dy, kernel_center[1] - dx] else 0
        lbp_val |= (to_bitshift << idx)
    return lbp_val
```
## Comparison of scikit's LBP vs self made LBP

<img src="./images/lbp comparison scikit vs self made.png" alt="drawing" width='720'/>

### Result using Chi-Square vs Euclidean

##### Using Chi-Square with 79 Histogram Kernel Size and 5 BFS Stride (Image Diluted to fill Empty values)

<img src="./images/chi square kernel 79 strd 5 dilate 3.png" alt="plot" width='720'/>

##### Using Chi-Square with 39 Histogram Kernel Size and 5 BFS Stride (Image Diluted to fill Empty values)

<img src="./images/chi square kernel 39 strd 5 dilate 3.png" alt="plot" width='720'/>

##### Using Euclidean with 79 Histogram Kernel Size and 5 BFS Stride (Image Diluted to fill Empty values)

<img src="./images/euclidean kernel 79 stride 5.png" alt="plot" width='720'/>

##### Using Euclidean with 39 Histogram Kernel Size and 5 BFS Stride (Image Diluted to fill Empty values)

<img src="./images/euclidean kernel 39 stride 5.png" alt="plot" width='720'/>

*main.py uses the chi square function, funcs_lbp_first_eucl.py is a temporary file using the euclidean distance*

*code hasn't been fully optimized*

