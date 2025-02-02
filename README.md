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

## Datasets

### Input
<img src="./images/road_1.jpg" alt="drawing" width='720'/>

### Output

<img src="./images/road_1_result.png" alt="drawing" width='720'/>
