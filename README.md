### **2D SDF calculation from a Binary Mask**
***

We introduce a 2D Signed Distance Function (SDF) which calculates the minimum distances between the points of a grid from a shape which is obtained from a binary mask. In the SDF approach the points of the grid which lies outside the shape will have a certain minimum distance from the boundary of the shape itself. Approaching the boundary decreases the minimum distance up to zero. When the boundary is crossed the distances become negative, sign that the considered point of the grid lies inside the shape. 
The program will calculate the SDF for each shape given in the input and will return the SDF related to all the shapes. 

### **Structure of the Program**
***
We start from a binary mask, first step is to identify whether we have multiple shapes or not. The shapes are then returned one by one. 
Each shape is a different binary mask, which is formed by single pixels. From each pixel a unitary cube (side equal to 1 unit) is created, and afterwards these cubes are merged to create the selected shape. 
From here the 2D SDF is calculated for each of them. 
Last step is to find the minimum distance of each point of the grid in the case of multiple shapes. 
The shape is plotted. 

### **The Project: Main Files**
***
1. [sdf_from_binary_mask.py](https://github.com/Luigi-Rugg95/ProjectSDF/blob/main/sdf_from_binary_mask.py) : in this file are contained all the functions necessary to calculate the 2D SDF. They are embedded in the class sdf_from_binary_mask, which has as attribute the input _binary mask_ and the _grid finess_, moreover we have as attribute the list of distances for each iterated shape. A method to generate a grid of points, and all the others necessary to generate the 2D SDF. The method sdf() returns the final distances which can then be plotted. 

2. [plotting.py](https://github.com/Luigi-Rugg95/ProjectSDF/blob/main/plotting.py): this file contains the method for plotting the SDF, given as input the output of [sdf()](https://github.com/Luigi-Rugg95/ProjectSDF/blob/main/sdf_from_binary_mask.py) and the grid. A personal color map is created also and the SDF is saved in a png file at 300 dpi. 

3. [main.py](https://github.com/Luigi-Rugg95/ProjectSDF/blob/main/main.py) : this file is used for running the calculations

4. [testing.py](https://github.com/Luigi-Rugg95/ProjectSDF/blob/main/testing.py) : this file contains all the test which are made. Considering how the program works, test are done on simple input: one unitary cube (labelled 1), two unitary cube side by side (labelled 2), two unitary cube separated (labelled 3). In this way we can check the reliability of the calculation of the sdf calculation, the reliability when multiple shapes are found, and in the case in which a shape is formed by more then a unitary cube (one pixel).

to run the script: 

pip install -r requirements.txt

run the main, where an input segmentation and grid_fines can be specified

SDF saved

### **Examples**
***
In the following we have an example of a single shape obtained from the input :
```Python
segmentation = numpy.array([[1,0,0],[1,1,0],[1,1,1]])
grid_finess = [ 1, 0.5, 0.1, 0.01] #from left to right
```

<img src="https://github.com/Luigi-Rugg95/ProjectSDF/blob/8088f94b767b2ceebdde9d593e12015039e2b227/Examples/shape_grid_1.png" alt="drawing" width="200"/><img src="https://github.com/Luigi-Rugg95/ProjectSDF/blob/91a5e337704ae9b28428e72b9db9494309a72a08/Examples/shape_grid_05.png" alt="drawing" width="200"/><img src="https://github.com/Luigi-Rugg95/ProjectSDF/blob/91a5e337704ae9b28428e72b9db9494309a72a08/Examples/shape_grid_01.png" alt="drawing" width="200"/><img src="https://github.com/Luigi-Rugg95/ProjectSDF/blob/91a5e337704ae9b28428e72b9db9494309a72a08/Examples/shape_grid_001.png" alt="drawing" width="200"/>

Example of two shapes obtained from the input: 
```Python
segmentation = numpy.array([[1,0,0],[0,0,0],[1,1,1]])
grid_finess = 0.01
```
<img src="https://github.com/Luigi-Rugg95/ProjectSDF/blob/f9e86612b303f9d5d2b009050bf152a9ed88443e/Examples/shape2_grid_001.png" alt="drawing" width="400"/>

Example of three shapes obtained from the input: 
```Python
segmentation = numpy.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
])
grid_finess = 0.01
```
<img src="https://github.com/Luigi-Rugg95/ProjectSDF/blob/f76813beb77e2930861e445be7cfd236e1e9f563/Examples/shape3_grid_001.png" alt="drawing" width="600"/>



