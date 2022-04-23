### **2D SDF calculation from a Binary Mask**
***

We introduce a 2D Signed Distance Function (SDF) which calculates the minimum distances between the points of a grid from a shape which is obtained from a binary mask. In the SDF approach the points of the grid which lies outside the shape will have a certain minimum distance from the boundary of the shape itself. Approaching the boundary decreases the minimum distance up to zero. When the boundary is crossed the distances become negative, sign that the considered point of the grid lies inside the shape. The program will calculate the SDF for each shape given in the input and will return the SDF related to all the shapes. 

### **Structure of the Program**
***