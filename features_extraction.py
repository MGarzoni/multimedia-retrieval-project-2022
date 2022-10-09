"""
Extract the following descriptors and add them to the attributes files:
    Area A  		number of pixels inside the segmented shape
    Perimeter l  		number of pixels along the boundary of A
    Compactness c 	l2/(4pA); how close is the shape to a disk (circle)
    Circularity  		1/c; basically same as compactness, just a different scale
    Centroid  		average of (x,y) coordinates of all pixels in shape
    Rectangularity 	 	A/AOBB; how close to a rectangle the shape is 			(OBB=oriented bounding box)
    Diameter  		largest distance between any two contour points
    Eccentricity  		|1|/| 2|, where 1 and 2 are the eigenvalues of 			the shape covariance matrix (in 2D, similar for 3D)

"""
