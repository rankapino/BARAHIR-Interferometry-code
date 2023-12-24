BARAHIR Image Code v9.0
------------------------------------------------
You can run the code as:
python BARAHIR_v9.py

The code runs well up to 95%
There are some bugs related mainly with CLEAN and TGSVD. 

# image reconstruction with the following method:
method = 'mtsvd' # Modified Truncated SVD Method (MTSVD) 
method = 'tgsvd' # Truncated Gereralizated SVD Method (TGSVD)
method = 'tsvd' # Truncated SVD Method (TSVD). Only with GCV, NCP and QOC
method = 'ttls' # Truncated Total Least Squares Method (TTLS)
method = 'tikh' # Tikhonov Method
method = 'disc' # Morozov Discrepancy Method
method = 'dsvd' # Dumped SVD Method

# Other Methods
method = 'lsqlin' # Linear Least-Sauare Method (LSQLIN)
method = 'logmart' # parallel LOG-entropy Multiplicative Algebraic Reconstruction Technique (LOGMART)
method = 'kaczmarz' # Kaczmarz's Method
method = 'maxent' # Maximum Entropy Method
method = 'lanczos' # Lanczos Bidiagonalization Method
method = 'clean' # CLEAN Method (Hogbom Algorithm)


# method to get the regularization parameter
param = 'gcv' # Generalized Cross-Validation (GCV)
param = 'ncp' # Normalized Cumulative Periodograms (NCP)
param = 'lcurve' # L-Curve
param = 'qoc' # Quasi-Optimality Criterion (QOC)

# The beam can be calculated with
AF
BF
BT

# There are the following examples
flagtest = 1 # Point at zenith
flagtest = 2 # Point at 60 degrees south
flagtest = 3 # Point at 60 degrees east
flagtest = 4 # Snake Eyes
flagtest = 5 # Straight line
flagtest = 6 # Parallel lines
flagtest = 7 # Perpendicular lines
flagtest = 8 # Centered square
flagtest = 9 # Centered rectangle
flagtest = 10 # Centered circle
flagtest = 11 # Centered ellipse
flagtest = 12 # Centered equilateral triangle
flagtest = 13 # Centered rectangle triangle
flagtest = 14 # Off center rectangle triangle
flagtest = 15 # Off center trapezoid rectangle
flagtest = 16 # Crab Nebula
#flagtest = 17 # Lena

