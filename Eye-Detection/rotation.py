from numpy import *
from math import sqrt

# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

class Rotation3d:
    def __init__(self, w, x, y, z, wp, xp, yp, zp):
        self.A = array([w, x, y, z])
        self.B = array([wp, xp, yp, zp])
        self.N = self.A.shape[0]
        
    def centeroidnp(self, M):
        length = M.shape[0]
        sum_x = sum(M[:, 0])
        sum_y = sum(M[:, 1])
        sum_z = sum(M[:, 2])
        return sum_x/length, sum_y/length, sum_z/length

    def rigid_transform_3D(self):

        N = len(self.A) # total points

        centroid_A = self.centeroidnp(self.A)
        centroid_B = self.centeroidnp(self.B)
        
        # centre the points
        AA = self.A - tile(centroid_A, (N, 1))
        BB = self.B - tile(centroid_B, (N, 1))

        # dot is matrix multiplication for array
        H = dot(transpose(AA),BB)

        U, S, Vt = linalg.svd(H)
        print(U, S, Vt)

        R = dot(transpose(Vt),transpose(U))

        # special reflection case
        if linalg.det(R) < 0:
            print("Reflection detected")
            Vt[2,:] *= -1
            R = dot(transpose(Vt),transpose(U))

        t = dot(-R,transpose(centroid_A)) + transpose(centroid_B)

        return( R, t)

"""
# Test with random data

# Random rotation and translation
R = mat(random.rand(3,3))
t = mat(random.rand(3,1))

# make R a proper rotation matrix, force orthonormal
U, S, Vt = linalg.svd(R)
R = U*Vt

# remove reflection
if linalg.det(R) < 0:
   Vt[2,:] *= -1
   R = U*Vt

# number of points
n = 10

A = mat(random.rand(n,3))
B = R*A.T + tile(t, (1, n))
B = B.T

# recover the transformation
ret_R, ret_t = rigid_transform_3D(A, B)

A2 = (ret_R*A.T) + tile(ret_t, (1, n))
A2 = A2.T

# Find the error
err = A2 - B

err = multiply(err, err)
err = sum(err)
rmse = sqrt(err/n)

print "Points A"
print A
print ""

print "Points B"
print B
print ""

print "Rotation"
print R
print ""

print "Translation"
print t
print ""

print "RMSE:", rmse
print "If RMSE is near zero, the function is correct!"
"""