import face_alignment
from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotation import Rotation

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

input = io.imread('images/im2.jpg')
preds = fa.get_landmarks(input)[-1]

"""
Functions
"""
"""
Norm a vector
"""
def normv(v):
    return(v / np.sqrt(np.sum(v**2)))

"""
Gaze direction inferred by the vectors x and y: Here we suppose that the gaze direction is orthogonal to the face plan.

We use the function cross() from the numpy library to get a vector orthogonal w to the plan formed by (u,v)
"""
h = int(np.linalg.norm(preds[45,:] - preds[36,:]))
v = int(np.linalg.norm(preds[27,:] - preds[51,:]))


w, x, y, z = preds[36,:], preds[45,:], preds[27,:], preds[51,:]
wp, xp, yp, zp = [-h//2, 0, 0], [h//2, 0, 0], [0, 100, 0], [0, 100 + v, 0]

print("w:", w, "x:", x, "y:", y, "z:", z)
print("wp:", wp, "xp:", xp, "yp:", yp, "zp:", zp)

rot = Rotation(w, x, y, z, wp, xp, yp, zp)
R, t = rot.rigid_transform_3D()

print(R)

R_2d = np.array([R[0,:2],R[1,:2]])
t_2d = t[:2]

np.save("R_2d", R_2d)
np.save("t_2d", t_2d)

u = normv(x)
v = normv(y)
w = np.cross(u,v)

if w[2] < 0:
    w = w * (-1)

"""
rot = Rotation(u, v, w)
angles = rot.find_angles()
print(np.dot(u,v), np.dot(w,v), np.dot(u,w))
"""

"""
Apply the rotations to all the face landmarks
"""
def apply_rot(M, R, t):
    l = len(M)
    c = len(M[0])
    N = np.array([[0 for k in range(c)] for i in range(l)])
    for i in range(l):
        N[i] = np.dot(R, M[i]) + t
    return N

pred_rod = apply_rot(preds, R, t)

"""
Get the centre of the eyes
"""
left_eye = (preds[45,:] + preds[42,:])//2
right_eye = (preds[39,:] + preds[36,:])//2

"""
Coordinates of the end of the line
"""
end_line_left = left_eye + 300 * w
end_line_right = right_eye + 300 * w

"""
Line
"""
line_left = np.array([left_eye,end_line_left])
line_right = np.array([right_eye, end_line_right])

print(line_left[:,0])

"""
Figure


center = pred_rod[30,:]
"""

fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(input)
"""
ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=2,linestyle='-',color='w',lw=1)
ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=2,linestyle='-',color='w',lw=1)
ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=2,linestyle='-',color='w',lw=1)
ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=2,linestyle='-',color='w',lw=1)
ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=2,linestyle='-',color='w',lw=1)
ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=2,linestyle='-',color='w',lw=1)
ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=2,linestyle='-',color='w',lw=1)
ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=2,linestyle='-',color='w',lw=1)
ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=2,linestyle='-',color='w',lw=1) 
ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=2,linestyle='-',color='w',lw=1) 
ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=2,linestyle='-',color='w',lw=1) 
ax.plot(line_left[:,0],line_left[:,1],marker='o',markersize=2,linestyle='-',color='b',lw=1)
ax.plot(line_right[:,0],line_right[:,1],marker='o',markersize=2,linestyle='-',color='b',lw=1)  
"""
ax.axis('off')

eyes_position = input[int(preds[21,1]):int(preds[28,1]), int(preds[36,0]):int(preds[45,0])]
cv2.imwrite("eyes.png", eyes_position)

ax = fig.add_subplot(1, 2, 2, projection='3d')

ax.scatter(pred_rod[:,0]*1.2,pred_rod[:,1],pred_rod[:,2],c="cyan", alpha=1.0, edgecolor='b')
"""
ax.plot3D(line_left[:,0]*1.2,line_left[:,1],line_left[:,2], color='blue')
ax.plot3D(line_right[:,0]*1.2,line_right[:,1],line_right[:,2], color='blue')

ax.scatter(pred_rod[:,0]*1.2,pred_rod[:,1],pred_rod[:,2],c="red", alpha=1.0, edgecolor='b')

ax.plot3D(np.array([preds[30,0],preds[30,0] + x[0]])*1.2,np.array([preds[30,1],preds[30,0] + x[1]]), np.array([0,x[2]]), color='blue' )
ax.plot3D(np.array([preds[30,0],preds[30,0] + y[0]])*1.2,np.array([preds[30,1],preds[30,0] + y[1]]), np.array([0,y[2]]), color='blue' )
ax.plot3D(np.array([preds[30,0],preds[30,0] + z[0]])*1.2,np.array([preds[30,1],preds[30,0] + z[1]]), np.array([0,z[2]]), color='blue' )

ax.plot3D(np.array([0,base_x[0]])*1.2,np.array([0,base_x[1]]), np.array([0,base_x[2]]), color='blue' )
ax.plot3D(np.array([0,base_y[0]])*1.2,np.array([0,base_y[1]]), np.array([0,base_y[2]]), color='blue' )

ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )
"""

ax.view_init(elev=90., azim=90.)
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()