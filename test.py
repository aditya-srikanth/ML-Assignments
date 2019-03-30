import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
w = np.array([-0.03314521 -0.99945055])
z = np.linspace(-3,2,500)
points = np.column_stack([z,z])
projection_points = points*w
normal_point_1 = norm.pdf(z,0,1)
normal_point_2 = norm.pdf(z,1,1)
normal_vector_1 = np.column_stack([normal_point_1,normal_point_1])*np.transpose(w)
normal_vector_2 = np.column_stack([normal_point_2,normal_point_2])*np.transpose(w)
normal_1 = projection_points + normal_vector_1
normal_2 = projection_points + normal_vector_2
plt.plot(normal_1[::1,0],normal_1[::1,1],'-',color='b',alpha=0.5)
plt.plot(normal_2[::1,0],normal_2[::1,1],'-',color='b',alpha=0.5)
plt.show()
