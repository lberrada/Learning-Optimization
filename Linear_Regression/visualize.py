# coding utf-8

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def visualize(method, mu_values, r2_values, d_values = None):
    """visualize results with a plot adapted to the number of parameters"""
    
    if method != "EDE":
        
        # plot Adjusted R2 against mu
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(mu_values, r2_values)
        ax.set_xlabel('Regularization Parameter')
        ax.set_ylabel('Adjusted $R^2$')
        plt.title("Cross Validation for " + method + " regression")
        plt.show()
        
    else:
        # plot Adjusted R2 against mu and d
        fig = plt.figure(figsize=(10, 7))
        ax = fig.gca(projection='3d')
        x = mu_values
        y = d_values
        X, Y = np.meshgrid(x, y)
        Z = r2_values.transpose()

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                linewidth=0, antialiased=False)
        
#         ax.set_zlim(0.51, 0.53)
        ax.set_xlabel('Regularization Parameter')
        ax.set_ylabel('Dimension Parameter')
        ax.set_zlabel('Adjusted $R^2$')
        ax.zaxis.set_major_locator(plt.LinearLocator(10))
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
        
        fig.colorbar(surf, shrink=0.5, aspect=7, cmap=plt.cm.coolwarm)
        
        plt.show()