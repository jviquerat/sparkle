# Generic imports
import math
import numpy as np
from   shapely.geometry import Polygon

# Custom imports
from sparkle.env.base_env import *

###############################################
### Environment for packing
class packing(base_env):

    # Create object
    def __init__(self, cpu, path, pms):

        # Fill structure
        self.name      = 'packing'
        self.base_path = path
        self.cpu       = cpu

        self.obj_type = "sphere"
        self.n_objs   = 6
        self.max_side = 7.0
        self.obj_size = 1.0

        if hasattr(pms, "obj_type"): self.obj_type = pms.obj_type
        if hasattr(pms, "obj_size"): self.obj_size = pms.obj_size
        if hasattr(pms, "n_objs"):   self.n_objs   = pms.n_objs
        if hasattr(pms, "max_side"): self.max_side = pms.max_side

        if (self.obj_type == "sphere"):
            self.dim  = 2*self.n_objs
            self.xmin = np.zeros(self.dim) + self.obj_size
            self.xmax = np.ones(self.dim)*self.max_side - self.obj_size

        if (self.obj_type == "triangle"):
            self.dim  = 3*self.n_objs
            self.xmin = np.zeros(self.dim)
            self.xmax = np.ones(self.dim)

        # For initial plotting
        self.dx_min   = 0.0
        self.dy_min   = 0.0
        self.min_side = self.max_side

        self.it_plt = 0

    # Reset environment
    def reset(self, run):

        self.path   = self.base_path+"/"+str(run)
        self.it_plt = 0

        return True

    # Cost function
    def cost(self, x):

        xc = self.convert(x)

        if (self.obj_type == "sphere"):
            lx = np.reshape(x, (-1,2))
            ri = self.cost_intersect_sphere(lx)
            rs = self.cost_min_side_sphere(lx)

        if (self.obj_type == "triangle"):
            lx     = np.reshape(x, (-1))
            j      = 0
            coords = np.zeros((self.n_objs,3,2))
            for i in range(self.n_objs):
                x = lx[j]
                y = lx[j+1]
                t = lx[j+2]

                coords[i,:,:] = self.get_tri_coords(x, y, t)
                j            += 3

            ri = self.cost_intersect_triangle(coords)
            rs = self.cost_min_side_triangle(coords)

        return ri+rs

    # Convert input x
    def convert(self, x):

        xc = np.zeros_like(x)

        xm = 0.0           + self.obj_size/math.sqrt(3.0)
        xp = self.max_side - self.obj_size/math.sqrt(3.0)
        ym = 0.0           + self.obj_size/math.sqrt(3.0)
        yp = self.max_side - self.obj_size/math.sqrt(3.0)
        tm =-math.pi/3.0
        tp = math.pi/3.0

        j = 0
        for i in range(self.n_objs):
            xc[j]   = xm + x[j  ]*(xp - xm)
            xc[j+1] = ym + x[j+1]*(yp - ym)
            xc[j+2] = tm + x[j+2]*(tp - tm)
            j += 3

        return xc

    # Get vertices coordinates from (cx, yt, angle)
    def get_tri_coords(self, x, y, t):

        xx = np.zeros((3,2))

        # Compute coordinates for centered triangle and shift to center
        xx[0,0] = (self.obj_size/math.sqrt(3.0))*math.cos(t) + x
        xx[0,1] = (self.obj_size/math.sqrt(3.0))*math.sin(t) + y
        xx[1,0] = (self.obj_size/math.sqrt(3.0))*math.cos(t + 2.0*math.pi/3.0) + x
        xx[1,1] = (self.obj_size/math.sqrt(3.0))*math.sin(t + 2.0*math.pi/3.0) + y
        xx[2,0] = (self.obj_size/math.sqrt(3.0))*math.cos(t - 2.0*math.pi/3.0) + x
        xx[2,1] = (self.obj_size/math.sqrt(3.0))*math.sin(t - 2.0*math.pi/3.0) + y

        return xx

    # Compute intersection cost
    def cost_intersect_sphere(self, x):

        r = 0.0

        # Check intersections
        dsum = 0.0
        for i in range(self.n_objs):
            for j in range(self.n_objs):
                if (i == j): continue

                x0    = x[i,0]
                y0    = x[i,1]
                x1    = x[j,0]
                y1    = x[j,1]
                d     = math.sqrt((x1-x0)**2 + (y1-y0)**2)

                # Optimal is d=2*radius
                d -= 2.0*self.obj_size
                if (d < 0.0): r += 5.0*abs(d)

        return r

    # Compute intersection cost
    def cost_intersect_triangle(self, coords):

        r = 0.0

        # Generate polygons
        plist = []
        for i in range(self.n_objs):
            p = Polygon([(coords[i,0,0], coords[i,0,1]),
                         (coords[i,1,0], coords[i,1,1]),
                         (coords[i,2,0], coords[i,2,1])])
            plist.append(p)

        # Check intersections
        dsum = 0.0
        for i in range(self.n_objs):
            for j in range(i+1,self.n_objs):
                d = plist[i].intersection(plist[j]).area
                d = math.sqrt(d)
                if (d > 0.0): r += 100.0*abs(d)

        return r

    # Compute minimal side cost
    def cost_min_side_sphere(self, x):

        # Compute min square side
        dx_min = 1.0e8
        dx_max =-1.0e8
        dy_min = 1.0e8
        dy_max =-1.0e8

        for i in range(self.n_objs):
            x0 = x[i,0]
            y0 = x[i,1]
            dx_min = min(x0, dx_min)
            dx_max = max(x0, dx_max)
            dy_min = min(y0, dy_min)
            dy_max = max(y0, dy_max)

        dx_min -= self.obj_size
        dy_min -= self.obj_size
        dx_max += self.obj_size
        dy_max += self.obj_size

        self.dx_min   = dx_min
        self.dy_min   = dy_min
        self.min_side = max(dx_max-dx_min, dy_max-dy_min)

        return self.min_side

    # Compute minimal side cost
    def cost_min_side_triangle(self, coords):

        # Compute min square side
        x_min = 1.0e8
        x_max =-1.0e8
        y_min = 1.0e8
        y_max =-1.0e8

        for i in range(self.n_objs):
            for k in range(3):
                x_min = min(coords[i,k,0], x_min)
                x_max = max(coords[i,k,0], x_max)
                y_min = min(coords[i,k,1], y_min)
                y_max = max(coords[i,k,1], y_max)

        self.dx_min   = x_min
        self.dy_min   = y_min
        self.min_side = max(x_max-x_min, y_max-y_min)

        return self.min_side

    # Rendering
    def render(self, x, c):

        if (self.it_plt == 0):
            os.makedirs(self.path+'/png', exist_ok=True)

        # Axis
        plt.clf()
        plt.cla()
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-0.5*self.max_side, 1.5*self.max_side])
        ax.set_ylim([-0.5*self.max_side, 1.5*self.max_side])
        ax.set_axis_off()
        fig.tight_layout()
        plt.margins(0,0)
        plt.subplots_adjust(0,0,1,1,0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        # Add rectangle
        rct = plt.Rectangle((-0.5*self.max_side,-0.5*self.max_side),
                            2.0*self.max_side, 2.0*self.max_side,
                            fc='none', ec='black', lw=2)
        plt.gca().add_patch(rct)

        # Add objects
        if (self.obj_type == "sphere"):
            lx = np.reshape(x, (-1,2))
            s  = self.cost_min_side_sphere(lx)

            for i in range(self.n_objs):
                circle = plt.Circle((lx[i,0], lx[i,1]), self.obj_size,
                                    fc='gray', ec='black')
                ax.add_patch(circle)

        if (self.obj_type == "triangle"):
            lx     = np.reshape(x, (-1))
            j      = 0
            coords = np.zeros((self.n_objs,3,2))
            for i in range(self.n_objs):
                x = lx[j]
                y = lx[j+1]
                t = lx[j+2]

                coords[i,:,:] = self.get_tri_coords(x, y, t)
                j            += 3
            s = self.cost_min_side_triangle(coords)

            for i in range(self.n_objs):
                triangle = plt.Polygon(coords[i,:,:], fc='gray', ec='black')
                ax.add_patch(triangle)

        # Add square
        rectangle = plt.Rectangle((self.dx_min, self.dy_min), s, s,
                                  fc='none', ec='blue', lw=2)
        plt.gca().add_patch(rectangle)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        # Save figure and close
        filename = self.path+"/png/"+str(self.it_plt)+".png"
        plt.axis('off')
        plt.savefig(filename, dpi=100,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        self.it_plt += 1

    # Close environment
    def close(self):
        pass
