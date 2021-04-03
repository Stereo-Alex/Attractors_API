
import datashader
import numpy as np, pandas as pd, datashader as ds
from datashader import transfer_functions as tf
from datashader.colors import inferno, viridis
from numba import jit
from math import sin, cos, sqrt, fabs
import yaml
import moviepy.video.io.ImageSequenceClip


from colorcet import palette
palette["viridis"]=viridis
palette["inferno"]=inferno





@jit(nopython=True)
def Clifford(x, y, a, b, c, d, *o):
    return sin(a * y) + c * cos(a * x), sin(b * x) + d * cos(b * y)




@jit(nopython=True)
def Svensson(x, y, a, b, c, d, *o):
    return d * sin(a * x) - sin(b * y),            c * cos(a * x) + cos(b * y)


@jit(nopython=True)
def Bedhead(x, y, a, b, *o):
    return sin(x*y/b)*y + cos(a*x-y),            x + sin(y)/b


@jit(nopython=True)
def Fractal_Dream(x, y, a, b, c, d, *o):
    return sin(y*b)+c*sin(x*b),            sin(x*a)+d*sin(y*a)



@jit(nopython=True)
def Hopalong1(x, y, a, b, c, *o):
    return y - sqrt(fabs(b * x - c)) * np.sign(x),            a - x
@jit(nopython=True)
def Hopalong2(x, y, a, b, c, *o):
    return y - 1.0 - sqrt(fabs(b * x - 1.0 - c)) * np.sign(x - 1.0),           a - x - 1.0


n=100000000

@jit(nopython=True)
def trajectory_coords(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=n):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in np.arange(n-1):
        x[i+1], y[i+1] = fn(x[i], y[i], a, b, c, d, e, f)
    return x,y

def trajectory(fn, x0, y0, a, b=0, c=0, d=0, e=0, f=0, n=n):
    x, y = trajectory_coords(fn, x0, y0, a, b, c, d, e, f, n)
    return pd.DataFrame(dict(x=x,y=y))



#get_ipython().run_cell_magic('time', '', 'df = trajectory(Clifford, 0, 0, -1.3, -1.3, -1.8, -1.9)')




#get_ipython().run_cell_magic('time', '', '\ncvs = ds.Canvas(plot_width = 900, plot_height = 900)\nagg = cvs.points(df, \'x\', \'y\')\nprint(agg.values[190:195,190:195],"\\n")')




df = trajectory(Clifford, 0, 0, -1.3, -1.3, -1.8, -1.9)


cvs = ds.Canvas(plot_width = 1000, plot_height = 1000)
agg = cvs.points(df, 'x', 'y')
print(agg.values[190:195,190:195],"\n")


ds.transfer_functions.Image.border=0

img = tf.shade(agg, cmap = viridis)





def dsplot(fn, vals, n=n, cmap=viridis, label=True):
    """Return a Datashader image by collecting `n` trajectory points for the given attractor `fn`"""
    lab = ("{}, "*(len(vals)-1)+" {}").format(*vals) if label else None
    df  = trajectory(fn, *vals, n=n)
    cvs = ds.Canvas(plot_width = 500, plot_height = 500)
    agg = cvs.points(df, 'x', 'y')
    img = tf.shade(agg, cmap=cmap, name=lab)
    return img




vals = yaml.load(open("strange_attractors.yml","r"), Loader=yaml.FullLoader)

def args(name):
    """Return a list of available argument lists for the given type of attractor"""
    return [v[1:] for v in vals if v[0]==name]  

def plot(fn, vals=None, **kw):

    vargs=args(fn.__name__) if vals is None else vals
    return tf.Images(*[dsplot(fn, v[1:], cmap=palette[v[0]][::-1], **kw) for v in vargs]).cols(1)





@jit(nopython=True)
def De_Jong(x, y, a, b, c, d, *o):
    return sin(a * y) - cos(b * x),            sin(c * x) - cos(d * y)



import numpy.random
num = 1
im = 0

#Clifford
func = De_Jong#Svensson
n = 1e10 
while np.array(im).sum() < 1e13: #how intresting/colorful #1e12 lower limit
    rvals=np.c_[np.zeros((num,2)), numpy.random.random((num,4))*4-2]
    vals = list(rvals[0])

    im = dsplot(func, rvals[0])
    print(np.array(im).sum()) 

#print(rvals)

plot(func, vals=[["bmw"]+list(rvals[i]) for i in range(len(rvals))], label=False)


# In[16]:


def myplot(fn, vals, n=n, cmap=viridis, label=True):
    """Return a Datashader image by collecting `n` trajectory points for the given attractor `fn`"""
    imgs = []
    lab = ("{}, "*(len(vals)-1)+" {}").format(*vals) if label else None
    df  = trajectory(fn, *vals, n)
    for i in np.geomspace(200, n, 50).astype(int):
        print(i)
        cvs = ds.Canvas(plot_width = 500, plot_height = 500)
        agg = cvs.points(df[:i], 'x', 'y')
        imgs.append(tf.shade(agg, cmap=palette["bmy"], name=lab))
    return imgs

import numpy.random

func = De_Jong #Svensson
n = 1e14
imgs_l = myplot(func, rvals[0])

tf.Images(*[i for i in imgs_l])




from PIL import Image  
import PIL  


for i in range(len(imgs_l)):
    imgs_l[i].to_pil().save('images/{}.png'.format(str(i).zfill(3)))

import glob
from PIL import Image

# filepaths

fp_in = "images/*.png"
fp_out = "flip.gif"


img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=250, loop=0)




list_gen = []

for i in range(0, 50):
    i = str(i)
    if len(i) <= 1:
        i = str(f'images/00{i}.png')
        list_gen.append(i)
    else:
        i = str(f'images/0{i}.png')
        list_gen.append(i) 

import os
image_folder='images'
fps=10

image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".png")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(list_gen, fps=fps)

clip.write_videofile('my_video.mp4')




