"""
Do not import this code in a "regular way!"
Instead do exec(open("gpu_par.py").read()) to call this module
"""
import ray
import os

ray.init(num_gpus=4) #if want to use less gpus - specify here

def get_gpu_id():
    try:
        gpu_id = ray.get_gpu_ids()[0]
    except:
        gpu_id = 0
    return gpu_id

# class RayObjs():
#     def __init__(self):
#         self.objs = []

ray_objs = []

def gpu_map(f, *args):
    @ray.remote(num_cpus=0.2, num_gpus=0.25)
    def f_gpu(*args):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        return f(*args)

    objs = [f_gpu.remote(*x) for x in zip(*args)]
    ray_objs.extend(objs)
    out = [ray.get(x) for x in objs]
    return out

def kill_gpu_processes():
    [ray.cancel(x) for x in ray_objs]