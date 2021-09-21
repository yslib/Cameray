import numpy as np
import taichi as ti
from realistic import RealisticCamera
ti.init(arch=ti.gpu)

pos = [10000.0,10000.0,10000.0]   # mm
center = [0.0,0.0,0.0]
world_up = [0.0,1.0,0.0]
resolution = [400, 400]
cam = RealisticCamera(resolution, pos, center, world_up)

pos = [10.0,10.0,10.0 ,1.0]
center = [0.0,0.0,0.0, 1.0]
up = [0.0,1.0, 0.0,1.0]


v = ti.Vector([0.0,0.0,0.0,0.0])
v = ti.Vector([*pos])
matr = ti.Matrix([pos,pos,pos,pos])

@ti.kernel
def test_focal_length():
    print('front z: ', cam.front_z())
    print('lens z: ', cam.rear_z())

    for i in ti.static(range(3)):
        x = 0.1 + i * 0.1
        so = ti.Vector([x, 0.0, 2000.0])
        sd = ti.Vector([0.0, 0.0, -1.0])
        fo = ti.Vector([x, 0.0, cam.rear_z() - 1.0])
        fd = ti.Vector([0.0,0.0,1.0])
        ok1, o1, d1 = cam.gen_ray_from_scene(so, sd)
        ok2, o2, d2 = cam.gen_ray_from_film(fo, fd)
        tf = -o1.x / d1.x
        tf2 = -o2.x / d2.x

    fz1,pz1, fz2, pz2 = cam.compute_thick_lens_approximation()
    print('fz1, pz1, fz2, pz2: ', fz1, pz1, fz2, pz2)
    print('first focal length, second focal length', cam.get_focal_length())

@ti.kernel
def test_pupils():
    cam.refocus(1200)
    cam.recompute_exit_pupil()

# test_pupils()

cam.refocus(10000.0)
cam.recompute_exit_pupil()

@ti.kernel
def gen_ray_test():
    cam.vignet[None] = 0
    for i in range(600):
        for j in range(400):
            weight, ray = cam.gen_ray_of(i, j)
            if weight > 0.0:
                print(i, j)

gen_ray_test()
