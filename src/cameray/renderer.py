import sys
from numpy.lib.type_check import real
import taichi as ti
import time
import math
import numpy as np
from .renderer_utils import ray_aabb_intersection, intersect_sphere, ray_plane_intersect, reflect, refract
from .realistic import RealisticCamera

ti.init(arch=ti.gpu)
res = (600, 800)
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)
# color_buffer = ti.Vector.field(3, dtype=ti.f32)

# ti.root.dense(ti.ji,res).place(color_buffer)

count_var = ti.field(ti.i32, shape=(1, ))

max_ray_depth = 10
eps = 1e-4
inf = 1e10
fov = 0.8

camera_pos = ti.Vector([0.0, 1.8, 10.0])

mat_none = 0
mat_lambertian = 1
mat_specular = 2
mat_glass = 3
mat_light = 4

light_y_pos = 30.0 - eps
light_x_min_pos = -0.25
light_x_range = 5
light_z_min_pos = 100.0
light_z_range = 12
light_area = light_x_range * light_z_range
light_min_pos = ti.Vector([light_x_min_pos, light_y_pos, light_z_min_pos])
light_max_pos = ti.Vector([
    light_x_min_pos + light_x_range, light_y_pos,
    light_z_min_pos + light_z_range
])
light_color = ti.Vector(list(np.array([0.9, 0.85, 0.7])))
light_normal = ti.Vector([0.0, -1.0, 0.0])

# No absorbtion, integrates over a unit hemisphere
lambertian_brdf = 1.0 / math.pi
# diamond!
refr_idx = 2.4

# right near sphere
sp1 = [0.0, 2.0, 225.0]
sp1_center = ti.Vector(sp1)
sp1_radius = 2.0
# left far sphere
sp2_center = ti.Vector([-0.28, 0.55, 0.8])
sp2_radius = 0.32



pos = [0.0, 3.0, 240.0]   # mm
center = [0.0,0.0,0.0]
world_up = [0.0, 1.0, 0.0]
real_cam = RealisticCamera(pos, center, world_up)

def make_box_transform_matrices():
    rad = math.pi / 10.0
    c, s = math.cos(rad), math.sin(rad)
    rot = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    translate = np.array([
        [1, 0, 0, -0.7],
        [0, 1, 0, 0],
        [0, 0, 1, 0.7],
        [0, 0, 0, 1],
    ])
    m = translate @ rot
    m_inv = np.linalg.inv(m)
    m_inv_t = np.transpose(m_inv)
    return ti.Matrix(m_inv), ti.Matrix(m_inv_t)


# left box
box_min = ti.Vector([-4.0, -1.0, 15.0])
box_max = ti.Vector([4.0, 30.0, 25.00])
box_m_inv, box_m_inv_t = make_box_transform_matrices()


@ti.func
def intersect_light(pos, d, tmax):
    hit, t, _ = ray_aabb_intersection(light_min_pos, light_max_pos, pos, d)
    if hit and 0 < t < tmax:
        hit = 1
    else:
        hit = 0
        t = inf
    return hit, t


@ti.func
def ray_aabb_intersection2(box_min, box_max, o, d):
    # Compared to ray_aabb_intersection2(), this also returns the normal of
    # the nearest t.
    intersect = 1

    near_t = -inf
    far_t = inf
    near_face = 0
    near_is_max = 0

    for i in ti.static(range(3)):
        if d[i] == 0:
            if o[i] < box_min[i] or o[i] > box_max[i]:
                intersect = 0
        else:
            i1 = (box_min[i] - o[i]) / d[i]
            i2 = (box_max[i] - o[i]) / d[i]

            new_far_t = max(i1, i2)
            new_near_t = min(i1, i2)
            new_near_is_max = i2 < i1

            far_t = min(new_far_t, far_t)
            if new_near_t > near_t:
                near_t = new_near_t
                near_face = int(i)
                near_is_max = new_near_is_max

    near_norm = ti.Vector([0.0, 0.0, 0.0])
    if near_t > far_t:
        intersect = 0
    if intersect:
        # TODO: Issue#1004...
        if near_face == 0:
            near_norm[0] = -1 + near_is_max * 2
        elif near_face == 1:
            near_norm[1] = -1 + near_is_max * 2
        else:
            near_norm[2] = -1 + near_is_max * 2

    return intersect, near_t, far_t, near_norm


@ti.func
def mat_mul_point(m, p):
    hp = ti.Vector([p[0], p[1], p[2], 1.0])
    hp = m @ hp
    hp /= hp[3]
    return ti.Vector([hp[0], hp[1], hp[2]])


@ti.func
def mat_mul_vec(m, v):
    hv = ti.Vector([v[0], v[1], v[2], 0.0])
    hv = m @ hv
    return ti.Vector([hv[0], hv[1], hv[2]])


@ti.func
def ray_aabb_intersection2_transformed(box_min, box_max, o, d):
    # Transform the ray to the box's local space
    obj_o = mat_mul_point(box_m_inv, o)
    obj_d = mat_mul_vec(box_m_inv, d)
    intersect, near_t, _, near_norm = ray_aabb_intersection2(
        box_min, box_max, obj_o, obj_d)
    if intersect and 0 < near_t:
        # Transform the normal in the box's local space to world space
        near_norm = mat_mul_vec(box_m_inv_t, near_norm)
    else:
        intersect = 0
    return intersect, near_t, near_norm


@ti.func
def intersect_scene(pos, ray_dir):
    closest, normal = inf, ti.Vector.zero(ti.f32, 3)
    c, mat = ti.Vector.zero(ti.f32, 3), mat_none

    # right near sphere
    cur_dist, hit_pos = intersect_sphere(pos, ray_dir, sp1_center, sp1_radius)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = (hit_pos - sp1_center).normalized()
        c, mat = ti.Vector([1.0, 1.0, 1.0]), mat_glass
    # left box
    hit, cur_dist, pnorm = ray_aabb_intersection2_transformed(
        box_min, box_max, pos, ray_dir)
    if hit and 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = ti.Vector([0.8, 0.5, 0.4]), mat_specular

    # left
    pnorm = ti.Vector([1.0, 0.0, 0.0])
    cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([-40.0, 0.0,
                                                               0.0]), pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = ti.Vector([0.65, 0.05, 0.05]), mat_lambertian
    # right
    pnorm = ti.Vector([-1.0, 0.0, 0.0])
    cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([40.0, 0.0, 0.0]),
                                      pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = ti.Vector([0.12, 0.45, 0.15]), mat_lambertian
    # bottom
    gray = ti.Vector([0.93, 0.93, 0.93])
    pnorm = ti.Vector([0.0, 1.0, 0.0])
    cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([0.0, -1.0, 0.0]),
                                      pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = gray, mat_lambertian
    # top
    pnorm = ti.Vector([0.0, -1.0, 0.0])
    cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([0.0, 30.0, 0.0]),
                                      pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = gray, mat_lambertian
    # far
    pnorm = ti.Vector([0.0, 0.0, 1.0])
    cur_dist, _ = ray_plane_intersect(pos, ray_dir, ti.Vector([0.0, 0.0, 0.0]),
                                      pnorm)
    if 0 < cur_dist < closest:
        closest = cur_dist
        normal = pnorm
        c, mat = gray, mat_lambertian
    # light
    hit_l, cur_dist = intersect_light(pos, ray_dir, closest)
    if hit_l and 0 < cur_dist < closest:
        # technically speaking, no need to check the second term
        closest = cur_dist
        normal = light_normal
        c, mat = light_color, mat_light

    return closest, normal, c, mat


@ti.func
def visible_to_light(pos, ray_dir):
    a, b, c, mat = intersect_scene(pos, ray_dir)
    return mat == mat_light


@ti.func
def dot_or_zero(n, l):
    return max(0.0, n.dot(l))


@ti.func
def mis_power_heuristic(pf, pg):
    # Assume 1 sample for each distribution
    f = pf**2
    g = pg**2
    return f / (f + g)


@ti.func
def compute_area_light_pdf(pos, ray_dir):
    hit_l, t = intersect_light(pos, ray_dir, inf)
    pdf = 0.0
    if hit_l:
        l_cos = light_normal.dot(-ray_dir)
        if l_cos > eps:
            tmp = ray_dir * t
            dist_sqr = tmp.dot(tmp)
            pdf = dist_sqr / (light_area * l_cos)
    return pdf


@ti.func
def compute_brdf_pdf(normal, sample_dir):
    return dot_or_zero(normal, sample_dir) / math.pi


@ti.func
def sample_area_light(hit_pos, pos_normal):
    # sampling inside the light area
    x = ti.random() * light_x_range + light_x_min_pos
    z = ti.random() * light_z_range + light_z_min_pos
    on_light_pos = ti.Vector([x, light_y_pos, z])
    return (on_light_pos - hit_pos).normalized()


@ti.func
def sample_brdf(normal):
    # cosine hemisphere sampling
    # first, uniformly sample on a disk (r, theta)
    r, theta = 0.0, 0.0
    sx = ti.random() * 2.0 - 1.0
    sy = ti.random() * 2.0 - 1.0
    if sx >= -sy:
        if sx > sy:
            # first region
            r = sx
            div = abs(sy / r)
            if sy > 0.0:
                theta = div
            else:
                theta = 7.0 + div
        else:
            # second region
            r = sy
            div = abs(sx / r)
            if sx > 0.0:
                theta = 1.0 + sx / r
            else:
                theta = 2.0 + sx / r
    else:
        if sx <= sy:
            # third region
            r = -sx
            div = abs(sy / r)
            if sy > 0.0:
                theta = 3.0 + div
            else:
                theta = 4.0 + div
        else:
            # fourth region
            r = -sy
            div = abs(sx / r)
            if sx < 0.0:
                theta = 5.0 + div
            else:
                theta = 6.0 + div
    # Malley's method
    u = ti.Vector([1.0, 0.0, 0.0])
    if abs(normal[1]) < 1 - eps:
        u = normal.cross(ti.Vector([0.0, 1.0, 0.0]))
    v = normal.cross(u)

    theta = theta * math.pi * 0.25
    costt, sintt = ti.cos(theta), ti.sin(theta)
    xy = (u * costt + v * sintt) * r
    zlen = ti.sqrt(max(0.0, 1.0 - xy.dot(xy)))
    return xy + zlen * normal


@ti.func
def sample_direct_light(hit_pos, hit_normal, hit_color):
    direct_li = ti.Vector([0.0, 0.0, 0.0])
    fl = lambertian_brdf * hit_color * light_color
    light_pdf, brdf_pdf = 0.0, 0.0
    # sample area light
    to_light_dir = sample_area_light(hit_pos, hit_normal)
    if to_light_dir.dot(hit_normal) > 0:
        light_pdf = compute_area_light_pdf(hit_pos, to_light_dir)
        brdf_pdf = compute_brdf_pdf(hit_normal, to_light_dir)
        if light_pdf > 0 and brdf_pdf > 0:
            l_visible = visible_to_light(hit_pos, to_light_dir)
            if l_visible:
                w = mis_power_heuristic(light_pdf, brdf_pdf)
                nl = dot_or_zero(to_light_dir, hit_normal)
                direct_li += fl * w * nl / light_pdf
    # sample brdf
    brdf_dir = sample_brdf(hit_normal)
    brdf_pdf = compute_brdf_pdf(hit_normal, brdf_dir)
    if brdf_pdf > 0:
        light_pdf = compute_area_light_pdf(hit_pos, brdf_dir)
        if light_pdf > 0:
            l_visible = visible_to_light(hit_pos, brdf_dir)
            if l_visible:
                w = mis_power_heuristic(brdf_pdf, light_pdf)
                nl = dot_or_zero(brdf_dir, hit_normal)
                direct_li += fl * w * nl / brdf_pdf
    return direct_li


@ti.func
def schlick(cos, eta):
    r0 = (1.0 - eta) / (1.0 + eta)
    r0 = r0 * r0
    return r0 + (1 - r0) * ((1.0 - cos)**5)


@ti.func
def sample_ray_dir(indir, normal, hit_pos, mat):
    u = ti.Vector([0.0, 0.0, 0.0])
    pdf = 1.0
    if mat == mat_lambertian:
        u = sample_brdf(normal)
        pdf = max(eps, compute_brdf_pdf(normal, u))
    elif mat == mat_specular:
        u = reflect(indir, normal)
    elif mat == mat_glass:
        cos = indir.dot(normal)
        ni_over_nt = refr_idx
        outn = normal
        if cos > 0.0:
            outn = -normal
            cos = refr_idx * cos
        else:
            ni_over_nt = 1.0 / refr_idx
            cos = -cos
        has_refr, refr_dir = refract(indir, outn, ni_over_nt)
        refl_prob = 1.0
        if has_refr:
            refl_prob = schlick(cos, refr_idx)
        if ti.random() < refl_prob:
            u = reflect(indir, normal)
        else:
            u = refr_dir
    return u.normalized(), pdf


stratify_res = 5
inv_stratify = 1.0 / 5.0


@ti.kernel
def taichi_render():
    for u, v in color_buffer:
        weight, r = real_cam.gen_ray_of(v, res[0]-u)
        if weight <= 0.0:
            continue
        ray_dir = r[1]
        pos = r[0]

        acc_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])
        depth = 0
        while depth < max_ray_depth:
            closest, hit_normal, hit_color, mat = intersect_scene(pos, ray_dir)
            if mat == mat_none:
                break

            hit_pos = pos + closest * ray_dir
            hit_light = (mat == mat_light)
            if hit_light:
                acc_color += throughput * light_color
                break
            elif mat == mat_lambertian:
                acc_color += throughput * sample_direct_light(
                    hit_pos, hit_normal, hit_color)

            depth += 1
            ray_dir, pdf = sample_ray_dir(ray_dir, hit_normal, hit_pos, mat)
            pos = hit_pos + 1e-4 * ray_dir
            if mat == mat_lambertian:
                throughput *= lambertian_brdf * hit_color * dot_or_zero(
                    hit_normal, ray_dir) / pdf
            else:
                throughput *= hit_color
        color_buffer[u, v] += weight*acc_color


# gui = ti.GUI('Realistic camera', res)
last_t = time.time()
i = 0

real_cam.refocus(4.5)
# real_cam.refocus(np.linalg.norm(np.array(sp1) - real_cam.get_position()))
real_cam.recompute_exit_pupil()


class Renderer:
    def __init__(self):
        self.camera = None
        self._iter = 0

    def render(self):
        taichi_render()

    def clear(self):
        color_buffer.from_numpy(np.zeros(res))
        self._iter = 0

    def var(self):
        pass

    def refocus(self, depth):
        global real_cam
        real_cam.refocus(depth)

    def recompute_exit_pupil(self):
        global real_cam
        real_cam.recompute_exit_pupil()

    def get_camera(self):
        global real_cam
        return real_cam

    def get_color_buffer_to_numpy(self):
        return color_buffer.to_numpy()


# while gui.running:
#     taichi_render()
#     interval = 2000
#     if i % interval == 0 and i > 0:
#         img = color_buffer.to_numpy() * (1 / (i + 1))
#         img = np.sqrt(img / img.mean() * 0.24)
#         var = np.var(img)
#         if var < 0.11790:
#             ti.imwrite(img, 'output.png')
#             break
#         print("{:.2f} samples/s ({} iters, var={})".format(
#             interval / (time.time() - last_t), i, var))
#         last_t = time.time()
#         gui.set_image(img)
#         gui.show()
#     i += 1
