import math
import taichi as ti
import numpy as np
from typing import List, Dict
from .renderer_utils import refract

max_elements = 30
max_draw_rays = 20
pupil_interval_count = 64
eps = 1e-5
inf = 9999999.0

def convert_raw_data_from_dict(raw_data:List[List[float]]):
    """
    Args: raw_data
    for example:
    [
        # curvature radius, thickness, index of refraction, aperture diameter
        [29.475,3.76,1.67,25.2],
        [84.83,0.12,1,25.2],
        [19.275,4.025,1.67,23],
        [40.77,3.275,1.699,23],
        [12.75,5.705,1,18],
        [0,4.5,0,17.1],
        [-14.495,1.18,1.603,17],
        [40.77,6.065,1.658,20],
        [-20.385,0.19,1,20],
        [437.065,3.22,1.717,20],
        [-39.73,5.0,1,20]
    ]
    """
    dict_data = []
    for elem in raw_data:
        dict_data.append(
            {'curvature_radius':elem[0],
            'thickness':elem[1],
            'eta':elem[2],
            'aperture_diameter':elem[3]
            })
    return dict_data

def convert_dict_data_from_raw(dict_data:List[Dict[str, List[float]]]):
    raw_data = []
    for elem_dict in dict_data:
        raw_data.append([
            elem_dict.get('curvature_radius', 0),
            elem_dict.get('thickness', 0),
            elem_dict.get('eta', 0),
            elem_dict.get('aperture_diameter', 0)
            ])
    return raw_data

@ti.func
def lerp(val, begin ,end):
    return begin * (1.0 - val) + val * end

@ti.func
def bound_union_with(bmin, bmax, pos):
    return ti.min(bmin, pos), ti.max(bmax, pos)

@ti.func
def make_bound2():
    return ti.Vector([inf, inf]), ti.Vector([-inf, -inf])

@ti.func
def inside_aabb(bmin, bmax, pos):
    return all(bmin <= pos) and all(pos <= bmax)



@ti.data_oriented
class RealisticCamera:
    def __init__(self, camera_pos, center, world_up):
        self.vignet = ti.field(ti.i32, shape=())

        self.curvature_radius = ti.field(ti.f32)
        self.thickness = ti.field(ti.f32)
        self.eta = ti.field(ti.f32)
        self.aperture_radius = ti.field(ti.f32)
        self.exitPupilBoundMin = ti.Vector.field(2, dtype=ti.f32)
        self.exitPupilBoundMax = ti.Vector.field(2, dtype=ti.f32)
        self.draw_rays = ti.Vector.field(3, dtype=ti.f32)

        ti.root.dense(ti.i, (max_elements, )).place(self.curvature_radius, self.thickness, self.eta, self.aperture_radius)
        ti.root.dense(ti.i, (pupil_interval_count, )).place(self.exitPupilBoundMin, self.exitPupilBoundMax)
        ti.root.dense(ti.ij, (max_draw_rays, max_elements + 2)).place(self.draw_rays)

        self.camera2world_point = ti.Matrix([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
        self.camera2world_vec = ti.Matrix([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])

        self.shutter = 0.1
        self.film_width = 36
        self.film_height = 24
        self.film_diagnal = math.sqrt(self.film_height * self.film_height + self.film_width * self.film_width)
        self.pixel_width = 800
        self.pixel_height = 600
        self.camera_pos = np.array(camera_pos)

        self._elem_count = 0
        self.load_lens_data([])
        self.set_camera(camera_pos, center, world_up)

    def get_resolution(self):
        return [self.pixel_width, self.pixel_height]

    def get_position(self):
        return self.camera_pos

    def set_camera(self, eye, center, world_up):
        """
        setup camera2world_pos transformation
        """
        eye = np.array(eye)
        center = np.array(center)
        world_up = np.array(world_up)

        self.camera_pos = eye

        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm > 0.0 else np.array([0.0,0.0,0.0])

        direction = normalize(center - eye)
        right = normalize(np.cross(direction, world_up))
        up = normalize(np.cross(right, direction))

        self.camera2world_point = ti.Matrix([
            [1.0/1000.0,0.0,0.0,0.0],
            [0.0,1.0/1000.0,0.0,0.0],
            [0.0,0.0,1.0/1000.0,0.0],
            [0.0,0.0,0.0,1.0]
        ]) @ ti.Matrix([
            [right[0], up[0], direction[0], 1000.0 * eye[0]],
            [right[1], up[1], direction[1], 1000.0 * eye[1]],
            [right[2], up[2], direction[2], 1000.0 * eye[2]],
            [0.0,0.0,0.0,1.0]])

        self.camera2world_vec = ti.Matrix([
            [right[0], up[0], direction[0]],
            [right[1], up[1], direction[1]],
            [right[2], up[2], direction[2]]
            ])

    def get_element_count(self):
        return self._elem_count

    def load_lens_data(self, lenses:List[List[float]]=[[]]):
        wide22 = [
            # curvature radius, thickness, index of refraction, aperture diameter
            [35.98738, 1.21638, 1.54, 23.716],
            [11.69718, 9.9957, 1, 17.996],
            [13.08714, 5.12622, 1.772, 12.364],
            [22.63294, 1.76924, 1.617, 9.812],
            [71.05802, 0.8184, 1, 9.152],
            [0,        2.27766,0, 8.756],
            [9.58584,2.43254,1.617,8.184],
            [11.28864,0.11506,1,9.152],
            [166.7765,3.09606,1.713,10.648],
            [7.5911,1.32682,1.805,11.44],
            [16.7662,3.98068,1,12.276],
            [7.70286,1.21638,1.617,13.42],
            [11.97328,10,1,17.996]
        ]
        dgauss50 = [
            [29.475,3.76,1.67,25.2],
            [84.83,0.12,1,25.2],
            [19.275,4.025,1.67,23],
            [40.77,3.275,1.699,23],
            [12.75,5.705,1,18],
            [0,4.5,0,17.1],
            [-14.495,1.18,1.603,17],
            [40.77,6.065,1.658,20],
            [-20.385,0.19,1,20],
            [437.065,3.22,1.717,20],
            [-39.73,5.0,1,20]
        ]
        telephoto=[
            [21.851,0	,1.529,19.0],
            [-34.546,5.008,1.599,17.8],
            [108.705,1.502,1.0,16.6],
            [  0,1.127,0,16.2],
            [-12.852,26.965,1.613,12.6],
            [19.813,1.502,1.603,13.4],
            [-20.378,5.008,1.0,14.8]
        ]
        telephoto250=[
            [54.6275,12.52,1.529,47.5],
            [-86.365,3.755,1.599,44.5],
            [271.7625,2.8175,1,41.5],
            [0,67.4125,0,40.5],
            [-32.13,3.755,1.613,31.5],
            [49.5325,12.52,1.603,33.5],
            [-50.945,0,1,37]
        ]

        lenses = lenses if lenses else dgauss50
        self._elem_count = len(lenses)
        self._lenses_data = lenses.copy()
        for _ in range(max(0, max_elements - self._elem_count)):
            lenses.append([0 for j in range(4)])
        a = np.array(lenses).transpose()
        self.curvature_radius.from_numpy(a[0])
        self.thickness.from_numpy(a[1])
        self.eta.from_numpy(a[2])
        self.aperture_radius.from_numpy(a[3]/2.0)

    def get_lenses_data(self):
        a = self.curvature_radius.to_numpy()[0:self._elem_count]
        b = self.thickness.to_numpy()[0:self._elem_count]
        c = self.eta.to_numpy()[0:self._elem_count]
        d = self.aperture_radius.to_numpy()[0:self._elem_count] * 2.0
        return np.stack((a,b,c,d)).transpose()

    @ti.func
    def camera_2_world(self, o, d):
        """
        Transform the ray from camera space to world space

        o: origin of the ray
        d: direction of the ray
        """
        wo = self.camera2world_point @ ti.Vector([o.x, o.y, o.z, 1.0])
        wd = self.camera2world_vec @ d
        return ti.Vector([wo.x,wo.y,wo.z]), wd

    @ti.func
    def rear_z(self):
        return self.thickness[self._elem_count - 1]

    @ti.func
    def front_z(self):
        z = 0.0
        for i in self.thickness:
            z += self.thickness[i]
        return z

    @ti.func
    def rear_radius(self):
        return self.curvature_radius[self._elem_count - 1]

    @ti.func
    def rear_aperture(self):
        return self.aperture_radius[self._elem_count - 1]

    @ti.kernel
    def recompute_exit_pupil(self):
        """
        pre-process exit pupil of the lens system
        """

        rearZ = self.rear_z()
        if rearZ <= 0.0:
            print('Not focus')
        rearRadius = self.rear_aperture()
        samples = 1024 * 1024
        half = 2.0 * rearRadius
        proj_bmin, proj_bmax = ti.Vector([-half, -half]), ti.Vector([half, half])
        for i in range(pupil_interval_count):
            r0 = ti.cast(i, ti.f32) / pupil_interval_count * self.film_diagnal / 2.0
            r1 = ti.cast(i + 1, ti.f32) / pupil_interval_count * self.film_diagnal / 2.0
            bmin, bmax = make_bound2()
            count = 0
            for j in range(samples):
                u, v= ti.random(), ti.random()
                film_pos = ti.Vector([lerp(ti.cast(j, ti.f32)/samples, r0, r1), 0.0, 0.0])
                x, y = lerp(u, -half, half), lerp(v, -half, half)
                lens_pos = ti.Vector([x, y, rearZ])
                if inside_aabb(bmin, bmax, ti.Vector([x, y])):
                    ti.atomic_add(count, 1)
                else:
                    ok, _, _ = self.gen_ray_from_film(film_pos, (lens_pos - film_pos).normalized())
                    if ok:
                        bmin, bmax = bound_union_with(bmin,bmax, ti.Vector([x, y]))
                        ti.atomic_add(count, 1)

            if count == 0:
                bmin, bmax = proj_bmin, proj_bmax

            # extents pupil bound
            delta = 2 * (proj_bmax - proj_bmin).norm() / ti.sqrt(samples)
            bmin -= delta
            bmax += delta

            self.exitPupilBoundMin[i] = bmin
            self.exitPupilBoundMax[i] = bmax

    @ti.func
    def sample_exit_pupil(self, film_pos, uv):
        """
        filme_pos: sampled point on the film
        uv: 2d sample

        Returns the sample point in the lenses space and the pupil area
        """

        r = film_pos.norm()
        index = ti.cast(ti.min(r / self.film_diagnal * 2.0 * pupil_interval_count, pupil_interval_count - 1), ti.i32)
        bmin, bmax = self.exitPupilBoundMin[index], self.exitPupilBoundMax[index]
        area = (bmax - bmin).dot(ti.Vector([1.0, 1.0]))
        sampled = lerp(uv, bmin, bmax)
        sint = film_pos.y / r if abs(r) >= eps else 0.0
        cost = film_pos.x / r if abs(r) >= eps else 1.0
        return ti.Vector(
            [
                cost * sampled.x - sint * sampled.y,
                sint * sampled.x + cost * sampled.y,
                self.rear_z()
            ]), area

    @ti.func
    def gen_ray(self, film_uv, lens_uv):

        """
        film_uv: samples on film
        lens_uv: samples on lens

        Returns:
            weight: non-zero if exit ray exists otherwise returns 0
            (ro, rd): exit ray
            film_pos: sampled point on film
        """
        extent = ti.Vector([self.film_width, self.film_height])
        film_pos_xy = lerp(film_uv, ti.Vector([0.0, 0.0]), extent)

        film_pos_xy = ti.Vector([film_pos_xy.x - self.film_width/2.0, self.film_height /2.0 - film_pos_xy.y])
        lens_pos, area = self.sample_exit_pupil(film_pos_xy, lens_uv)
        film_pos = ti.Vector([film_pos_xy.x, film_pos_xy.y, 0.0])
        o, d = film_pos, lens_pos - film_pos
        exit, out_o ,out_d = self.gen_ray_from_film(o, d)

        weight = 0.0
        if not exit:
            self.vignet[None] += 1

        if exit:
            cost = out_d.z
            cos4t = cost * cost * cost * cost
            weight = self.shutter * cos4t * area /(self.rear_z() * self.rear_z())
        pos = lerp(film_uv, ti.Vector([0.0, 0.0]), ti.Vector([self.pixel_width, self.pixel_height]))
        return weight, self.camera_2_world(out_o, out_d), pos

    @ti.func
    def gen_ray_of(self, px, py):
        """
        px, py: pixel position of final image
        """
        lens_uv = ti.Vector([ti.random(), ti.random()])
        u, v = ti.cast(px, ti.f32) / self.pixel_width, ti.cast(py, ti.f32) / self.pixel_height
        film_uv = ti.Vector([u, v])

        extent = ti.Vector([self.film_width, self.film_height])
        film_pos_xy = lerp(film_uv, ti.Vector([0.0, 0.0]), extent)

        film_pos_xy = ti.Vector([film_pos_xy.x - self.film_width/2.0, self.film_height /2.0 - film_pos_xy.y])
        lens_pos, area = self.sample_exit_pupil(film_pos_xy, lens_uv)
        film_pos = ti.Vector([film_pos_xy.x, film_pos_xy.y, 0.0])
        o, d = film_pos, lens_pos - film_pos
        exit, out_o ,out_d = self.gen_ray_from_film(o, d)

        weight = 0.0
        if not exit:
            self.vignet[None] += 1

        if exit:
            cost = out_d.z
            cos4t = cost * cost * cost * cost
            weight = self.shutter * cos4t * area /(self.rear_z() * self.rear_z())

        return weight, self.camera_2_world(out_o, out_d)

    @ti.func
    def compute_cardinal_points(self, in_ro, out_ro, out_rd):
        """
        Returns the z coordinate of the principal plane the the focal point
        (fz, pz) in the lenses space
        note: input vectors are in camera space
        """
        tf = -out_ro.x / out_rd.x
        tp = (in_ro.x - out_ro.x) / out_rd.x
        return -(out_ro + out_rd * tf).z, -(out_ro + out_rd * tp).z


    @ti.func
    def compute_thick_lens_approximation(self):
        """
        Returns the focal length and the z of principal plane
        return fz1, pz1, fz2, pz2 in lense space
        """

        x = self.film_diagnal * 0.001
        so = ti.Vector([x, 0.0, self.front_z() + 1.0])
        sd = ti.Vector([0.0, 0.0, -1.0])
        fo = ti.Vector([x, 0.0, self.rear_z() - 1.0])
        fd = ti.Vector([0.0, 0.0, 1.0])
        ok1, o1, d1 = self.gen_ray_from_scene(so, sd)
        ok2, o2, d2 = self.gen_ray_from_film(fo, fd)
        assert ok1 == True and ok2 == True
        fz, pz = self.compute_cardinal_points(so, o1, d1)
        fz1, pz1 = self.compute_cardinal_points(fo, o2, d2)
        assert fz1 < pz1 and pz < fz
        return fz, pz, fz1, pz1

    @ti.func
    def get_focal_length(self):
        fz, pz ,fz1 ,pz1 = self.compute_thick_lens_approximation()
        return fz - pz, pz1 - fz1

    @ti.func
    def focus_thick_camera(self, focus_distance):
        """
        focus_distance > 0
        """
        fz1, pz1, fz2, pz2 = self.compute_thick_lens_approximation()
        f = fz1 - pz1
        assert f > 0
        z = -focus_distance
        delta = 0.5 * (pz2 - z + pz1 - ti.sqrt((pz2 - z - pz1)*(pz2-z-4*f-pz1) ))
        return self.thickness[self._elem_count - 1] + delta

    @ti.kernel
    def refocus(self, focus_distance:ti.f32):
        rf = self.focus_thick_camera(focus_distance * 1000.0)
        self.thickness[self._elem_count - 1] = rf

    @ti.func
    def intersect_with_sphere(self,
                            center,
                            radius,
                            ro,
                            rd):
        """
        center: z depth of lens sphere center
        """

        o = ro - ti.Vector([0.0, 0.0, center])
        ok = True

        # translate the sphere to original
        A = rd.x * rd.x + rd.y * rd.y + rd.z * rd.z
        B = 2 * ( rd.x * o.x + rd.y * o.y + rd.z * o.z)
        C = o.x * o.x + o.y * o.y + o.z * o.z - radius * radius
        delta = B * B - 4 * A * C
        if delta < 0:
            ok = False

        root_delta = ti.sqrt(delta)
        t = 0.0
        n = ti.Vector([0.0,0.0,0.0])

        if ok:
            q = 0.0
            if B < 0:
                q = -0.5 * (B - root_delta)
            else:
                q = -0.5 * (B + root_delta)
            t0, t1 = q/A , C/q
            if t0 > t1:
                t0, t1 = t1, t0

            # t0, t1 = (-B - root_delta) / (2 * A), (-B + root_delta) / (2 * A)

            closer = (rd.z > 0) ^ (radius < 0)
            t = ti.min(t0, t1) if closer else ti.max(t0, t1)
            if t < 0:
                ok = False
            if ok:
                n = (o + t * rd).normalized()
                n = -n if n.dot(-rd) < 0.0 else n

        return ok, t, n


    @ti.func
    def gen_ray_from_scene(self, ori, dir):
        ro, rd = ti.Vector([ti.cast(ori.x,ti.f32), ori.y, -ori.z]), ti.Vector([ti.cast(dir.x,ti.f32), dir.y, -dir.z]).normalized()
        elemZ = -self.front_z()
        ok = True
        t = 0.0
        n = ti.Vector([0.0,0.0,0.0])
        for _ in range(1):  # force the inner loop serialized so that break could be used
            for i in range(self._elem_count):
                is_stop = self.curvature_radius[i] == 0.0
                if is_stop:
                    t = (elemZ - ro.z) / rd.z
                else:
                    radius = self.curvature_radius[i]
                    centerZ = elemZ + radius
                    isect, t, n = self.intersect_with_sphere(centerZ, radius, ro, rd)
                    if not isect:
                        ok = False
                        break

                assert t > 0.0

                hit = ro + rd * t
                r = hit.x * hit.x + hit.y * hit.y
                if r > self.aperture_radius[i] * self.aperture_radius[i]:  # out of the element aperture
                    ok = False
                    break
                ro = ti.Vector([hit.x, hit.y, hit.z])

                if not is_stop:
                    # refracted by lens
                    etaI = 1.0 if i == 0 or self.eta[i - 1] == 0.0 else self.eta[i - 1]
                    etaT = self.eta[i] if self.eta[i] != 0.0 else 1.0
                    rd.normalized()
                    has_r, d = refract(rd, n, etaI/etaT)
                    if not has_r:
                        ok = False
                        break
                    rd = ti.Vector([d.x, d.y, d.z])

                elemZ += self.thickness[i]

        return ok, ti.Vector([ro.x, ro.y, -ro.z]), ti.Vector([rd.x, rd.y, -rd.z]).normalized()

    @ti.func
    def gen_ray_from_film(self, ori, dir):
        """
        Input ray is the initial ray sampled from film to the rear lens element.
        Returns True and the output ray if the ray could be pass the lens system
        or returns False
        """
        ro, rd = ti.Vector([ori.x, ori.y, -ori.z]), ti.Vector([dir.x, dir.y, -dir.z]).normalized()
        elemZ = 0.0
        ok = True
        t = 0.0
        n = ti.Vector([0.0,0.0,0.0])
        for _ in range(1):  # force the inner loop serialized so that break could be allowed
            for ii in range(self._elem_count):
                i = self._elem_count - ii - 1
                elemZ -= self.thickness[i]
                is_stop = self.curvature_radius[i] == 0.0
                if is_stop:
                    if rd.z >= 0.0:
                        ok = False
                        break
                    t = (elemZ - ro.z) / rd.z
                else:
                    radius = self.curvature_radius[i]
                    centerZ = elemZ + radius
                    isect, t, n = self.intersect_with_sphere(centerZ, radius, ro, rd)
                    if not isect:
                        ok = False
                        break

                assert t > 0.0
                hit = ro + rd * t
                r = hit.x * hit.x + hit.y * hit.y
                if r > self.aperture_radius[i] * self.aperture_radius[i]:  # out of the element aperture
                    ok = False
                    break

                ro = ti.Vector([hit.x, hit.y, hit.z])

                if not is_stop:
                    # refracted by lens
                    etaI = self.eta[i]
                    etaT = self.eta[i - 1] if i > 0 and self.eta[i - 1] != 0.0 else 1.0   # the outer of 0-th element is air, whose eta is 1.0
                    # rd.normalized()
                    has_r, d = refract(rd, n, etaI/etaT)
                    if not has_r:
                        ok = False
                        break
                    rd = ti.Vector([d.x, d.y, d.z])

        return ok, ti.Vector([ro.x, ro.y, -ro.z]), ti.Vector([rd.x, rd.y, -rd.z]).normalized()

    @ti.kernel
    def gen_draw_rays_from_film(self):
        """
        draw the bound ray
        """
        r = self.aperture_radius[self._elem_count - 1]
        step = 0.01
        count = ti.cast(r / step,ti.i32)
        for j in range(1):
            for i in range(count):
                y = r - i * step
                ori, dir = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([y, 0.0, self.rear_z()])
                ok, a, b = self.gen_ray_from_film(ori, dir)
                if ok:
                    self.draw_ray_from_film(ori, dir, 0)
                    break

    @ti.kernel
    def gen_draw_rays_from_scene(self):
        pass

    @ti.func
    def draw_ray_from_film(self, ori, dir, ind):
        ro, rd = ti.Vector([ori.x, ori.y, -ori.z]), ti.Vector([dir.x, dir.y, -dir.z]).normalized()
        elemZ = 0.0
        ok = True
        t = 0.0
        n = ti.Vector([0.0,0.0,0.0])
        for _ in range(1):  # force the inner loop serialized so that break could be allowed
            for ii in range(self._elem_count):
                i = self._elem_count - ii - 1
                elemZ -= self.thickness[i]
                is_stop = self.curvature_radius[i] == 0.0
                if is_stop:
                    if rd.z >= 0.0:
                        ok = False
                        break
                    t = (elemZ - ro.z) / rd.z
                else:
                    radius = self.curvature_radius[i]
                    centerZ = elemZ + radius
                    isect, t, n = self.intersect_with_sphere(centerZ, radius, ro, rd)
                    if not isect:
                        ok = False
                        break

                hit = ro + rd * t
                r = hit.x * hit.x + hit.y * hit.y
                if r > self.aperture_radius[i] * self.aperture_radius[i]:  # out of the element aperture
                    ok = False
                    break

                self.draw_rays[ind, ii] = ro

                ro = ti.Vector([hit.x, hit.y, hit.z])

                if not is_stop:
                    # refracted by lens
                    etaI = self.eta[i]
                    etaT = self.eta[i - 1] if i > 0 and self.eta[i - 1] != 0.0 else 1.0   # the outer of 0-th element is air, whose eta is 1.0
                    has_r, d = refract(rd, n, etaI/etaT)
                    if not has_r:
                        ok = False
                        break
                    rd = ti.Vector([d.x, d.y, d.z])

        self.draw_rays[ind, self._elem_count] = ro
        self.draw_rays[ind, self._elem_count + 1] = ro + rd * 20.0
        # return ok, ti.Vector([ro.x, ro.y, -ro.z]), ti.Vector([rd.x, rd.y, -rd.z]).normalized()

    @ti.func
    def draw_ray_from_scene(self, ori, dir, ind):
        ro, rd = ti.Vector([ti.cast(ori.x,ti.f32), ori.y, -ori.z]), ti.Vector([ti.cast(dir.x,ti.f32), dir.y, -dir.z]).normalized()
        elemZ = -self.front_z()
        ok = True
        t = 0.0
        n = ti.Vector([0.0,0.0,0.0])
        for _ in range(1):  # force the inner loop serialized so that break could be used
            for i in range(self._elem_count):
                is_stop = self.curvature_radius[i] == 0.0
                if is_stop:
                    t = (elemZ - ro.z) / rd.z
                else:
                    radius = self.curvature_radius[i]
                    centerZ = elemZ + radius
                    isect, t, n = self.intersect_with_sphere(centerZ, radius, ro, rd)
                    if not isect:
                        ok = False
                        break

                assert t > 0.0

                hit = ro + rd * t
                r = hit.x * hit.x + hit.y * hit.y
                if r > self.aperture_radius[i] * self.aperture_radius[i]:  # out of the element aperture
                    ok = False
                    break

                self.draw_rays[ind, i] = ro
                ro = ti.Vector([hit.x, hit.y, hit.z])

                if not is_stop:
                    # refracted by lens
                    etaI = 1.0 if i == 0 or self.eta[i - 1] == 0.0 else self.eta[i - 1]
                    etaT = self.eta[i] if self.eta[i] != 0.0 else 1.0
                    rd.normalized()
                    has_r, d = refract(rd, n, etaI/etaT)
                    if not has_r:
                        ok = False
                        break
                    rd = ti.Vector([d.x, d.y, d.z])

                elemZ += self.thickness[i]

        self.draw_rays[ind, self._elem_count] = ro
        self.draw_rays[ind, self._elem_count + 1] = ro + rd * 20.0

    def get_ray_points(self):
        return self.draw_rays.to_numpy()
