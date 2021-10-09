import math
import dearpygui.dearpygui as dpg

from . import lens_preset
from typing import List, Any, Callable, Dict
from base.msg_queue import msg
from gui.widget import Widget, PropertyWidget, AttributeValueType
import numpy as np
import networkx as nx
from gui.lens_editor.node import SceneNode, FilmNode, LensSurfaceGroupNode

from core.renderer import real_cam, color_buffer
class LensCanvasWidget(Widget):
    def __init__(self, *, parent: int, film_height=24.0, callback:Callable[[None],None]=None):
        super().__init__(parent=parent, callback=callback)
        height = 400
        with dpg.child(parent=parent,autosize_x=True,no_scrollbar=True,height=400) as self._widget_id:
            with dpg.drawlist(label='lenses',parent=self._widget_id) as self._drawlist_id:
                self.film_height = film_height
                self.axis_y = int(height / 2.0)
                self.origin_z = 0
                self.scale = 5.0
                self.lense_length = 0.0
                self.max_radius = 0.0
                self.world_matrix = np.array([[1.0,0.0,0.0], [0.0,1.0 ,0.0],[0.0,0.0,1.0]])
                self.screen_matrix = np.array([[1.0,0.0,0.0], [0.0,1.0 ,0.0],[0.0,0.0,1.0]])

        # self._drawlist_id = self.widget()

    def drawlist(self):
        return self._drawlist_id

    def _setup_transform(self,scale: float, origin_z:float, axis_y:float):
        """
        Updates transform matrix whenever the lense size or canvas size changes
        """
        self.world_matrix = np.array([
            [scale,0.0,0.0],
            [0.0,scale,0.0],
            [0.0,0.0,1.0]
            ])

        self.screen_matrix = np.array([
            [-1, -0, origin_z],
            [0, -1, axis_y],
            [0,0,1]
        ])

        self.world_to_screen = self.screen_matrix @ self.world_matrix

    def _world_to_screen(self, point: List[float]):
        return (self.world_to_screen @ np.array([point[0], point[1], 1.0]))[0:2]

    def _draw_frame(self, padding = 5):
        # rect = dpg.get_item_rect_size(self.drawlist())
        rect = (dpg.get_item_width(self.drawlist()), dpg.get_item_height(self.drawlist()))
        poly = [
            [padding,padding], # topleft
            [rect[0] - padding, padding], # topright
            [rect[0]- padding, rect[1] - padding], # bottomright
            [padding,rect[1] - padding] # bottomleft
        ]
        dpg.draw_polyline(poly, parent=self.drawlist(),closed=True)

    def _update_canvas(self, lense_length:float, lense_radius:float):
        """
        Updates whenever the lense size or canvas size changes
        """
        w, h = dpg.get_item_width(self.drawlist()), dpg.get_item_height(self.drawlist())
        rect = (w, h)
        self.scale = min(rect[0] / lense_radius, rect[1] / lense_length)
        self.axis_y = rect[1] / 2.0
        self.origin_z = rect[0]/2.0 + lense_length * self.scale / 2.0
        self._setup_transform(self.scale, self.origin_z, self.axis_y)

    def _draw_impl(self, lenses, lense_length, lense_radius):
        self._update_canvas(lense_length, lense_radius)
        self.clear_drawinglist()
        self._draw_frame()
        z = lense_length
        # draw lense groups
        for i in range(len(lenses)):
            is_stop = lenses[i][0] == 0.0
            r = lenses[i][3]/2.0
            if is_stop:
                self._draw_aperture_stop(z, r, color=[255.0, 255.0, 0.0], thickness=4.0)
            else:
                a, b = self._draw_arch(z, lenses[i][0], min(min(6.0, lenses[i][3]), abs(lenses[i][0])))
                if i > 0 and lenses[i - 1][2] != 1 and lenses[i-1][2] != 0:
                    self._draw_line(a, first)
                    self._draw_line(b, last)  # draw connection between element surface
                first, last = a, b

            z -= lenses[i][1]

        self._draw_axis()
        self._draw_film()

    def draw_lenses(self, lenses:np.ndarray):

        # workaround
        rect = dpg.get_item_rect_size(self.widget())
        dpg.configure_item(self.drawlist(), width=rect[0] - 20,height=rect[1] - 20)
        ####

        length = 0.0
        max_radius = 0.0
        for i in range(len(lenses)):
            thickness = lenses[i][1]
            if math.isnan(thickness):
                continue
            else:
                length += lenses[i][1]
            max_radius = max(max_radius, lenses[i][3] / 2.0)

        if length == 0.0 or max_radius == 0.0:
            return

        self.lense_length = length
        self.max_radius = max_radius


        self._draw_impl(lenses, length, max_radius)

    def draw_rays(self, rays:np.array, count:int, color=[255.0,255.0,255.0,255.0], thickness=1.0, symetric=False):
        points = []
        point1 = []
        for i in range(count):
            p = rays[0][i]
            p0 = self._world_to_screen([-p[2],p[0]])
            p1 = self._world_to_screen([-p[2],-p[0]])
            points.append(p0)
            point1.append(p1)

        dpg.draw_polyline(points, parent=self.drawlist(), color=color, thickness=thickness)
        dpg.draw_polyline(point1, parent=self.drawlist(), color=color, thickness=thickness)

    def _draw_line(self, p0:List[float], p1:List[float], color=[255.0,255.0,255.0], thickness=1.0):
        dpg.draw_line(self._world_to_screen(p0), self._world_to_screen(p1), color=color, thickness=thickness,parent=self.drawlist())

    def _draw_axis(self):
        p0 = self._world_to_screen([self.lense_length + 10,0])
        p1 = self._world_to_screen([-10,0])
        dpg.draw_line(p0, p1, parent=self.drawlist())

    def _draw_film(self, color=[255.0,255.0,255.0], thickness = 4.0):
        p0 = self._world_to_screen([0, self.film_height / 2])
        p1 = self._world_to_screen([0, -self.film_height/2])
        dpg.draw_line(p0, p1, color=color, thickness=thickness, parent=self.drawlist())

    def _draw_aperture_stop(self, z:float, aperture_radius:float, color=[255.0,255.0,255.0], thickness=2.0):
        p0 = self._world_to_screen([z, aperture_radius + 10])
        p1 = self._world_to_screen([z, aperture_radius])
        p2 = self._world_to_screen([z, -aperture_radius])
        p3 = self._world_to_screen([z, -aperture_radius - 10])
        dpg.draw_line(p0, p1, color=color, thickness=thickness, parent=self.drawlist())
        dpg.draw_line(p2, p3, color=color, thickness=thickness, parent=self.drawlist())

    def _draw_arch(self, z: float, curvature_radius:float, aperture_radius:float, color=[255.0,255.0,255.0], thickness=1.0):
        """
        There is no built-in arch drawing API in DearPyGui. So arch is done with
        segmented by polylines.

        Returns the two end points of the arch
        """
        center = z - curvature_radius
        if abs(curvature_radius) < 1e-5:
            return [z, 0.0], [z, 0.0]
        half = math.asin(aperture_radius/curvature_radius)
        min_theta = -2 * half
        max_theta = 2 * half
        seg_count = 30
        points = []
        first = []
        last = []
        for i in range(seg_count):
            t = i * 1.0 / seg_count
            theta = min_theta * (1.0 - t) + t * max_theta
            p0 = (center + curvature_radius * math.cos(theta))
            p1 = curvature_radius * math.sin(theta)
            if i == 0:
                first = [p0, p1]
            elif i == seg_count - 1:
                last = [p0, p1]
            points.append(self._world_to_screen([p0, p1]))
        dpg.draw_polyline(points=points,parent=self.drawlist(),color=color,thickness=thickness)
        return first, last

    def clear_drawinglist(self):
        dpg.delete_item(self.drawlist(), children_only=True)



class EditorEventType:
    EVENT_NODE_ADD= 0x000
    EVENT_NODE_DELETE = 0x001
    EVENT_NODE_UPDATE = 0x002
    EVENT_NODE_DELETE_ALL = 0x003

    EVENT_LINK_ADD = 0x200
    EVENT_LINK_DELETE = 0x201


    EVENT_SURFACE_ADD = 0x300
    EVENT_SURFACE_DELETE = 0x301
    EVENT_SURFACE_ATTRIBUTE_CHANGED = 0x302

    EVENT_UPDATE_ALL = 0xFFF


class ToolBar(Widget):
    def __init__(self, *, parent: int, callback: Callable[[Any], None]):
        super().__init__(parent=parent, callback=callback)
        self.add_node_button = dpg.add_button(label='Add Node')
        dpg.add_same_line()
        self.remove_node_button = dpg.add_button(label='Remove Nodes')
        dpg.add_same_line()
        self.remove_link_button = dpg.add_button(label='Remove Links')
        dpg.add_same_line()
        self.clear_all_button = dpg.add_button(label='Clear')
        dpg.add_same_line()
        self.save_button = dpg.add_button(label='Save')
        dpg.add_same_line()
        self.open_button = dpg.add_button(label='Open')
        dpg.add_same_line()
        self.preset_combo = dpg.add_combo(list(lens_preset.lens_data.keys()),label='Lens Preset',width=150)
        dpg.add_same_line()
        self.auto_arrange = dpg.add_button(label='Auto Arrange')


class LensEditorWidget(Widget):
    def __init__(self,*,update_callback:Callable[[Any],None], parent: int):
        super().__init__(parent=parent, callback=update_callback)
        self._lense_data:List[Dict[str, List[float]]]= []
        self._valid_lenses = False
        self._editor_id = -1

        with dpg.group(horizontal=False,parent=parent) as self._widget_id:
            self._toolbar = ToolBar(parent=self._widget_id,callback=self.callback())
            with dpg.node_editor(parent=self._widget_id,callback=self._link_add_callback, delink_callback=self._link_delete_callback, height = 800) as self._editor_id:
                pass
            dpg.configure_item(self._toolbar.add_node_button,callback = lambda s,a,u:self.add_lens_group(LensSurfaceGroupNode(name='LensGroup', parent=self._editor_id, update_callback=self.callback())))
            dpg.configure_item(self._toolbar.clear_all_button,callback = lambda s,a,u:self.clear_lense_group())
            dpg.configure_item(self._toolbar.preset_combo,callback = lambda s,a,u:self.set_lense_data(lens_preset.lens_data.get(a,[])))
            dpg.configure_item(self._toolbar.remove_node_button,callback = lambda s,a,u:self.remove_selected_nodes())
            dpg.configure_item(self._toolbar.remove_link_button,callback = lambda s,a,u:self.remove_selected_links())
            dpg.configure_item(self._toolbar.auto_arrange,callback = lambda s,a,u:self.auto_arrange())

        # self._editor_id = self.widget()
        self.widget_g = nx.Graph()
        self.attri_g = nx.Graph()

        self._add_default_node()

    def _link_add_callback(self, sender:int, app_data:Any, user_data:Any):
        """
        input_end
        output_end
        """
        self._add_link_impl(app_data[0], app_data[1])
        self._invoke_update(event=EditorEventType.EVENT_LINK_ADD)

    def _link_delete_callback(self, sender:int, app_data:Any, user_data:Any):
        """
        app_data: edge item
        """
        self._remove_link_impl(app_data)
        self._invoke_update(event=EditorEventType.EVENT_LINK_DELETE)

    def _add_link_impl(self, output_end, input_end):
        output_node, input_node = dpg.get_item_user_data(output_end), dpg.get_item_user_data(input_end)
        deleted_attri_edge = []
        deleted_node_edge = []
        for nbr, datadict in self.attri_g.adj[input_end].items():
            deleted_attri_edge.append((nbr, input_end))
            edge_item = datadict.get('edge_item', None)
            if edge_item:
                dpg.delete_item(edge_item)
                adj_output_node = dpg.get_item_user_data(nbr)
                deleted_node_edge.append((adj_output_node, input_node))

        self.attri_g.remove_edges_from(deleted_attri_edge)
        self.widget_g.remove_edges_from(deleted_node_edge)

        link = dpg.add_node_link(output_end, input_end, parent=self._editor_id, user_data={'input_node':input_node,'output_node':output_node,'input_end':input_end,'output_end':output_end})
        self.widget_g.add_edge(output_node, input_node, edge_item=link)
        self.attri_g.add_edge(output_end, input_end, edge_item=link)

    def _remove_link_impl(self, edge_item):
        udata = dpg.get_item_user_data(edge_item)
        input_node, output_node = udata['input_node'], udata['output_node']
        self.widget_g.remove_edge(output_node,input_node)
        self.attri_g.remove_edge(udata['output_end'],udata['input_end'])
        dpg.delete_item(edge_item)

    def _add_node_impl(self, lens_group_node):
        self.widget_g.add_node(lens_group_node.widget())
        if isinstance(lens_group_node, SceneNode):
            self.attri_g.add_node(lens_group_node.output_end())
        elif isinstance(lens_group_node, FilmNode):
            self.attri_g.add_node(lens_group_node.input_end())
        else:
            self.attri_g.add_node(lens_group_node.input_end())
            self.attri_g.add_node(lens_group_node.output_end())

    def _remove_selected_nodes_impl(self):
        """
        TODO::

        dpg.get_selected_nodes may hava a potential BUG

        """
        print("_remove_selected_nodes_impl::dpg.get_selected_nodes may hava a potential BUG")
        selected_nodes = dpg.get_selected_nodes(self._editor_id)
        for x in selected_nodes:
            self._remove_node_impl(dpg.get_item_user_data(x))

    def _remove_selected_links_impl(self):
        selected_links = dpg.get_selected_links(self._editor_id)
        for link in selected_links:
            self._remove_link_impl(link)

    def _remove_node_impl(self, lens_group_node):
        item = lens_group_node.widget()
        self.widget_g.remove_node(item)
        attri_ends = []
        if isinstance(lens_group_node, SceneNode):
            attri_ends.append(lens_group_node.output_end())
        elif isinstance(lens_group_node, FilmNode):
            attri_ends.append(lens_group_node.input_end())
        else:
            attri_ends.append(lens_group_node.input_end())
            attri_ends.append(lens_group_node.output_end())

        deleted_edge = []
        for end in attri_ends:
            for nbr, datadict in self.attri_g.adj[end].items():
                edge_item = datadict['edge_item']
                deleted_edge.append((nbr, end))
                dpg.delete_item(edge_item)

        self.attri_g.remove_edges_from(deleted_edge)
        lens_group_node.delete()

    def _remove_all_node_impl(self):
        list(map(lambda x: self._remove_node_impl(dpg.get_item_user_data(x)), list(self.widget_g.nodes)))

    def _add_default_node(self):
        self._scene_node = SceneNode(self._editor_id, self.callback())
        self._film_node = FilmNode(self._editor_id, self.callback())
        self._add_node_impl(self._scene_node)
        self._add_node_impl(self._film_node)

    def add_lens_group(self, node:LensSurfaceGroupNode):
        self._add_node_impl(node)
        self._invoke_update(event=EditorEventType.EVENT_NODE_ADD)

    def remove_lense_group(self, nw:Widget):
        self._remove_node_impl(nw)
        self._invoke_update(event=EditorEventType.EVENT_NODE_DELETE)

    def remove_selected_nodes(self):
        self._remove_selected_nodes_impl()
        self._invoke_update(event=EditorEventType.EVENT_NODE_DELETE)

    def remove_selected_links(self):
        self._remove_selected_links_impl()
        self._invoke_update(event=EditorEventType.EVENT_LINK_DELETE)

    def clear_lense_group(self):
        self._remove_all_node_impl()
        self._add_default_node()
        self._invoke_update(event=EditorEventType.EVENT_NODE_DELETE)

    def auto_arrange(self):
        rect = dpg.get_item_rect_size(self._editor_id)
        all_nodes = self.get_lense_group()
        s = len(all_nodes)
        if s < 2:
            return

        s = int(s / 2.0)
        w_interval = rect[0] / (s + 2)
        h_interval = rect[1] / 2.0
        for i in range(len(all_nodes)):
            y = (i % 2) * h_interval + 100
            x = (i / 2) * w_interval + 100
            dpg.configure_item(all_nodes[i].widget(),pos=(x,y))


    def get_lense_group(self):
        res = list(nx.all_simple_paths(self.widget_g, self._scene_node.widget(), self._film_node.widget()))
        ret = []
        if len(res) > 0:
            ret = list(map(lambda x: dpg.get_item_user_data(x), res[0]))
        return ret

    def get_lense_data(self):
        groups = self.get_lense_group()
        lense_data = []
        for g in range(1, len(groups) - 1):  # excludes scene and film node
            c = groups[g].get_surface_count()
            for ind in range(c):
                surf = groups[g].get_surface(ind)
                lense_data.append(surf.dump())
        return lense_data

    def set_lense_data(self, lense_data:List[List[float]]):
        self._remove_all_node_impl()
        self._add_default_node()

        # parse lense data
        stack = []
        output_end = self._scene_node.output_end()
        for surf in lense_data:
            if surf[2] != 1.0 and surf[0] != 0:  # surface begin
                stack.append(surf)
            elif surf[2] == 1.0 and surf[0] != 0: # surface end
                stack.append(surf)
                lense_group = LensSurfaceGroupNode.create_sphere_lense_group(name='LensGroup',parent=self._editor_id,group_data=stack, callback=self.callback())
                self._add_node_impl(lense_group)
                input_end = lense_group.input_end()
                self._add_link_impl(output_end=output_end,input_end=input_end)
                output_end = lense_group.output_end()
                stack = []
            elif surf[0] == 0.0 and surf[2] == 0.0:  # aperture surface
                aperture = ApertureStop(parent=self._editor_id,value_update_callback=self.callback())
                aperture.block_callback(True)
                aperture.get_surface(0).load(surf)
                self._add_node_impl(aperture)
                input_end = aperture.input_end()
                self._add_link_impl(output_end=output_end,input_end=input_end)
                output_end = aperture.output_end()
                aperture.block_callback(False)

        if len(stack) > 0:
            print('Wrong lense data')

        self._add_link_impl(output_end=output_end, input_end=self._film_node.input_end())
        # closed
        # parse data end

        self.auto_arrange()
        self._invoke_update(event=EditorEventType.EVENT_NODE_UPDATE)

    def get_film_size(self):
        return self._film_node.get_film_size()

    def get_focus_depth(self):
        return self._scene_node.get_focus_depth()

    def get_focus_state(self):
        return self._scene_node.get_focus_state()

    def get_keep_rendering(self):
        return self._film_node.get_keep_rendering()




class LensDesignerWidget(Widget):

    def __init__(self, parent: int):
        super().__init__(parent=parent)
        with dpg.group(parent=parent,horizontal=False) as self._widget_id:
            self._lense_canvas: LensCanvasWidget = LensCanvasWidget(parent=self.widget(),callback=self._canvas_update)
            self._node_editor: LensEditorWidget = LensEditorWidget(update_callback=self._editor_update, parent=self.widget())

            pos = [0.0, 3.0, 24.0]
            center = [0.0, 0.0, 0.0]
            world_up = [0.0, 1.0, 0.0]
            # self.camera = RealisticCamera(pos, center, world_up)
            self._paint_canvas()


    def _editor_update(self, *args, **kwargs):
        """
        Update Realistic camera here
        """
        lense_data = self._node_editor.get_lense_data()
        self._canvas_update(lense_data = lense_data)

    def _canvas_update(self, *args, **kwargs):
        self._update_camera(kwargs['lense_data'])

    def _paint_canvas(self):
        ray_points = real_cam.get_ray_points_buffer()
        new_lense_data = real_cam.get_lenses_data()
        self._lense_canvas.draw_lenses(np.array(new_lense_data))
        self._lense_canvas.draw_rays(ray_points, real_cam.get_element_count() + 2, color=[0, 0, 255])


    @msg
    def _update_camera(self, lense_data):
        real_cam.load_lens_data(lense_data)
        self._node_editor.get_focus_state() and real_cam.refocus(self._node_editor.get_focus_depth())
        if self._node_editor.get_keep_rendering():
            color_buffer.from_numpy(np.zeros((800, 600, 3)))
        real_cam.gen_draw_rays_from_film()
        self._paint_canvas()