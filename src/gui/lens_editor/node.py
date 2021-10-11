from typing import List, Any, Callable, Dict
from gui.widget import Widget, PropertyWidget, AttributeValueType
from gui.lens_editor.surface import LensSurface, LensSphereSurface, ApertureSurface
from gui.lens_editor.image_viewer import ImageViewer
from base.msg_queue import msg
import dearpygui.dearpygui as dpg
import numpy as np
from core.renderer import real_cam, color_buffer, taichi_render
from base.platform import get_home_dir


"""
Base node widget definition
"""
class WidgetNode(Widget):
    def __init__(self,*, name:str, parent:int, callback:Callable[[Any],Any]=None):
        super(WidgetNode, self).__init__(parent=parent, callback=callback)
        self._attri_dict:Dict[str,Any] = {}
        self._callback = callback
        self._input_attr = None
        self._output_attr = None
        with dpg.node(label=name,parent=parent, user_data=self) as self._widget_id:
            pass

    def add_attribute(self, attri_name:str, attri_type:int):
        if attri_name not in self._attri_dict.keys():
            with dpg.node_attribute(label=attri_name, attribute_type=attri_type, parent=self.widget(),user_data=self.widget()) as attri:
                self._attri_dict[attri_name] = (attri, {})
                return attri
        return None

    def get_attribute(self, attri_name:str):
        return self._attri_dict.get(attri_name, (None, {}))[0]

    def remove_attribute(self, attri_name):
        if attri_name in self._attri_dict.keys():
            dpg.delete_item(self._attri_dict[attri_name][0])

    def add_value(self,*,attri_name:str,
                        value_name:str,
                        value_type:int,
                        default_value:Any,
                        size:int=4,
                        callback:Callable[[Any], Any]=None):

        attri_id = self.get_attribute(attri_name)
        attri_id, value_dict = self._attri_dict.get(attri_name, (None, {}))
        if attri_id is None:
            print('No corresponding attribute :', attri_name)
            return

        width = 100

        if value_name in value_dict.keys():
            print(value_name, 'has already existed in attribute ', attri_name)
            return None
        else:
            if value_type == AttributeValueType.ATTRI_FLOAT:
                value_id = dpg.add_input_float(label=value_name, callback=callback,default_value=default_value,parent=attri_id,width=width)
            elif value_type == AttributeValueType.ATTRI_FLOATX:
                value_id = dpg.add_input_floatx(label=value_name, callback=callback,default_value=default_value,size=size, parent=attri_id,width=width)
            elif value_type == AttributeValueType.ATTRI_INT:
                value_id = dpg.add_input_int(label=value_name,callback=callback,default_value=default_value, parent=attri_id,width=width)
            value_dict[value_name] = value_id

        return value_id


    def get_attri_value_item_id(self,attri_name:str, value_name:str):
        return self._attri_dict.get(attri_name, (-1, {}))[1].get(value_name, None)

    def get_value(self, attri_name:str, value_name:str):
        item_id = self.get_attri_value_item_id(attri_name=attri_name,value_name=value_name)
        if item_id is not None:
            return dpg.get_value(item_id)

        print('No such value: ', value_name, ' of ', attri_name)
        return None

    def set_value(self, attri_name:str, value_name:str, value:Any):
        item_id = self.get_attri_value_item_id(attri_name=attri_name,value_name=value_name)
        if item_id is not None:
            dpg.configure_item(item=item_id, value=Any)
            return
        print('No such value: ', value_name, ' of ', attri_name)


class MidNode(WidgetNode):
    def __init__(self, *, name: str, parent: int, callback: Callable[[Any], Any]):
        super().__init__(name=name, parent=parent, callback=callback)
        self._input_end = self.add_attribute('input', dpg.mvNode_Attr_Input)
        self._output_end = self.add_attribute('output', dpg.mvNode_Attr_Output)

    def input_end(self):
        return self._input_end

    def output_end(self):
        return self._output_end

class InputNode(WidgetNode):
    def __init__(self, *, name: str, parent: int, callback: Callable[[Any], Any]):
        super().__init__(name=name, parent=parent, callback=callback)
        self._input_end = self.add_attribute('input', dpg.mvNode_Attr_Input)

    def input_end(self):
        return self._input_end

class OutputNode(WidgetNode):
    def __init__(self, *, name: str, parent: int, callback: Callable[[Any], Any]):
        super().__init__(name=name, parent=parent, callback=callback)
        self._output_end = self.add_attribute('output', dpg.mvNode_Attr_Output)

    def output_end(self):
        return self._output_end



class ApertureStop(MidNode):
    def __init__(self, *, parent: int,value_update_callback:Callable[[Any], None]=None):
        super().__init__(name='Aperture Stop', parent=parent, callback=value_update_callback)
        self.static_attri = self.add_attribute('Aperture', attri_type=dpg.mvNode_Attr_Static)
        self.aperture_surface = ApertureSurface(self.static_attri, self.callback())

    def get_surface(self, ind:int):
        return self.aperture_surface

    def get_surface_count(self):
        return 1


class SceneNodeParam(Widget):
    focus_depth = PropertyWidget(name='FocusDepth', property_type=AttributeValueType.ATTRI_FLOAT,min_value=0.0, max_value=10000.0,width=100)
    enable_focus = PropertyWidget(name='EnableFocus', property_type=AttributeValueType.ATTRI_BOOL, width=100)
    ray_angle = PropertyWidget(name='ParallelRayAngle', property_type=AttributeValueType.ATTRI_FLOAT, min_value=0.0, max_value=60.0, width=100)
    ray_count = PropertyWidget(name='ParallelRayCount', property_type=AttributeValueType.ATTRI_INT, min_value=2, max_value=100, width=100)
    def __init__(self, *, parent: int, callback: Callable[[Any], None]):
        super().__init__(parent=parent, callback=callback)
        self._widget_id = parent
        self.focus_depth = 1.0
        self.enable_focus = True
        self.ray_angle = 0.0
        self.ray_count = 5


class SceneNode(OutputNode):
    def __init__(self,parent:int, value_update_callback:Callable[[Any], None]=None):
        super().__init__(name='Scene',parent=parent,callback=value_update_callback)
        attri = self.add_attribute(attri_name='focus depth',attri_type=dpg.mvNode_Attr_Static)
        self._param = SceneNodeParam(parent=attri,callback=self.callback())

    def get_focus_depth(self):
        return self._param.focus_depth

    def get_focus_state(self):
        return self._param.enable_focus

class FilmNodeParam(Widget):
    film_size = PropertyWidget(name='FilmSize', property_type=AttributeValueType.ATTRI_FLOATX,min_value=0.0,max_value=1000.0,size=2,width=100)
    render_window = PropertyWidget(name='RenderWindow', property_type=AttributeValueType.ATTRI_BOOL, width=100)
    def __init__(self, *, parent: int, callback: Callable[[Any], None]):
        super().__init__(parent=parent, callback=callback)
        self._widget_id = parent
        self.film_size = (36.00,24.00)
        self.render_window = True
        self.image_viewer = ImageViewer(parent=self.widget())
        self._enable_render = True

        if self._enable_render:
            self._render()


    @msg
    def _render(self):
        i = 0
        color_buffer.from_numpy(np.zeros((800, 600, 3)))
        last_t = 0.0
        while True:
            taichi_render()
            interval = 50
            if i % interval == 0 and i > 0:
                img = color_buffer.to_numpy() * (1 / (i + 1))
                img = np.sqrt(img / img.mean() * 0.24)
                self.image_viewer.from_numpy(img)

            i+=1
            if self._enable_render:
                yield
            else:
                break
        return

    def property_changed(self, s, a, u):
        '''
        Filters the RenderWindow property changes
        '''
        if s == getattr(self, '_RenderWindow'):
            dpg.configure_item(self.image_viewer.widget(), show=a)
            self._enable_render = a
            if self._enable_render:
                self._render()
        return super().property_changed(s, a, u)


class FilmNode(InputNode):
    def __init__(self, parent:int,value_update_callback:Callable[[Any], None]=None):
        super().__init__(name='Film',parent=parent,callback=value_update_callback)
        attri = self.add_attribute(attri_name='film size',attri_type=dpg.mvNode_Attr_Static)
        self._param = FilmNodeParam(parent=attri,callback=self.callback())

        self._save_dir_settings = dpg.add_button(parent=attri,label='Select directory...', callback=self._select_directory)
        dpg.add_same_line()
        self._dir_text = dpg.add_text(get_home_dir(),parent=attri)
        self._save_button = dpg.add_button(parent=attri, label='Save', callback=self._save_callback)

    def _save_callback(self, s, a, u):
        path = dpg.get_value(self._dir_text)
        self._param.image_viewer.save_image(dpg.get_value(self._dir_text))

    def _select_directory(self, s,a,u):
        with dpg.file_dialog(label='Select directory', directory_selector=True, callback=lambda s, a, u: self._update_dir_text(a.get('file_path_name', get_home_dir()))):
            pass

    def _update_dir_text(self, dir:str):
        dir = dir.lstrip("\\")   # I don't known why there is a slash prefix on Windows
        dpg.set_value(self._dir_text, dir)

    def get_film_size(self):
        return self._param.film_size

    def get_keep_rendering(self):
        return self._param.render_window

class LensSurfaceGroupNode(MidNode):
    def __init__(self,*,name:str,parent:int, update_callback:Callable[[Any], Any]=None):
        super().__init__(name=name, parent=parent,callback=update_callback)
        self._lense_surface_group:List[LensSurface] = []
        self._surface_data_value_id:List[int] = []
        self.input_attri_item_id = self.add_attribute('surface count', attri_type=dpg.mvNode_Attr_Static)

        self.count_attri = self.add_value(attri_name='surface count',
        value_name="Surface Count",
        value_type=AttributeValueType.ATTRI_INT,
        default_value=0,
        callback=lambda s,a,u:self._update_surface(int(a)))


    def _update_surface(self, count):
        cur_count = len(self._lense_surface_group)
        print('surface count changed: ',count, cur_count)

        delta = count - cur_count
        if delta > 0:
            for _ in range(delta):
                surf = LensSphereSurface(self.input_attri_item_id, callback=self.callback())
                self._lense_surface_group.append(surf)
        elif delta < 0:
            for item in self._lense_surface_group[count:]:
                item.delete()
            del self._lense_surface_group[count:]

        self._invoke_update()

    def load(self, raw_group_data: List[List[float]]):
        self._clear_surface_data()
        surfs = []
        for s in raw_group_data:
            surf = LensSphereSurface(self.input_attri_item_id, callback=self.callback())
            surf.load(s)
            surfs.append(surf)

        self._lense_surface_group = surfs
        dpg.set_value(self.count_attri, len(self._lense_surface_group))
        self._invoke_update()

    def get_surface(self, ind:int):
        return self._lense_surface_group[ind]

    def get_surface_count(self):
        return len(self._lense_surface_group)

    def _clear_surface_data(self):
        for surf in self._lense_surface_group:
            surf.delete()
        self._lense_surface_group = []

    def clear_surface_data(self):
        """
        Invode update signal
        """
        self._clear_surface_data()
        self._invoke_update()

    @staticmethod
    def create_sphere_lense_group(*, name, parent, group_data:List[List[float]], callback=None):
        group = LensSurfaceGroupNode(name=name, parent=parent,update_callback=callback)
        group.block_callback(True)
        group.load(group_data)
        group.block_callback(False)
        return group