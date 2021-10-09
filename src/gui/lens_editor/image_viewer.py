from typing import List
from gui.widget import Widget
import dearpygui.dearpygui as dpg
import numpy as np

class ImageViewer(Widget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        with dpg.child(parent=parent, label='Image',height=600,width=800) as self._widget_id:
            self._texture_container = dpg.add_texture_registry(label='Texture')
            self._width:int = 0
            self._height:int = 0
            self._texture_id:int = None

    def _release_texture(self):
        if self._texture_id is not None:
            dpg.delete_item(self._texture_id)
        self._texture_id = None

    def _set_or_recreate_texture(self, width, height, norm_rbga:List[float]):
        if width <= 0 or height <= 0:
            return
        if not self.valid() or self._width != width or self._height != height:
            self._release_texture()
            rect = dpg.get_item_rect_size(self.parent())
            self._texture_id = dpg.add_dynamic_texture(width, height, norm_rbga, parent=self._texture_container)
            dpg.add_image(self._texture_id,parent=self.widget(), width=rect[0], height=rect[1])
            self._width = width
            self._height = height
            return
        dpg.set_value(self._texture_id, norm_rbga)

    def set_image_norm_rgba(self,width:int, height:int, rgba:List[float]):
        """
        each channel of rgba is range from [0 1]
        """
        self._set_or_recreate_texture(width,height,rgba)

    def from_numpy(self, data:np.ndarray):
        shape = data.shape
        norm_rgba = []
        if shape[2] > 4:
            return
        if shape[2] == 3:
            rgba = np.concatenate((data, np.ones((shape[0],shape[1], 1),dtype=np.float32)), axis=2)
            self.set_image_norm_rgba(shape[1],shape[0],rgba.flatten())
        elif shape[2] == 4:
            self.set_image_norm_rgba(shape[1],shape[0],data.flatten())
        elif shape[2] == 2 or shape[2] == 1:
            pass

    def to_numpy(self)->np.ndarray:
        raise NotImplementedError

    def width(self)->int:
        return self._width

    def height(self)->int:
        return self._height

    def valid(self)->bool:
        return self._width > 0 and self._height > 0 and self._texture_id != None