from typing import List, Callable
from gui.widget import Widget, PropertyWidget, AttributeValueType
import dearpygui.dearpygui as dpg

class LensSurface(Widget):
    def __init__(self, parent: int, callback:Callable[[None],None]=None):
        super().__init__(parent=parent,callback=callback)



class LensSphereSurface(LensSurface):
    curvature_radius = PropertyWidget('Curvature Radius', AttributeValueType.ATTRI_FLOAT,-1000,1100.0,100)
    thickness = PropertyWidget('Thickness', AttributeValueType.ATTRI_FLOAT,0.0,1000.0,100)
    eta = PropertyWidget('Eta', AttributeValueType.ATTRI_FLOAT,0.0,100.0,100)
    aperture_radius = PropertyWidget('Aperture Radius', AttributeValueType.ATTRI_FLOAT,0.0,1000.0,100)
    def __init__(self, parent: int, callback:Callable[[None],None]):
        super().__init__(parent=parent,callback=callback)
        with dpg.tree_node(label='SphereElement',parent=parent) as self._widget_id:
            self.curvature_radius = 0.0
            self.thickness = 0.0
            self.eta = 0.0
            self.aperture_radius = 0.0

    def dump(self):
        return [ self.curvature_radius,self.thickness,self.eta,self.aperture_radius]

    def property_changed(self, s, a, u):
        self._invoke_update()

    def load(self, data:List[float]= [0.0,0.0,0.0,0.0]):
        self.curvature_radius = data[0]
        self.thickness = data[1]
        self.eta = data[2]
        self.aperture_radius = data[3]

class ApertureSurface(LensSurface):
    thickness = PropertyWidget('Thickness', AttributeValueType.ATTRI_FLOAT,0.0,100.0,100)
    aperture_radius = PropertyWidget('Aperture Radius', AttributeValueType.ATTRI_FLOAT,0.0,100.0,100)
    def __init__(self, parent: int, callback:Callable[[None],None]):
        super().__init__(parent=parent,callback=callback)
        self._widget_id = parent
        self.thickness = 0.0
        self.aperture_radius = 0.0

    def dump(self):
        return [ 0.0,self.thickness, 0.0 ,self.aperture_radius]

    def load(self, data:List[float]= [0.0,0.0,0.0,0.0]):
        self.thickness = data[1]
        self.aperture_radius = data[3]