import dearpygui.dearpygui as dpg
from dearpygui.logger import mvLogger
from typing import Callable, Any

_logger = None

def init_global_logger(parent:int):
    global _logger
    if _logger is None:
        _logger = mvLogger(parent)

def get_logger()->mvLogger:
    return _logger


def _config(sender, keyword, user_data):
    widget_type = dpg.get_item_type(sender)
    items = user_data

    if widget_type == "mvAppItemType::mvRadioButton":
        value = True

    else:
        keyword = dpg.get_item_label(sender)
        value = dpg.get_value(sender)

    if isinstance(user_data, list):
        for item in items:
            dpg.configure_item(item, **{keyword: value})
    else:
        dpg.configure_item(items, **{keyword: value})

class Widget:
    def __init__(self,*, parent:int,callback:Callable[[Any],None]=None):
        self._widget_id:int = None
        self._parent_id:int = parent
        self._block = False
        self._callback = callback

    def widget(self)->int:
        return self._widget_id

    def parent(self)->int:
        return self._parent_id

    def __hash__(self):
        return self._widget_id

    def property_changed(self, s, a, u):
        self._invoke_update(sender=s, app_data=a, user_data=u)

    def block_callback(self, block):
        self._block = block

    def _invoke_update(self, *args, **kwargs):
        not self._block and callable(self._callback) and self._callback(*args, **kwargs)

    def callback(self):
        return self._callback

    def delete(self):
        self._widget_id and dpg.delete_item(self._widget_id)
        self._widget_id = None

    def __del__(self):
        self.delete()

class AttributeValueType:
    ATTRI_FLOAT = 0
    ATTRI_FLOATX = 1
    ATTRI_INT = 2
    ATTRI_BOOL = 3

def PropertyWidget(name:str, property_type:int, min_value:Any=None, max_value:Any=None, width=20,height=10,size=4):
    storage_name = '_' + name
    @property
    def prop(self:Widget):
        return dpg.get_value(getattr(self, storage_name))

    @prop.setter
    def prop(self:Widget, value):
        if not hasattr(self, storage_name):
            if property_type == AttributeValueType.ATTRI_FLOAT:
                item_id = dpg.add_input_float(
                    label=name,
                    default_value=value,
                    min_value=min_value,
                    max_value=max_value,
                    width=width,
                    parent=self.widget(),
                    callback=self.property_changed)
            elif property_type == AttributeValueType.ATTRI_FLOATX:
                item_id = dpg.add_input_floatx(
                    label=name,
                    default_value=value,
                    min_value=min_value,
                    max_value=max_value,
                    width=width,
                    parent=self.widget(),
                    size=size,
                    callback=self.property_changed)
            elif property_type == AttributeValueType.ATTRI_INT:
                item_id = dpg.add_input_int(
                    label=name,
                    default_value=value,
                    min_value=min_value,
                    max_value=max_value,
                    width=width,
                    parent=self.widget(),
                    callback=self.property_changed)
            elif property_type == AttributeValueType.ATTRI_BOOL:
                item_id = dpg.add_checkbox(label=name,parent=self.widget(),default_value=value,callback=self.property_changed)
            setattr(self, storage_name, item_id)
        else:
            item_id = getattr(self, storage_name)
            dpg.set_value(item_id, value)
    return prop
