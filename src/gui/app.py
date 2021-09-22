import sys
import os
from typing import Callable, Dict, Any, List, Tuple
import configparser

sys.path.append('..')
import dearpygui.dearpygui as dpg
from base.msg_queue import get_msg_queue, msg, get_coroutine
from gui.lens_designer import LensDesignerWidget

CMR_CONFIG_FILE_PATH = r''
# CMR_FONT_FILE_PATH = r'C:\Windows\Fonts\msyh.ttc'
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


msgqueue = get_msg_queue()
coroutine = get_coroutine()

class App:
    def __init__(self) -> None:
        self._setup_init()
        self._setup_uuid()
        self._setup_style()
        self._setup_window()
        self._setup_viewport()

    def _on_app_close(self, s,a,u):
        dpg.delete_item(s)
        self._app_config.write(open(CMR_CONFIG_FILE_PATH,'a'))
        print('_on_app_close')

    def _setup_style(self):
        with dpg.font_registry():
            # Change your font here
            # dpg.add_font(CMR_FONT_FILE_PATH, 18, default_font=True)
            pass

    def _setup_init(self):
        self._app_config = configparser.ConfigParser()
        self._app_config.read(CMR_CONFIG_FILE_PATH)

    def _setup_uuid(self):
        self._gui_id_app:int = dpg.generate_uuid()
        self._lense_designer_widget:LensDesignerWidget = None

    def _gui_viewport_resize_event(self, sender, a, u):
        """
        Keep the root widget fill up the viewport
        """
        dpg.set_item_height(self._gui_id_app, a[3])
        dpg.set_item_width(self._gui_id_app, a[2])

    def _setup_viewport(self):
        if not dpg.is_viewport_created():
            icon = PROJECT_DIR+'/icon.png'
            vp = dpg.create_viewport(small_icon=icon,title='Cameray', large_icon=icon,width=1920,height=1080)
            dpg.set_viewport_resize_callback(lambda a, b:self._gui_viewport_resize_event(a, b, self._gui_id_app))
            dpg.setup_dearpygui(viewport=vp)
            dpg.show_viewport(vp)
            dpg.set_viewport_title(title='Cameray')
            # dpg.set_viewport_decorated(False)
            dpg.set_viewport_resizable(False)

    def _setup_window(self):

        with dpg.window(label="Cameray",id=self._gui_id_app,
                        on_close=self._on_app_close,
                        pos=(0, 0),
                        no_title_bar=True,
                        no_move=True,
                        no_resize=True):

            self._lense_designer_widget:LensDesignerWidget = LensDesignerWidget(self._gui_id_app)

    def _window_resize_callback(self, s,a,u):
        pass

    def show(self):
        global coroutine
        while(dpg.is_dearpygui_running()):
            while not msgqueue.empty():
                event = msgqueue.get()
                if callable(event):
                    cr = event()
                    if not cr:
                        continue
                    try:
                        next(cr)
                    except StopIteration:
                        continue
                coroutine.append(cr)

            if coroutine:
                copy_cr = coroutine
                coroutine = []
                news = []
                for cr in copy_cr:
                    try:
                        cr.send(0)
                        news.append(cr)
                    except StopIteration:
                        continue
                coroutine.extend(news)

            dpg.render_dearpygui_frame()
        dpg.cleanup_dearpygui()


def main():
    App().show()


if __name__ == '__main__':
    main()