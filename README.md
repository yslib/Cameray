Cameray
---

**Cameray** is a lens editor and simulator for fun. It's could be used for studying an optics system of DSLR in an interactive way. But the project is in a very early version. The program is still crash-prone and also lack of many realistic camera features now.


![](./feature.jpg)

Usage
---

Just clone the repo and go to the root directory and run

You can install it by pip using:

```shell
python -m pip install cameray
```

and then enter

```shell
cameray
```
to run it.

Or, just clone the repo and go to the root directory and run

```shell
python -m pip install setuptools
python setup.py sdist
python setup.py install
```
and enter

```shell
cameray
```

to run it.

Credits
---

- [Taichi][1] is the core of the simulator. It's a powerful programming language embedded in Python for high-performance numerical computations. Without the help of it, this project can not be implemented in such an efficient way in Python. **(The ray-tracing part of the project is from Taichi's cornell box example now. A general renderer should replace it in the near future.)**

- [DearPyGui][2], an easy-to-use Python GUI framework based on [ImGUI][3]. Besides the native ImGUI widgets, it also wraps many other 3rd party ImGUI plugins (Imnodes). It's convenient to provide a friendly GUI for Cameray.

Dependencies
---

**Cameray** only could be run in Python 3.7/3.7/3.8/3.9 because Taichi only support these versions.

[1]: (https://github.com/taichi-dev/taichi)
[2]: (https://github.com/hoffstadt/DearPyGui)
[3]: (https://github.com/ocornut/imgui)


Roadmap
---

  - [ ] Stablity

  - [ ] A more realistic camera model considering general optical spectrum

  - [ ] General renderer

  - [ ] A more detail 2D camera illustration. (More kinds of rays and lens data)

  - [ ] Undo/Redo for Editor