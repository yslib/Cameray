import rawpy
import os
import io
import subprocess
from typing import Dict, Any, List
import numpy as np

class Image:
    def __init__(self, fullname:str='', data=None):
        self._fullname:str = fullname
        self._data:np.ndarray = data
        self._meta:Dict[str,Any] = {}
        self._bracket:ImageBracket = None
        self._read:bool = False if data is None else True
        self._read_meta_data(self._fullname)

        # read meta data
    def _read_meta_data(self, filename):
        """
        Bad perfermance if reads a lot of images
        """
        if filename == '':
            return
        proc = subprocess.Popen("dcraw -i -v {}".format(filename),
                                shell=True, stdout=subprocess.PIPE)
        out, err = proc.communicate()
        out = io.StringIO(out.decode())
        lines = out.readlines()
        meta = {}
        for p in lines:
            if p.startswith('Shutter: '):
                val = p[9:-5]
                meta['shutter'] = eval(val)
                meta['EV'] = 0.0
            if p.startswith('ISO speed: '):
                val = p[11:-1]
                meta['ISO'] = eval(val)

        self._meta = meta
        print(meta)

    def _lazy_read(self):
        if not self._read:
            with rawpy.imread(self._fullname) as raw:
                rgb = raw.postprocess(
                    gamma=(1, 1), no_auto_bright=True, output_bps=16, use_auto_wb=True)
            self._data = rgb
            self._read = True

    def unload_data(self):
        self._read = False
        self._data = None
        pass

    def valid(self):
        return self.fullname != '' and self.fullname is not None

    def set_owner_bracket(self, bracket):
        self._bracket = bracket

    @property
    def fullname(self):
        return self._fullname

    @property
    def filename(self):
        return os.path.basename(self._fullname)

    @property
    def meta(self):
        return self._meta

    @property
    def data(self):
        """
        Lazy reading
        """
        self._lazy_read()
        return self._data


class ImageBracket:
    def __init__(self,images:List[Image], bracket_name:str=''):
        self._bracket_name = bracket_name
        self._images = images
        for img in self._images:
            img.set_owner_bracket(self)

    @property
    def images(self)->List[Image]:
        return self._images

    def remove_image(self, index: int)->Image:
        if index >=0 and index < len(self._images):
            ret = self._images[index]
            ret.set_owner_bracket(None)
        else:
            ret = Image()
        return ret

    @property
    def name(self):
        return self._bracket_name

def open_image_as_bracket(fullnames:List[str], max_count:int=0):
    images:List[Image] = []
    count = 0
    for fullname in fullnames:
        if os.path.isfile(fullname) == True:
            #TODO:: check if file is an image
            if max_count > 0 and count >= max_count:
                break
            images.append(Image(fullname))
            count += 1

    def pred(image:Image):
        return image.filename

    images.sort(key=pred)
    return [ImageBracket(images,'Bracket')]

def open_path_as_brackets(path: str, max_count:int=0)->List[ImageBracket]:
    """
    Open images in the given pass as an array of ImageBrakcet,
    each ImageBrackets includes images with different exposure
    """

    images:List[Image] = []
    count = 0
    for filename in os.listdir(path):
        fullname = os.path.join(path,filename)
        if os.path.isfile(fullname) == True:
            #TODO:: check if file is an image
            if max_count > 0 and count >= max_count:
                break
            images.append(Image(fullname))
            count += 1

    def pred(image:Image):
        return image.filename

    images.sort(key=pred)

    # After sort images by filename. we need group the into brackets.
    # We add them one by one from the first by order into a new bracket unitl
    # another image with the exsisting EV is encountered

    image_brackets: List[ImageBracket] = []
    ev_lut:Dict[float, int] = {}
    bracket: List[Image]= []
    count = 0
    for img in images:
        ev_val = img.meta.get('EV', None)
        if ev_val is None:
            print('Error: image {} has no EV value'.format(img.filename))
        if ev_val in ev_lut.keys():  # another image with the exsisting EV is encountered
            image_brackets.append(ImageBracket(bracket, 'bracket_{}'.format(count)))
            ev_lut = {}
            bracket = []
            count += 1

            ev_lut[ev_val] = 1
            bracket.append(img)
        else:
            ev_lut[ev_val] = 1
            bracket.append(img)

    if len(bracket) > 0:
        image_brackets.append(ImageBracket(bracket, 'bracket_{}'.format(count)))

    return image_brackets

def test1():
    test_path = 'D:/photo/2021-6-25/100MSDCF'
    brackets = open_path_as_brackets(test_path, 10)
    for b in brackets:
        print('Bracket {} includes: '.format(b.name()))
        for img in b.images:
            print('{} with {} ev'.format(img.filename, img.meta['EV']))

def test2():
    import os
    import sys
    from PIL import Image
    from PIL.ExifTags import TAGS
    print(sys.argv)
    image = sys.argv[1]
    for (tag,value) in Image.open(image)._getexif().iteritems():
            print('%s = %s' % (TAGS.get(tag), value))


def test3():
    import rawkit.metadata as rmt
    import rawkit.raw as r
    import sys
    with r.Raw(sys.argv[1]) as raw:
        print(raw.Metadata)

if __name__ == '__main__':
    test3()
    pass
