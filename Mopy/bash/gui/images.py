# -*- coding: utf-8 -*-
#
# GPL License and Copyright Notice ============================================
#  This file is part of Wrye Bash.
#
#  Wrye Bash is free software: you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation, either version 3
#  of the License, or (at your option) any later version.
#
#  Wrye Bash is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Wrye Bash.  If not, see <https://www.gnu.org/licenses/>.
#
#  Wrye Bash copyright (C) 2005-2009 Wrye, 2010-2024 Wrye Bash Team
#  https://github.com/wrye-bash
#
# =============================================================================
"""Encapsulate wx images."""
from __future__ import annotations

import os
from io import BytesIO
from math import ceil, log
from struct import unpack_from

import wx as _wx
import wx.svg as _svg
from wx import BITMAP_SCREEN_DEPTH, BITMAP_TYPE_ICO

from ._gui_globals import get_image, get_image_dir
from .base_components import Lazy, scaled
from ..bolt import deprint, Path
from ..exception import ArgumentError

class GuiImage(Lazy):
    """Wrapper around various native image classes."""
    # allow to directly access the _native_window (force via _resolve?)
    _bypass_native_init = True

    img_types = {
        '.bmp': _wx.BITMAP_TYPE_BMP,
        '.ico': _wx.BITMAP_TYPE_ICO,
        '.jpeg': _wx.BITMAP_TYPE_JPEG,
        '.jpg': _wx.BITMAP_TYPE_JPEG,
        '.png': _wx.BITMAP_TYPE_PNG,
        '.svg': None, # Special handling needed, see _is_svg
        '.tif': _wx.BITMAP_TYPE_TIF,
        '.tga': _wx.BITMAP_TYPE_TGA,
    }

    def __init__(self, img_path, iconSize=-1, imageType=None, quality=None,
            img_data:bytes=None, *args, **kwargs):
        self._img_data=img_data
        self._img_path = img_path
        if not self.allow_create():
            raise ArgumentError(f'Missing resource file: {self._img_path}')
        super().__init__(*args, **kwargs)
        self.iconSize = iconSize
        self._img_type = imageType
        self._quality = quality

    def allow_create(self):
        # FIXME what?
        return (self._img_path is None and self._img_data) or os.path.exists(self._img_path.split(';')[0])

    def rescaled(self, w, h):
        """Returns a rescaled wx.Bitmap object from this one."""
        new = self._native_widget.ConvertToImage()
        new.Rescale(w,h)
        return _wx.Bitmap(new)

    def get_img_size(self):
        return self._native_widget.GetWidth(), self._native_widget.GetHeight()

    @classmethod
    def from_path(cls, img_path: str| Path, imageType=None, iconSize=-1,
                  quality=None):
        """Static factory - creates an Image component from an image file."""
        _root, extension = os.path.splitext(img_path := f'{img_path}')
        try:
            img_type = imageType or cls.img_types[extension.lower()]
        except KeyError:
            deprint(f'Unknown image extension {extension}')
            img_type = _wx.BITMAP_TYPE_ANY
        if (is_svg := img_type is None) and iconSize == -1:
            raise ArgumentError('You must specify iconSize to '
                                'rasterize an SVG to a bitmap!')
        if not os.path.isabs(img_path):
            img_path = os.path.join(get_image_dir(), img_path)
        if cls is not GuiImage:
            return cls(img_path, iconSize, img_type, quality)
        if img_type == _wx.BITMAP_TYPE_ICO:
            return _BmpFromIcoPath(img_path, iconSize, img_type, quality)
        elif is_svg:
            return _SvgFromPath(img_path, iconSize, img_type, quality)
        else:
            return _BmpFromPath(img_path, iconSize, img_type, quality)

class _SvgFromPath(GuiImage):
    """Wrap an svg."""
    _native_widget: _wx.BitmapBundle.FromBitmaps

    @property
    def _native_widget(self):
        if not self._is_created():
            with open(self._img_path, 'rb') as ins:
                svg_data = ins.read()
            if b'var(--invert)' in svg_data:
                svg_data = svg_data.replace(b'var(--invert)',
                    b'#FFF' if self._should_invert_svg() else b'#000')
            svg_img = _svg.SVGimage.CreateFromBytes(svg_data)
            # Use a bitmap bundle so we get an actual high-res asset at high
            # DPIs, rather than wx deciding to scale up the low-res asset
            wanted_svgs = [svg_img.ConvertToScaledBitmap((s, s))
                           for s in (self.iconSize, scaled(self.iconSize))]
            self._cached_args = (wanted_svgs,)
        return super()._native_widget

    @staticmethod
    def _should_invert_svg():
        from .. import bass
        return bass.settings['bash.use_reverse_icons']

class IcoFromPng(GuiImage):
    """Create a wx.Icon from a GuiImage instance - no new uses please!"""
    _native_widget: _wx.Icon

    def __init__(self, gui_image):
        super(GuiImage, self).__init__() # bypass GuiImage.__init__
        self._gui_image = gui_image

    @property
    def _native_widget(self):
        if self._is_created(): return self._cached_widget
        native = super()._native_widget # create a plain wx.Icon
        native_bmp = self._resolve(self._gui_image)
        if isinstance(native_bmp, _wx.BitmapBundle):
            native_bmp = native_bmp.GetBitmap(native_bmp.GetDefaultSize())
        native.CopyFromBitmap(native_bmp)
        return native

class _IcoFromPath(GuiImage):
    """Only used internally in _BmpFromIcoPath."""
    _native_widget: _wx.Icon

    @property
    def _native_widget(self):
        if self._is_created(): return self._cached_widget
        self._cached_args = self._img_path, self._img_type, self.iconSize, \
            self.iconSize
        widget = super()._native_widget
        # we failed to get the icon? (when display resolution changes)
        ##: Ut: I (hope I) carried previous logic to new API but is there a
        # better way (and/or any leaks)?
        if not all(self.get_img_size()):
            self._cached_args = self._img_path, _wx.BITMAP_TYPE_ICO
            self.native_destroy()
            return super()._native_widget
        return widget
class _IcoFromRaw(GuiImage):
    _native_widget: _wx.Icon

    @property
    def _native_widget(self):
        if self._is_created(): return self._cached_widget
        self._cached_args = self._img_data, self._img_path, self._img_type, self.iconSize, \
            self.iconSize
        class IcoFile:
            def __init__(self, buf):
                s = buf.read(6)
                if s[:4] != b"\0\0\1\0":
                    msg = "not an ICO file"
                    raise SyntaxError(msg)

                self.buf=buf
                self.entry=[]
                # Number of items in file

                self.nb_items = unpack_from("<H", s, 4)[0]

                # Get headers for each item
                for i in range(self.nb_items):
                    s = buf.read(16)

                    icon_header = {
                        "width": s[0],
                        "height": s[1],
                        "nb_color": s[2],
                        # No. of colors in image (0 if >=8bpp)
                        "reserved": s[3],
                        "planes": unpack_from("<H", s, 4)[0],
                        "bpp": unpack_from("<H", s, 6)[0],
                        "size": unpack_from("<H", s, 8)[0],
                        "offset": unpack_from("<H", s, 12)[0],
                    }

                    # See Wikipedia
                    for j in ("width", "height"):
                        if not icon_header[j]:
                            icon_header[j] = 256

                    # See Wikipedia notes about color depth.
                    # We need this just to differ images with equal sizes
                    icon_header["color_depth"] = (
                            icon_header["bpp"]
                            or (
                                    icon_header["nb_color"] != 0
                                    and ceil(log(icon_header["nb_color"], 2))
                            )
                            or 256
                    )

                    icon_header["dim"] = (
                    icon_header["width"], icon_header["height"])
                    icon_header["square"] = icon_header["width"] * icon_header[
                        "height"]

                    self.entry.append(icon_header)

                self.entry = sorted(self.entry, key=lambda x: x["color_depth"])
                # ICO images are usually squares
                self.entry = sorted(self.entry, key=lambda x: x["square"],
                    reverse=True)

        pog = IcoFile(BytesIO(self._img_data))
        img = _wx.Bitmap(pog.buf.read() ,pog.entry[0]['width'],pog.entry[0][
                    'height'])
        img = _wx.Image(BytesIO(self._img_data),BITMAP_TYPE_ICO)
        img.Scale(16,16)
        return _wx.Bitmap(img)

class Ico2Bitmap:
    @classmethod
    def conv(cls, ico_data: bytes) -> (int,int,bytes,bool):
        """Parse image from file-like object containing ico file data"""
        buf = BytesIO(ico_data)
        # check magic
        s = buf.read(6)
        if s[:4] != b"\0\0\1\0":
            msg = "not an ICO file"
            raise SyntaxError(msg)

        s = buf.read(16)
        width = s[0]
        height = s[1]
        has_alpha = unpack_from("<H", s, 6)[0] == 32 # bit depth
        return width, height, ico_data, has_alpha,



        # Number of items in file
        # Get headers for each item
        for i in range(unpack_from("<H", s, 4)[0]):
            s = buf.read(16)

            icon_header = {
                "width": s[0],
                "height": s[1],
                "nb_color": s[2],
                # No. of colors in image (0 if >=8bpp)
                "reserved": s[3],
                "planes": unpack_from("<H", s, 4)[0],
                "bpp": unpack_from("<H", s, 6)[0],
                "size": unpack_from("<H", s, 8)[0],
                "offset": unpack_from("<H", s, 12)[0],
            }

            # See Wikipedia
            for j in ("width", "height"):
                if not icon_header[j]:
                    icon_header[j] = 256

            # See Wikipedia notes about color depth.
            # We need this just to differ images with equal sizes
            icon_header["color_depth"] = (
                    icon_header["bpp"]
                    or (
                            icon_header["nb_color"] != 0
                            and ceil(log(icon_header["nb_color"], 2))
                    )
                    or 256
            )

            icon_header["dim"] = (
                icon_header["width"], icon_header["height"])
            icon_header["square"] = icon_header["width"] * icon_header[
                "height"]

            self.entry.append(icon_header)

        self.entry = sorted(self.entry, key=lambda x: x["color_depth"])
        # ICO images are usually squares
        self.entry = sorted(self.entry, key=lambda x: x["square"],
            reverse=True)

class IcoFile:
    def __init__(self, buf):
        """
        Parse image from file-like object containing ico file data
        """

        # check magic




    def megaparsemapam(self) -> (bytes, int, int):
        return self.buf, self.entry[0]['width'], self.entry[0]['height']
    def sizes(self):
        """
        Get a list of all available icon sizes and color depths.
        """
        return {(h["width"], h["height"]) for h in self.entry}

    def getentryindex(self, size, bpp=False):
        for i, h in enumerate(self.entry):
            if size == h["dim"] and (bpp is False or bpp == h["color_depth"]):
                return i
        return 0

    def getimage(self, size, bpp=False):
        """
        Get an image from the icon
        """
        return self.frame(self.getentryindex(size, bpp))

    def frame(self, idx):
        """
        Get an image from frame idx
        """

        header = self.entry[idx]

        self.buf.seek(header["offset"])
        data = self.buf.read(8)
        self.buf.seek(header["offset"])

        if data[:8] == PngImagePlugin._MAGIC:
            # png frame
            im = PngImagePlugin.PngImageFile(self.buf)
            Image._decompression_bomb_check(im.size)
        else:
            # XOR + AND mask bmp frame
            im = BmpImagePlugin.DibImageFile(self.buf)
            Image._decompression_bomb_check(im.size)

            # change tile dimension to only encompass XOR image
            im._size = (im.size[0], int(im.size[1] / 2))
            d, e, o, a = im.tile[0]
            im.tile[0] = d, (0, 0) + im.size, o, a

            # figure out where AND mask image starts
            bpp = header["bpp"]
            if 32 == bpp:
                # 32-bit color depth icon image allows semitransparent areas
                # PIL's DIB format ignores transparency bits, recover them.
                # The DIB is packed in BGRX byte order where X is the alpha
                # channel.

                # Back up to start of bmp data
                self.buf.seek(o)
                # extract every 4th byte (eg. 3,7,11,15,...)
                alpha_bytes = self.buf.read(im.size[0] * im.size[1] * 4)[3::4]

                # convert to an 8bpp grayscale image
                mask = Image.frombuffer(
                    "L",  # 8bpp
                    im.size,  # (w, h)
                    alpha_bytes,  # source chars
                    "raw",  # raw decoder
                    ("L", 0, -1),  # 8bpp inverted, unpadded, reversed
                )
            else:
                # get AND image from end of bitmap
                w = im.size[0]
                if (w % 32) > 0:
                    # bitmap row data is aligned to word boundaries
                    w += 32 - (im.size[0] % 32)

                # the total mask data is
                # padded row size * height / bits per char

                total_bytes = int((w * im.size[1]) / 8)
                and_mask_offset = header["offset"] + header["size"] - total_bytes

                self.buf.seek(and_mask_offset)
                mask_data = self.buf.read(total_bytes)

                # convert raw data to image
                mask = Image.frombuffer(
                    "1",  # 1 bpp
                    im.size,  # (w, h)
                    mask_data,  # source chars
                    "raw",  # raw decoder
                    ("1;I", int(w / 8), -1),  # 1bpp inverted, padded, reversed
                )

                # now we have two images, im is XOR image and mask is AND image

            # apply mask image as alpha channel
            im = im.convert("RGBA")
            im.putalpha(mask)

        return im

class _BmpFromIcoPath(GuiImage):
    _native_widget: _wx.Bitmap

    @property
    def _native_widget(self):
        if self._is_created(): return self._cached_widget
        img_ico = _IcoFromPath(self._img_path, self.iconSize, self._img_type)
        w, h = img_ico.get_img_size()
        self._cached_args = w, h
        native = super()._native_widget
        native.CopyFromIcon(self._resolve(img_ico))
        # Hack - when user scales windows display icon may need scaling
        if (self.iconSize != -1 and w != self.iconSize or
            h != self.iconSize): # rescale !
            scaled = native.ConvertToImage().Scale(self.iconSize,
                self.iconSize, _wx.IMAGE_QUALITY_HIGH)
            self._cached_args = scaled,
            return super()._native_widget
        return native

class ImgFromPath(GuiImage):
    """Used internally in _BmpFromPath but also used to create a wx.Image
    directly."""
    _native_widget: _wx.Image

    @property
    def _native_widget(self):
        if self._is_created(): return self._cached_widget
        self._cached_args = self._img_path, self._img_type
        native = super()._native_widget
        if self.iconSize != -1:
            # Don't use the scaled icon size here - _BmpFromPath performs its
            # own scaling and Screen_ConvertTo wouldn't want to scale anyways
            wanted_size = self.iconSize
            if self.get_img_size() != (wanted_size, wanted_size):
                native.Rescale(wanted_size, wanted_size,
                    _wx.IMAGE_QUALITY_HIGH)
        if self._quality is not None: # This only has an effect on jpgs
            native.SetOption(_wx.IMAGE_OPTION_QUALITY, self._quality)
        return native

    def save_bmp(self, imagePath, exten='.jpg'):
        return self._native_widget.SaveFile(imagePath, self.img_types[exten])

class _BmpFromPath(GuiImage):
    _native_widget: _wx.BitmapBundle.FromBitmaps

    @property
    def _native_widget(self):
        # Pass wx.Image to wx.Bitmap
        base_img: _wx.Image = self._resolve(ImgFromPath(self._img_path,
            imageType=self._img_type))
        scaled_imgs = [base_img]
        if self.iconSize != -1:
            # If we can, also add a scaled-up version so wx stops trying to
            # scale this by itself - using a higher-res image here if we have
            # one would be better, but that would be very difficult to
            # implement, something for the (far) future
            wanted_size = scaled(self.iconSize)
            scaled_imgs.append(base_img.Scale(wanted_size, wanted_size,
                quality=_wx.IMAGE_QUALITY_HIGH))
        self._cached_args = (list(map(_wx.Bitmap, scaled_imgs)),)
        return super()._native_widget

class BmpFromStream(GuiImage):
    """Call init directly - hmm."""
    _native_widget: _wx.Bitmap

    def __init__(self, bm_width, bm_height, stream_data, with_alpha):
        super(GuiImage, self).__init__() # bypass GuiImage.__init__
        self._with_alpha = with_alpha
        self._stream_data = stream_data
        self._bm_height = bm_height
        self._bm_width = bm_width

    @property
    def _native_widget(self):
        if self._is_created(): return self._cached_widget
        wx_depth = (32 if self._with_alpha else 24)
        wx_fmt = (_wx.BitmapBufferFormat_RGBA if self._with_alpha
                  else _wx.BitmapBufferFormat_RGB)
        self._cached_args = (self._bm_width, self._bm_height, wx_depth)
        native = super()._native_widget
        native.CopyFromBuffer(self._stream_data, wx_fmt)
        self._stream_data = None # save some memory
        return native

    def save_bmp(self, imagePath, exten='.jpg'):
        self._native_widget.ConvertToImage()
        return self._native_widget.SaveFile(imagePath, self.img_types[exten])

class StaticBmp(GuiImage):
    """This one has a parent and a default value - we should generalize the
    latter."""
    _native_widget: _wx.StaticBitmap

    def __init__(self, parent, gui_image=None):
        super(GuiImage, self).__init__( # bypass GuiImage.__init__
            bitmap=self._resolve(gui_image or get_image('warning.32')))
        self._parent = parent

#------------------------------------------------------------------------------
class ImageList(Lazy):
    """Wrapper for wx.ImageList. Allows ImageList to be specified before
    wx.App is initialized."""
    _native_widget: _wx.ImageList

    def __init__(self, il_width, il_height):
        super().__init__()
        self.width = il_width
        self.height = il_height
        self._images = []
        self._indices = None

    @property
    def _native_widget(self):
        if self._is_created(): return self._cached_widget
        # scaling crashes if done before the wx.App is initialized
        self._cached_args = scaled(self.width), scaled(self.height)
        return super()._native_widget

    def native_init(self, *args, **kwargs):
        kwargs.setdefault('recreate', False)
        freshly_created = super().native_init(*args, **kwargs)
        ##: Accessing these like this feels wrong - maybe store the scaled size
        # somewhere and retrieve it here?
        scaled_sb_size = self._cached_args[0:2]
        if freshly_created: # ONCE! we don't support adding more images
            self._indices = {}
            for k, im in self._images:
                nat_img = self._resolve(im)
                if isinstance(nat_img, _wx.BitmapBundle):
                    nat_img = nat_img.GetBitmap(scaled_sb_size)
                self._indices[k] = self._native_widget.Add(nat_img)

    def img_dex(self, *args) -> int | None:
        """Return the index of the specified image in the native control."""
        return None if (a := args[0]) is None else self._indices[a]
