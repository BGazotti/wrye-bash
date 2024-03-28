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

#  Copyright (C) 2015-2016 Fadhil Mandaga (Original work)
#  Copyright (C) 2020 Sturla Bae (Modified work)
#  https://github.com/57ur14/pefile-extract-icon
#  Copyright (C) 2024 Wrye Bash Team
#  https://github.com/wrye-bash
#
# =============================================================================
import struct
import pefile

class ExtractIcon(object):
    GRPICONDIRENTRY_format = ('GRPICONDIRENTRY',
        ('B,Width', 'B,Height','B,ColorCount','B,Reserved',
         'H,Planes','H,BitCount','I,BytesInRes','H,ID'))
    GRPICONDIR_format = ('GRPICONDIR',
        ('H,Reserved', 'H,Type','H,Count'))
    RES_ICON = 1
    RES_CURSOR = 2

    def __init__(self, filepath=None, pefile_pe=None):
        if filepath:
            self.pe = pefile.PE(filepath)
        elif pefile_pe:
            self.pe = pefile_pe
        else:
            raise Exception('Neither filepath nor pefile PE object provided')

    def find_resource_base(self, type):
        rt_base_idx = None
        try:
            rt_base_idx = [entry.id for
                entry in self.pe.DIRECTORY_ENTRY_RESOURCE.entries].index(
                    pefile.RESOURCE_TYPE[type]
            )
        except Exception:
            pass

        if rt_base_idx is not None:
            return self.pe.DIRECTORY_ENTRY_RESOURCE.entries[rt_base_idx]

        return None

    def find_resource(self, type, res_index):
        rt_base_dir = self.find_resource_base(type)

        if res_index < 0:
            try:
                idx = [entry.id for entry in rt_base_dir.directory.entries].index(-res_index)
            except:
                return None
        else:
            try:
                idx = res_index if res_index < len(rt_base_dir.directory.entries) else None
            except AttributeError:
                idx = None

        if idx is None: return None

        test_res_dir = rt_base_dir.directory.entries[idx]
        res_dir = test_res_dir
        if test_res_dir.struct.DataIsDirectory:
            # another Directory
            # probably language take the first one
            try:
                res_dir = test_res_dir.directory.entries[0]
            except IndexError:
                return None
        if res_dir.struct.DataIsDirectory:
            # Ooooooooooiconoo no !! another Directory !!!
            return None

        return res_dir

    def get_group_icons(self):
        rt_base_dir = self.find_resource_base('RT_GROUP_ICON')
        if not rt_base_dir:
            return None
        groups = list()
        for res_index in range(0, len(rt_base_dir.directory.entries)):
            grp_icon_dir_entry = self.find_resource('RT_GROUP_ICON', res_index)

            if not grp_icon_dir_entry:
                continue

            data_rva = grp_icon_dir_entry.data.struct.OffsetToData
            size = grp_icon_dir_entry.data.struct.Size
            data = self.pe.get_memory_mapped_image()[data_rva:data_rva+size]
            if len(data) == 0:
                continue
            file_offset = self.pe.get_offset_from_rva(data_rva)

            grp_icon_dir = pefile.Structure(self.GRPICONDIR_format, file_offset=file_offset)
            grp_icon_dir.__unpack__(data)

            if grp_icon_dir.Reserved != 0 or grp_icon_dir.Type != self.RES_ICON:
                continue
            offset = grp_icon_dir.sizeof()

            entries = list()
            for idx in range(0, grp_icon_dir.Count):
                grp_icon = pefile.Structure(self.GRPICONDIRENTRY_format, file_offset=file_offset+offset)
                try:
                    grp_icon.__unpack__(data[offset:])
                except pefile.PEFormatError:
                    return None
                offset += grp_icon.sizeof()
                entries.append(grp_icon)

            groups.append(entries)
        return groups

    def best_icon(self, entries):
        b = 0
        w = 0
        best = None
        for i in range(len(entries)):
            icon = entries[i]
            if icon.BitCount > b:
                b = icon.BitCount
                best = i
            if icon.Width > w and icon.BitCount == b:
                w = icon.Width
                b = icon.BitCount
                best = i
        return best

    def get_windows_preferred_icon(self):
        """
        Identify the group and the index of the icon windows normally would display.
        Returns the group and the icon index. These values can be used as input for
        other functions in this class to retrieve the icon itself.
        """
        group_icons = self.get_group_icons()
        if group_icons != None:
            best_icon = 0
            for group in group_icons:
                if len(group) == 0:
                    continue
                best_icon = self.best_icon(group)
                return group, best_icon
        return None, None

    def get_raw_windows_preferred_icon(self):
        """
        Retrieve the (raw) icon that Windows normally would display.
        Returns the icon if a suitable icon was found.
        If no suitable icon was found, None is returned.
        """
        group, icon = self.get_windows_preferred_icon()
        if group != None and icon != None:
            return self.export_raw(group, icon)
        else:
            return None

    def get_icon(self, index):
        icon_entry = self.find_resource('RT_ICON', -index)
        if not icon_entry:
            return None

        data_rva = icon_entry.data.struct.OffsetToData
        size = icon_entry.data.struct.Size
        data = self.pe.get_memory_mapped_image()[data_rva:data_rva+size]

        return data

    def export_raw(self, entries, index = None):
        if index is not None:
            entries = entries[index:index+1]

        ico = struct.pack('<HHH', 0, self.RES_ICON, len(entries))
        data_offset = None
        data = []
        info = []
        for grp_icon in entries:
            if data_offset is None:
                data_offset = len(ico) + ((grp_icon.sizeof()+2) * len(entries))

            nfo = grp_icon.__pack__()[:-2] + struct.pack('<L', data_offset)
            info.append( nfo )

            raw_data = self.get_icon(grp_icon.ID)
            if not raw_data: continue

            data.append( raw_data )
            data_offset += len(raw_data)

        raw = ico + b''.join(info) + b''.join(data)
        return raw

    def export(self, entries, index = None):
        raw = self.export_raw(entries, index)
        import wx
        img = wx.Image(len(raw),raw)
        return img  # TODO wx image from bytes, proper scali
