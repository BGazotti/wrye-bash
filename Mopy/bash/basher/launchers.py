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
#  Wrye Bash copyright (C) 2005-2009 Wrye, 2010-2023 Wrye Bash Team
#  https://github.com/wrye-bash
#
# =============================================================================
import os

from .. import bass

class Launcher:

    _path: str
    _args: str

    def __init__(self, path: str, args: str):
        """Simple 'Launcher' object consisting of a
        path to a file and a string of arguments. """
        self._path = path
        self._args = args
    @property
    def path(self):
        return self._path

    @property
    def args(self):
        return self._args

# Internals ===================================================================

def _save_launcher(launcher_name: str, launcher: Launcher):
    with open(os.path.join(bass.dirs[u'launchers'], launcher_name), mode='w') \
            as launcher_file:
        launcher_file.writelines((launcher.path.strip(),
                                  os.linesep, launcher.args.strip()))


def _load_launchers() -> dict[str | bytes, Launcher]:
    """Reads all configured launchers from where they're saved."""

    launcher_list = {}
    for launcher_filename in os.listdir(bass.dirs['launchers']):
        with open(os.path.join(bass.dirs['launchers'], launcher_filename))\
                as launcher_file:
            read_launcher = Launcher(launcher_file.readline().strip(),
                launcher_file.readline().strip())
            # need to strip to not stack newlines
        if read_launcher:
            launcher_list[launcher_filename] = read_launcher
    return launcher_list

_launcher_list = _load_launchers()
"""Dictionary where the key is the displayed launcher name (as well as its file
name under the Launchers folder) and value is a Launcher object.
Why is it a dictionary? Because that way the key mapping pairs well with the
ListBox. There's no other reason."""

# Wrappers for the launcher dict ==============================================
def retrieve_launchers():
    """Returns a shallow copy of the dictionary of available launchers;
    modifications to this dict will not be passed on to the (file) backend.
    Changes to the launchers must be made via the wrapper methods."""
    return _launcher_list.copy()

def save_launcher(launcher_name: str, launcher: Launcher):
    """Saves a launcher.
    launcher_name: the launcher's displayed name, dictionary key and file
    name."""
    _launcher_list[launcher_name] = launcher
    _save_launcher(launcher_name, launcher)


def remove_launcher(launcher_name: str):
    """Removes a launcher, by name."""
    del _launcher_list[launcher_name]
    os.remove(os.path.join(bass.dirs['launchers'], launcher_name))


