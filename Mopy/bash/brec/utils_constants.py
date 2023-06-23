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
"""Houses the parts of brec that didn't fit anywhere else or were needed by
almost all other parts of brec."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from itertools import chain

from .. import bolt
from ..bolt import Flags, attrgetter_cache, cstrip, decoder, flag, \
    structs_cache
from ..exception import StateError

# no local imports, imported everywhere in brec

# Form ids --------------------------------------------------------------------
class FormId:
    """Immutable class wrapping an (integer) plugin form ID. These must be
    instantiated in a ModReader context which injects the master table to
    use for the long fid conversions. Base class performs no conversion."""

    def __init__(self, int_val):
        if not isinstance(int_val, int):
            ##: re add : {int_val!r} when setDefault is gone - huge performance
            # impact as it run for all _Tes4Fid and blows - repr is expensive!
            raise TypeError('Only int accepted in FormId')
        self.short_fid = int_val

    # factories
    __master_formid_type: dict[bolt.FName, type] = {} # cache a formid type per mod
    _form_id_classes: dict[tuple[bolt.FName], type] = {} # cache a formid type per masters list
    @classmethod
    def from_tuple(cls, fid_tuple):
        """Return a FormId (subclass) instance with a given long id - does not
        implement mod_dex which means use sparingly - mostly used for
        parsers (csvs) and through game.master_fid."""
        try:
            return cls.__master_formid_type[fid_tuple[0]](fid_tuple[1])
        except KeyError:
            class __FormId(cls):
                @bolt.fast_cached_property
                def long_fid(self):
                    return fid_tuple[0], self.short_fid
                @property
                def mod_dex(self):
                    """short_fid corresponds to object_dex in this case."""
                    raise StateError(f'mod_dex undefined for {self} built '
                                     f'from tuple')
            cls.__master_formid_type[fid_tuple[0]] = __FormId
            return __FormId(fid_tuple[1])

    @classmethod
    def from_object_id(cls, modIndex, objectIndex):
        """Return a FormId instance with a shortid generated by a mod and
        object index - use sparingly!"""
        return cls(objectIndex | (modIndex << 24))

    @staticmethod
    def from_masters(augmented_masters):
        """Return a subclass of FormId using the specified masters for long fid
        conversions."""
        try:
            form_id_type = FormId._form_id_classes[augmented_masters]
        except KeyError:
            class _FormID(FormId):
                @bolt.fast_cached_property
                def long_fid(self, *, __masters=augmented_masters):
                    try:
                        return __masters[self.mod_dex], \
                               self.short_fid & 0xFFFFFF
                    except IndexError:
                        # Clamp HITMEs by using at most max_masters for master
                        # index
                        return __masters[-1], self.short_fid & 0xFFFFFF
            form_id_type = FormId._form_id_classes[augmented_masters] = _FormID
        return form_id_type

    @bolt.fast_cached_property
    def long_fid(self):
        """Don't map by default."""
        return self.short_fid

    @property # ~0.006s on a 60s BP - no need to cache
    def object_dex(self):
        """Always recoverable from short fid."""
        return self.short_fid & 0x00FFFFFF

    @property
    def mod_dex(self):
        """Always recoverable from short fid - but see from_tuple."""
        return self.short_fid >> 24

    @property # ~0.03s on a 60s BP - no need to cache
    def mod_fn(self):
        """Return the mod id - will raise if long_fid is not a tuple."""
        try:
            return self.long_fid[0]
        except TypeError:
            raise StateError(f'{self!r} not in long format')

    def is_null(self):
        """Return True if we are a round 0."""
        # Use object_dex instead of short_fid here since 01000000 is also NULL
        return self.object_dex == 0

    # Hash and comparisons
    def __hash__(self):
        return hash(self.long_fid)

    def __eq__(self, other):
        with suppress(AttributeError):
            return self.long_fid == other.long_fid
        if other is None:
            return False
        elif isinstance(self.long_fid, type(other)):
            return self.long_fid == other
        return NotImplemented

    def __ne__(self, other):
        with suppress(AttributeError):
            return self.long_fid != other.long_fid
        if other is None:
            return True
        elif isinstance(self.long_fid, type(other)):
            return self.long_fid != other
        return NotImplemented

    def __lt__(self, other):
        with suppress(TypeError):
            # If we're in a write context, compare FormIds properly
            return short_mapper(self) < short_mapper(other)
        # Otherwise, use alphanumeric order
        ##: This is a hack - rewrite _AMerger to not sort and absorb all
        # mergers (see #497). Same with all the other compare dunders
        with suppress(AttributeError):
            return self.long_fid < other.long_fid
        if isinstance(self.long_fid, type(other)):
            return self.long_fid < other
        return NotImplemented

    def __ge__(self, other):
        with suppress(TypeError):
            return short_mapper(self) >= short_mapper(other)
        with suppress(AttributeError):
            return self.long_fid >= other.long_fid
        if isinstance(self.long_fid, type(other)):
            return self.long_fid >= other
        return NotImplemented

    def __gt__(self, other):
        with suppress(TypeError):
            return short_mapper(self) > short_mapper(other)
        with suppress(AttributeError):
            return self.long_fid > other.long_fid
        if isinstance(self.long_fid, type(other)):
            return self.long_fid > other
        return NotImplemented

    def __le__(self, other):
        with suppress(TypeError):
            return short_mapper(self) <= short_mapper(other)
        with suppress(AttributeError):
            return self.long_fid <= other.long_fid
        if isinstance(self.long_fid, type(other)):
            return self.long_fid <= other
        return NotImplemented

    # avoid setstate/getstate round trip
    def __deepcopy__(self, memodict={}):
        return self # immutable

    def __copy__(self):
        return self # immutable

    def __getstate__(self):
        raise NotImplementedError("You can't pickle a FormId")

    def __str__(self):
        if isinstance(self.long_fid, tuple):
            return f'({self.long_fid[0]}, {self.long_fid[1]:06X})'
        else:
            return f'{self.long_fid:08X}'

    def __repr__(self):
        return f'{type(self).__name__}({self})'

    # Action API --------------------------------------------------------------
    def dump(self):
        return short_mapper(self)

class _NoneFid:
    """Special FormId value of NONE, which sorts last always.  Used in FO4, and
    internally for sorted lists which don't have a FormId but need to sort last.

    NOTE: Not derived from FormId, since we want this to blow if FormId's other
    methods are called on this.
    """
    def __init__(self):
        pass

    def __str__(self) -> str:
        return 'NONE'

    def __repr__(self) -> str:
        return 'FormId(NONE)'

    def __lt__(self, other: FormId | _NoneFid) -> bool:
        if isinstance(other, (FormId, _NoneFid)):
            return False
        return NotImplemented

    def __le__(self, other: FormId | _NoneFid) -> bool:
        return not self > other

    def __gt__(self, other: FormId | _NoneFid) -> bool:
        if isinstance(other, FormId):
            return True
        elif isinstance(other, _NoneFid):
            return False
        return NotImplemented

    def __ge__(self, other: FormId | _NoneFid) -> bool:
        return not self < other

    def __eq__(self, other: FormId | _NoneFid) -> bool:
        if isinstance(other, FormId):
            return False
        elif isinstance(other, _NoneFid):
            return True
        return NotImplemented

    def __ne__(self, other: FormId | _NoneFid) -> bool:
        return not self == other

    def dump(self) -> int:
        return 0xFFFFFFFF

class _Tes4Fid(FormId):
    """The special formid of the plugin header record - aka 0. Also used
    as a MelStruct default and when we set the form id to "zero" in some
    edge cases."""
    def dump(self): return 0

    @bolt.fast_cached_property
    def long_fid(self):
        from .. import bush
        return bush.game.master_fid(0).long_fid

# cache an instance of Tes4 and export that to the rest of Bash
ZERO_FID = _Tes4Fid(0)
NONE_FID = _NoneFid()

# Global FormId class used to wrap all formids of currently loading mod. It
# must be set by the mod reader context manager based on the currently loading
# plugin
FORM_ID: type[FormId] | None = None

# Global short mapper function. Set by the plugin output context manager.
# Maps the fids based on the masters of the currently dumped plugin
short_mapper: Callable | None = None
short_mapper_no_engine: Callable | None = None

# Used by Mel classes to wrap fid elements.
FID = lambda x: FORM_ID(x)

class _DummyFid(_Tes4Fid):
    """Used by setDefault (yak) - will blow on dump, make sure you replace
    it with a proper FormId."""
    def dump(self):
        raise NotImplementedError('Dumping a dummy fid')
DUMMY_FID = _DummyFid(0)

# Random stuff ----------------------------------------------------------------
int_unpacker = structs_cache['I'].unpack

class FixedString(str):
    """An action for MelStructs that will decode and encode a fixed-length
    string. Note that you do not need to specify defaults when using this."""
    __slots__ = ('_str_length',)
    _str_encoding = bolt.pluginEncoding

    def __new__(cls, str_length, target_str: str | bytes = ''):
        if isinstance(target_str, str):
            decoded_str = target_str
        else:
            decoded_str = '\n'.join(
                decoder(x, cls._str_encoding, avoidEncodings=('utf8', 'utf-8'))
                for x in cstrip(target_str).split(b'\n'))
        new_str = super(FixedString, cls).__new__(cls, decoded_str)
        new_str._str_length = str_length
        return new_str

    def __call__(self, new_str):
        # 0 is the default, so replace it with whatever we currently have
        return self.__class__(self._str_length, new_str or str(self))

    def __deepcopy__(self, memodict={}):
        return self # immutable

    def __copy__(self):
        return self # immutable

    def dump(self):
        return bolt.encode_complex_string(self, max_size=self._str_length,
                                          min_size=self._str_length)

class AutoFixedString(FixedString):
    """Variant of FixedString that uses chardet to detect encodings."""
    _str_encoding = None

# Common flags ----------------------------------------------------------------
class AMgefFlags(Flags):
    """Base class for MGEF data flags shared by all games."""
    hostile: bool = flag(0)
    recover: bool = flag(1)
    detrimental: bool = flag(2)
    no_hit_effect: bool = flag(27)

class AMgefFlagsTes4(AMgefFlags):
    """Base class for MGEF data flags from Oblivion to FO3."""
    mgef_self: bool = flag(4)
    mgef_touch: bool = flag(5)
    mgef_target: bool = flag(6)
    no_duration: bool = flag(7)
    no_magnitude: bool = flag(8)
    no_area: bool = flag(9)
    fx_persist: bool = flag(10)
    use_skill: bool = flag(19)
    use_attribute: bool = flag(20)
    spray_projectile_type: bool = flag(25)
    bolt_projectile_type: bool = flag(26)

    @property
    def fog_projectile_type(self) -> bool:
        """If flags 25 and 26 are set, specifies fog_projectile_type."""
        mask = 0b00000110000000000000000000000000
        return (self._field & mask) == mask

    @fog_projectile_type.setter
    def fog_projectile_type(self, new_fpt: bool) -> None:
        mask = 0b00000110000000000000000000000000
        new_bits = mask if new_fpt else 0
        self._field = (self._field & ~mask) | new_bits

class MgefFlags(AMgefFlags):
    """Implements the MGEF data flags used since Skyrim."""
    snap_to_navmesh: bool = flag(3)
    no_hit_event: bool = flag(4)
    dispel_with_keywords: bool = flag(8)
    no_duration: bool = flag(9)
    no_magnitude: bool = flag(10)
    no_area: bool = flag(11)
    fx_persist: bool = flag(12)
    gory_visuals: bool = flag(14)
    hide_in_ui: bool = flag(15)
    no_recast: bool = flag(17)
    power_affects_magnitude: bool = flag(21)
    power_affects_duration: bool = flag(22)
    painless: bool = flag(26)
    no_death_dispel: bool = flag(28)

##: xEdit marks these as unknown_is_unused, at least in Skyrim, but it makes no
# sense because it also marks all 32 of its possible flags as known
class BipedFlags(Flags):
    """Base Biped flags element. Includes logic for checking if armor/clothing
    can be marked as playable. Should be subclassed to add the appropriate
    flags and, if needed, the non-playable flags."""
    _not_playable_flags: set[str] = set()

    @property
    def any_body_flag_set(self) -> bool:
        check_flags = set(type(self)._names) - type(self)._not_playable_flags
        return any(getattr(self, flg_name) for flg_name in check_flags)

# Sort Keys -------------------------------------------------------------------
fid_key = attrgetter_cache[u'fid']

_perk_type_to_attrs = {
    0: attrgetter_cache[('pe_quest', 'pe_quest_stage')],
    1: attrgetter_cache['pe_ability'],
    2: attrgetter_cache[('pe_entry_point', 'pe_function')],
}

def perk_effect_key(e):
    """Special sort key for PERK effects."""
    perk_effect_type = e.pe_type
    # The first three are always present, the others depend on the perk
    # effect's type
    extra_vals = _perk_type_to_attrs[perk_effect_type](e)
    if not isinstance(extra_vals, tuple):
        # Second case from above, only a single attribute returned.
        # DATA subrecords are sometimes absent after the PRKE subrecord,
        # leading to a None for pe_ability - sort those last (valid IDs
        # shouldn't be 0)
        return (e.pe_rank, e.pe_priority, perk_effect_type,
                extra_vals or NONE_FID)
    else:
        return e.pe_rank, e.pe_priority, perk_effect_type, *extra_vals

def gen_coed_key(base_attrs: tuple[str, ...]):
    """COED is optional, so all of its attrs may be None. Account
    for that to avoid TypeError when some entries have COED present
    and some don't."""
    base_attrgetter = attrgetter_cache[base_attrs]
    def _ret_key(e):
        return (*base_attrgetter(e), e.item_condition or 0.0,
                e.item_owner or NONE_FID, e.item_global or NONE_FID)
    return _ret_key

# Constants -------------------------------------------------------------------

# Null strings (for default empty byte arrays)
null1 = b'\x00'
null2 = null1 * 2
null3 = null1 * 3
null4 = null1 * 4

# TES4 Group/Top Types
group_types = {0: 'Top', 1: 'World Children', 2: 'Interior Cell Block',
               3: 'Interior Cell Sub-Block', 4: 'Exterior Cell Block',
               5: 'Exterior Cell Sub-Block', 6: 'Cell Children',
               7: 'Topic Children', 8: 'Cell Persistent Children',
               9: 'Cell Temporary Children',
               10: 'Cell Visible Distant Children/Quest Children'}

# Helpers ---------------------------------------------------------------------
def get_structs(struct_format):
    """Create a struct and return bound unpack, pack and size methods in a
    tuple."""
    _struct = structs_cache[struct_format]
    return _struct.unpack, _struct.pack, _struct.size

def gen_color(color_attr_pfx: str) -> list[str]:
    """Helper method for generating red/green/blue/unused color attributes."""
    return [f'{color_attr_pfx}_{c}' for c in ('red', 'green', 'blue',
                                              'unused')]

def gen_color3(color_attr_pfx: str) -> list[str]:
    """Helper method for generating red/green/blue color attributes."""
    return [f'{color_attr_pfx}_{c}' for c in ('red', 'green', 'blue')]

def gen_ambient_lighting(attr_prefix):
    """Helper method for generating a ton of repetitive attributes that are
    shared between a couple record types (wbAmbientColors in xEdit)."""
    color_types = [f'directional_{t}' for t in (
        'x_plus', 'x_minus', 'y_plus', 'y_minus', 'z_plus', 'z_minus')]
    color_types.append('specular')
    color_iters = chain.from_iterable(gen_color(d) for d in color_types)
    ambient_lighting = [f'{attr_prefix}_ac_{x}' for x in color_iters]
    return ambient_lighting + [f'{attr_prefix}_ac_scale']

# Distributors ----------------------------------------------------------------
# Shared distributor for LENS records
lens_distributor = {
    b'DNAM': 'fade_distance_radius_scale',
    b'LFSP': {
        b'DNAM': 'lens_flare_sprites',
    },
}

# Shared distributor for PERK records
perk_distributor = {
    b'DESC': {
        b'CTDA|CIS1|CIS2': 'conditions',
        b'DATA': 'perk_trait',
    },
    b'PRKE': {
        b'CTDA|CIS1|CIS2|DATA': 'perk_effects',
    },
}
