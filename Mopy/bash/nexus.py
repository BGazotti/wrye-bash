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
#  Wrye Bash copyright (C) 2005-2009 Wrye, 2010-2022 Wrye Bash Team
#  https://github.com/wrye-bash
#  Mopy/bash/games.py copyright (C) 2016 Utumno: Original design
#
# =============================================================================
#
# Everything in this file is based on Ganda's pynxm
# (https://github.com/GandaG/pynxm). See also Mopy/LICENSE-THIRD-PARTY.
# Modifications have been made to fit Wrye Bash's code style, refactor it to
# take advantage of Python 3 features, create a higher-level API and make
# third-party dependencies optional.
#
# Original copyright notice and disclaimer:
# =============================================================================
# Copyright 2019 Daniel Nunes
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""A Python wrapper for the Nexus API. Based on pynxm.

For documentation on the API, see the following link:
https://app.swaggerhub.com/apis-docs/NexusMods/nexus-mods_public_api_params_in_form_data/1.0

Some terminology:
 - game domain: Nexus game ID (e.g. 'skyrim' for
                https://www.nexusmods.com/skyrim/mods/1840)
 - mod id: Nexus mod ID (e.g. '1840' for
           https://www.nexusmods.com/skyrim/mods/1840)
 - colour scheme: Refers to the different colours the site can have, depending
                  on what game you're modding. Note that we use the British
                  English spelling (colour) here for consistency with the API,
                  whereas all other parts of WB use the American English
                  spelling (color)"""

from __future__ import annotations

__author__ = 'Ganda, Infernio'

# First see if we even have the dependencies necessary to use the Nexus API
try:
    import requests
except ImportError as e:
    raise ImportError('requests missing, Nexus API unavailable') from e
try:
    from websocket import create_connection
except ImportError as e:
    raise ImportError('websocket-client missing, Nexus API unavailable') from e

import json
import platform
import uuid
import webbrowser
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from . import bass
from .exception import LimitReachedError, RequestError, \
    EndorsedWithoutDownloadError, EndorsedTooSoonError

_USER_AGENT = (f'WryeBash/{bass.AppVersion} ({platform.platform()}; '
               f'{platform.architecture()[0]}) '
               f'{platform.python_implementation()}/'
               f'{platform.python_version()}')
_BASE_URL = 'https://api.nexusmods.com/v1/'

# Typing-related things go here
_OptStrDict = dict[str, str] | None

# Internal API ----------------------------------------------------------------
class _Op(Enum):
    """HTTP request methods."""
    DELETE = 'DELETE'
    GET = 'GET'
    POST = 'POST'

class _JsonParsable:
    """Base class for classes that will be parsed from JSON based on their type
    annotations and a special '_parsers' class var (see below). Call
    parse_from_json to create instances."""
    # Specifies special handling for any number of attributes in the parsed
    # JSON dict. Each 'parser' is a function taking the JSON dict and the
    # attribute being parsed and returning the parsed object
    _parsers: dict = {}

    def __init__(self, **_kwargs): # To make PyCharm shut up
        super().__init__()

    @classmethod
    def parse_from_json(cls, json_dict: dict):
        if json_dict is None:
            return None # Handle optional parsables
        inst_args = {}
        for cls_attr, cls_type_str in cls.__annotations__.items():
            attr_parser = cls._parsers.get(cls_attr)
            if attr_parser is None:
                # No special parser, access JSON dict directly
                parsed_obj = json_dict[cls_attr]
            else:
                parsed_obj = attr_parser(json_dict, cls_attr)
            inst_args[cls_attr] = parsed_obj
        return cls(**inst_args)

def _remap(remap_attr: str):
    """Simple parser that uses a different attribute for accessing the JSON
    dict. Used to avoid bad names (e.g. name) and builtins (e.g. id)."""
    def _remap_func(json_dict, _cls_attr):
        return json_dict[remap_attr]
    return _remap_func

# Public API ------------------------------------------------------------------
class FileCategory(Enum):
    """The valid categories that files (referring to downloadable files, i.e.
    packages, on the Nexus) can have."""
    MAIN_FILES = 'main'
    UPDATES = 'update'
    OPTIONAL_FILES = 'optional'
    MISCELLANEOUS_FILES = 'miscellaneous'
    OLD_FILES = 'old_version'

class UpdatePeriod(Enum):
    """The valid periods for retrieving updated mods."""
    ONE_DAY = '1d'
    ONE_WEEK = '1w'
    ONE_MONTH = '1m'

@dataclass
class NxModUpdate(_JsonParsable):
    """Represents a recent update to a mod."""
    # The ID of the mod that has been updated
    mod_id: int
    # The timestamp on which the last edit to a file from the mod was made
    latest_file_update: int
    # The timestamp on which the last edit to the mod's page was made
    latest_mod_activity: int

@dataclass
class NxUploadUser(_JsonParsable):
    """Represents a user who uploaded a mod."""
    _parsers = {
        'user_name': _remap('name'),
    }
    # The ID of the user who uploaded the mod
    member_id: int
    # The ID of the group the user belongs to
    member_group_id: int
    # The screen name of the user
    user_name: str

@dataclass
class NxModEndorsement(_JsonParsable):
    """Represents a user's endorsement relative to a mod."""
    _parsers = {
        'endorse_timestamp': _remap('timestamp'),
        'endorse_version': _remap('version'),
    }
    # One of 'Undecided', 'Endorsed' or 'Abstained'
    endorse_status: str
    # The timestamp at which the mod was endorsed. None if endorse_status is
    # 'Undecided'
    endorse_timestamp: int | None
    ##: Seems to always be None, Pickysaurus said it may be unused
    endorse_version: str | None

@dataclass
class NxMod(_JsonParsable):
    _parsers = {
        'mod_display_name': _remap('name'),
        'mod_summary': _remap('summary'),
        'mod_description': _remap('description'),
        'mod_uid': _remap('uid'),
        'nx_game_id': _remap('game_id'),
        'mod_version': _remap('version'),
        'created_by': _remap('author'),
        'mod_status': _remap('status'),
        'mod_available': _remap('available'),
        'upload_user': lambda d, a: NxUploadUser.parse_from_json(d['user']),
        'endorsement': lambda d, a: NxModEndorsement.parse_from_json(d[a]),
    }
    # The user-visible name for this mod
    mod_display_name: str
    # A short description of this mod
    mod_summary: str
    # The full description of this mod, including markup (BBCode and/or HTML)
    mod_description: str = field(repr=False)
    # The URL to the main picture used for this mod
    picture_url: str
    # The number of times this mod has been downloaded
    mod_downloads: int
    # The number of times this mod has been downloaded, counting each user only
    # once
    mod_unique_downloads: int
    # A unique ID representing this mod
    mod_uid: int
    # The game-relatively unique ID of this mod
    mod_id: int
    # The internal ID of the game this mod was uploaded for
    nx_game_id: int
    # Whether or not to allow rating this mod
    allow_rating: bool
    # The Nexus domain name for the game this mod was uploaded for
    domain_name: str
    # The internal ID of the category in which this mod is filed
    category_id: int
    # The latest version of this mod
    mod_version: str
    # The number of times this mod has been endorsed
    endorsement_count: int
    # The timestamp on which this mod was created
    created_timestamp: int
    # Same as created_timestamp, but as a human-readable date
    created_time: str
    # The timestamp on which this mod was last updated
    updated_timestamp: int
    # Same as updated_timestamp, but as a human-readable date
    updated_time: str
    # The screen name of the user who authored this mod
    created_by: str
    # The screen name of the user who uploaded this mod
    uploaded_by: str
    # The URL of the Nexus profile of the user who uploaded this mod
    uploaded_users_profile_url: str
    # Whether or not this mod contains adult content
    contains_adult_content: bool
    # The current status of this mod. Usually 'published'
    mod_status: str
    # Whether or not this mod is currently available (i.e. its page can be
    # accessed, it can be downloaded, etc.)
    mod_available: bool
    # More detailed information about the user who uploaded this mod
    upload_user: NxUploadUser
    # Information about the endorsement of this mod by the current user
    endorsement: NxModEndorsement | None

@dataclass
class NxFile(_JsonParsable):
    """Represents details about a file (aka a package, aka the actual things
    you download from the Nexus)."""
    _parsers = {
        'nx_file_ids': _remap('id'),
        'nx_file_uid': _remap('uid'),
        'nx_file_id': _remap('file_id'),
        'file_display_name': _remap('name'),
        'nx_file_version': _remap('version'),
        'raw_size': _remap('size'),
        'uploaded_file_name': _remap('file_name'),
        'parent_mod_version': _remap('mod_version'),
        'file_description': _remap('description'),
        'file_md5': lambda d, a: d.get('md5'),
    }
    # A list of IDs that uniquely identify this file - seems to be file ID and
    # game ID
    nx_file_ids: list[int]
    # A unique ID representing this file
    nx_file_uid: int
    # A game-relatively unique ID for this file
    nx_file_id: int
    # The name of this file displayed in the Files tab
    file_display_name: str
    # The version of this file
    nx_file_version: str
    # The internal ID of the category that this file is in
    category_id: int
    # The name of the category that this file is in (e.g. 'MAIN')
    category_name: str
    # Whether or not this is the 'pimary' file, i.e. the one that will be
    # downloaded if you click the main 'Vortex' button on the mod page
    is_primary: bool
    # The size of this file, seems to be in kilobytes - probably better to use
    # size_kb or size_in_bytes (see below)
    raw_size: int
    # The name that this file originally had when its uploader uploaded it
    uploaded_file_name: str
    # The timestamp on which this file was uploaded
    uploaded_timestamp: int
    # Same as uploaded_timestamp, but as a human-readable date
    uploaded_time: str
    # The current version of the mod that this file belongs to
    parent_mod_version: str
    # A URL linking to a virus scan website's results for this file (currently
    # VirusTotal)
    external_virus_scan_url: str
    # The description shown for this file on the Files tab
    file_description: str
    # The size of this file in kilobytes
    size_kb: int
    # The size of this file in bytes
    size_in_bytes: int
    # The HTML that makes up the changelog entry for this file's version (shown
    # if you click on the version number next to the file's download counter).
    # May be None if the file's version has no changelog entry
    changelog_html: str | None
    # A link to a 'content preview' of the file. This is a JSON document
    # offering a tree view of the files and folders within an archive, along
    # with their sizes (it's what Nexus uses if you click on 'Preview file
    # contents' underneath the file's download button)
    content_preview_link: str
    # The MD5 hash of this file. Only present when this file is retrieved as
    # part of an MD5 search
    file_md5: str | None

@dataclass
class NxDownloadedFile(_JsonParsable):
    """Represents a file that the user has downloaded."""
    _parsers = {
        'dl_mod': lambda d, a: NxMod.parse_from_json(d['mod']),
        'dl_file': lambda d, a: NxFile.parse_from_json(d['file_details']),
    }
    # The mod from which the file was downloaded
    dl_mod: NxMod
    # Details about the downloaded file
    dl_file: NxFile

@dataclass
class NxFileUpdate(_JsonParsable):
    """Represents a single file update, i.e. a point at which a newer file has
    replaced/obsoleted an older file."""
    # File ID of the old file
    old_file_id: int
    # File ID of the new file
    new_file_id: int
    # Name of the old file (matches what the file would be named if you'd
    # downloaded it from the Nexus)
    old_file_name: str
    # Name of the new file (matches what the file would be named if you'd
    # downloaded it from the Nexus)
    new_file_name: str
    # Timestamp on which the new file was uploaded
    uploaded_timestamp: int
    # Same as uploaded_timestamp, but as a human-readable date
    uploaded_time: str

@dataclass
class NxModFiles(_JsonParsable):
    """Represents files from a mod's Files tab."""
    _parsers = {
        'files_list': lambda d, a: [NxFile.parse_from_json(e)
                                    for e in d['files']],
        'file_updates': lambda d, a: [NxFileUpdate.parse_from_json(e)
                                      for e in d[a]]
    }
    # The files on this mod's Files tab (potentially subject to filters, e.g.
    # only the Main Files)
    files_list: list[NxFile]
    # Past updates to this mod's files, i.e. metadata on which newer files are
    # updates to which older files
    file_updates: list[NxFileUpdate]

@dataclass
class NxCategory(_JsonParsable):
    """Represents a category that can be used for mods."""
    _parsers = {
        'category_name': _remap('name'),
    }
    # The internal ID used for this category
    category_id: int
    # The user-facing name of this category
    category_name: str
    # The category ID of this category's parent. May be False, which indicates
    # that the category has no parent
    parent_category: int | bool

@dataclass
class NxGame(_JsonParsable):
    """Represents a game on the Nexus."""
    _parsers = {
        'num_authors': _remap('authors'),
        'game_categories': lambda d, a: [NxCategory.parse_from_json(e)
                                         for e in d['categories']],
        'num_downloads': _remap('downloads'),
        'nx_game_id': _remap('id'),
        'game_display_name': _remap('name'),
        'num_mods': _remap('mods'),
    }
    # The date on which this game was approved by Nexus
    approved_date: int
    # The number of mod authors for this game
    num_authors: int
    # The categories available for this game
    game_categories: list[NxCategory] = field(repr=False)
    # The Nexus domain name for this game (e.g. 'skyrim')
    domain_name: str
    # The total number of times users have downloaded mods for this game
    num_downloads: int
    # The number of files available for this game
    file_count: int
    # The total number of endorsements users have left on mods for this game
    file_endorsements: int
    # The total number of times users have viewed mods for this game
    file_views: int
    # The URL for the Nexus forums section for this game
    forum_url: str
    # The genre of this game
    genre: str
    # The internal ID used for this game
    nx_game_id: int
    # The number of mods published for this game
    num_mods: int
    # The user-facing name of this game
    game_display_name: str
    # The Nexus URL of this game
    nexusmods_url: str

@dataclass
class NxUser(_JsonParsable):
    """Represents a Nexus user."""
    _parsers = {
        'user_api_key': _remap('key'),
        'user_name': _remap('name'),
        'user_email': _remap('email'),
    }
    # The ID of the user
    user_id: int
    # The API key that is currently being used by the user
    user_api_key: str = field(repr=False)
    # The screen name of the user
    user_name: str
    # Whether or not the user currently has a premium subscription
    is_premium: bool
    # Whether or not the user is a 'Supporter', i.e. had a premium subscription
    # at some point in the past
    is_supporter: bool
    # The email address used by the user
    user_email: str
    # The URL to the user's profile *picture*
    profile_url: str

@dataclass
class NxTrackedMod(_JsonParsable):
    """Represents a tracked mod on the Nexus."""
    # The game-relatively unique ID of the tracked mod
    mod_id: int
    # The Nexus domain name for this game (e.g. 'skyrim')
    domain_name: str

@dataclass
class NxEndorsement(_JsonParsable):
    """Represents a user's endorsement of a mod."""
    _parsers = {
        'endorsement_date': _remap('date'),
        'endorsement_version': _remap('version'),
        'endorsement_status': _remap('status'),
    }
    # The game-relatively unique ID of the endorsed mod
    mod_id: int
    # The Nexus domain name for this game (e.g. 'skyrim')
    domain_name: str
    # The date on which the mod was endorsed
    endorsement_date: str
    ##: Seems to always be None, Pickysaurus said it may be unused
    endorsement_version: str | None
    # Either 'Abstained' or 'Endorsed', indicating whether or not the
    # endorsement is still active
    endorsement_status: str

@dataclass
class NxColourScheme(_JsonParsable):
    """Represents a Nexus colour scheme."""
    _parsers = {
        'colour_scheme_id': _remap('id'),
        'colour_scheme_name': _remap('name'),
    }
    # The internal ID used by the colour scheme
    colour_scheme_id: int
    # The name of the colour scheme
    colour_scheme_name: str
    # The primary colour used for the colour scheme
    primary_colour: str
    # The secondary colour used for the colour scheme
    secondary_colour: str
    # The 'darker' colour used for the colour scheme
    darker_colour: str

class Nexus:
    """The main class used for connecting to the Nexus API. Requires an API key
    from your Nexus account."""
    def __init__(self, api_key: str):
        self._session = requests.Session()
        self._session.headers.update({
            'user-agent': _USER_AGENT,
            'apikey': api_key,
            'content-type': 'application/json',
            # See the Nexus docs, they request these two always be present:
            # https://help.nexusmods.com/article/114-api-acceptable-use-policy
            'application-name': 'Wrye Bash',
            'application-version': bass.AppVersion,
        })
        # Rate limiting information - set to -1 (= unknown) by default
        self._hourly_limit = -1
        self._hourly_remaining = -1
        self._daily_limit = -1
        self._daily_remaining = -1

    # Internal API ------------------------------------------------------------
    def _make_request(self, req_op: _Op, req_endpoint: str,
            req_payload: _OptStrDict = None, req_data: _OptStrDict = None,
            req_headers: _OptStrDict = None):
        """Performs an actual web request via the current session.

        :param req_op: The HTTP request method to use. See _Op.
        :param req_endpoint: The API endpoint to use. See API docs.
        :param req_payload: The parameters to send in the request's query
            string.
        :param req_data: The data to send in the body of the request.
        :param req_headers: The HTTP headers to send with the request."""
        if req_payload is None:
            req_payload = {}
        if req_data is None:
            req_data = {}
        if req_headers is None:
            req_headers = {}
        response = self._session.request(req_op.value,
            _BASE_URL + req_endpoint, params=req_payload, data=req_data,
            headers=req_headers, timeout=30)
        try:
            self._update_rate_limit_info(response.headers)
        except KeyError:
            # This will get logged below with status code and message
            pass
        status_code = response.status_code
        if status_code not in (200, 201):
            if status_code == 429:
                raise LimitReachedError()
            else:
                try:
                    msg = response.json()['message']
                except KeyError:
                    msg = response.json()['error']
                raise RequestError(status_code, msg)
        return response.json()

    def _update_rate_limit_info(self, response_headers):
        """Updates the cached rate limit information from this new response."""
        self._hourly_limit = int(response_headers['x-rl-hourly-limit'])
        self._hourly_remaining = int(response_headers['x-rl-hourly-remaining'])
        self._daily_limit = int(response_headers['x-rl-daily-limit'])
        self._daily_remaining = int(response_headers['x-rl-daily-remaining'])

    def _mod_endorse_shared(self, game_domain: str, mod_id: int,
            endpoint_file: str):
        """Shared code of mod_endorse and mod_disendorse."""
        try:
            self._make_request(_Op.POST,
                f'games/{game_domain}/mods/{mod_id}/{endpoint_file}.json')
        except RequestError as e:
            if e.status_code == 403:
                if e.orig_msg == 'NOT_DOWNLOADED_MOD':
                    raise EndorsedWithoutDownloadError()
                elif e.orig_msg == 'TOO_SOON_AFTER_DOWNLOAD':
                    raise EndorsedTooSoonError()
            raise

    # Public API --------------------------------------------------------------
    @classmethod
    def sso(cls, app_slug: str, sso_token: str, sso_id: str | None = None):
        """Application login via Single Sign-On (SSO).

        :param app_slug: A string with the application slug.
        :param sso_token: A string with the connection token.
        :param sso_id: An optional string with an id used in previous
            connections.
        :return: A 'Nexus' instance, ready to be used."""
        ws = create_connection('wss://sso.nexusmods.com')
        if sso_id is None:
            sso_id = str(uuid.uuid4())
        ws.send(json.dumps({'id': sso_id, 'token': sso_token}))
        webbrowser.open(f'https://www.nexusmods.com/sso?id={sso_id}'
                        f'&application={app_slug}')
        api_key = ws.recv()
        return cls(api_key)

    # Mods - operations on multiple mods --------------------------------------
    def mods_updated(self, game_domain: str,
            period: UpdatePeriod) -> list[NxModUpdate]:
        """Returns a list of mods that have been updated in a given period,
        with timestamps of their last update.

        :param game_domain: A string with the Nexus game domain.
        :param period: The period for which to return updated mods."""
        return [NxModUpdate.parse_from_json(d) for d in self._make_request(
            _Op.GET, f'games/{game_domain}/mods/updated.json',
            req_payload={'period': period.value})]

    def mods_latest_added(self, game_domain: str) -> list[NxMod]:
        """Retrieve the 10 latest added mods for a specified game.

        :param game_domain: A string with the Nexus game domain."""
        return [NxMod.parse_from_json(d) for d in self._make_request(
            _Op.GET, f'games/{game_domain}/mods/latest_added.json')]

    def mods_latest_updated(self, game_domain: str) -> list[NxMod]:
        """Retrieve 10 latest updated mods for a specified game.

        :param game_domain: A string with the Nexus game domain."""
        return [NxMod.parse_from_json(d) for d in self._make_request(
            _Op.GET, f'games/{game_domain}/mods/latest_updated.json')]

    def mods_trending(self, game_domain: str) -> list[NxMod]:
        """Retrieve 10 trending mods for a specified game.

        :param game_domain: A string with the Nexus game domain."""
        return [NxMod.parse_from_json(d) for d in self._make_request(
            _Op.GET, f'games/{game_domain}/mods/trending.json')]

    # Mod - operations on a single mod ----------------------------------------
    def mod_changelogs(self, game_domain: str,
            mod_id: int) -> dict[str, list[str]]:
        """Returns a dict mapping versions to lists of strings. Each string
        represent a single entry in the changelog for that version.

        Example:
            A Nexus changelog like this:
                · 1.0.0
                    > Initial release
                · 1.0.1
                    > Fixed mod deleting C: drive
                    > Fixed mod overheating CPU when pressing Enter
            Would result in this output from mod_changelogs:
            {'1.0.0': ['Initial release'],
             '1.0.1': ['Fixed mod deleting C: drive',
                       'Fixed mod overheating CPU when pressing Enter']}

        :param game_domain: A string with the Nexus game domain.
        :param mod_id: A Nexus mod id."""
        return self._make_request(_Op.GET,
            f'games/{game_domain}/mods/{mod_id}/changelogs.json')

    def mod_details(self, game_domain: str, mod_id: int) -> NxMod:
        """Retrieve specified mod details, from a specified game.

        :param game_domain: A string with the Nexus game domain.
        :param mod_id: A Nexus mod id."""
        return NxMod.parse_from_json(self._make_request(
            _Op.GET, f'games/{game_domain}/mods/{mod_id}.json'))

    def mod_search(self, game_domain: str,
            md5_hash: str) -> list[NxDownloadedFile]:
        """Searches for a mod given its md5 hash.

        :param game_domain: A string with the Nexus game domain.
        :param md5_hash: The mod's md5 hash - this is the hash of the package
           downloaded from the Nexus. Must be in hexadecimal form, so use
           hashlib.md5.hexdigest() or something similar."""
        # lower() because letters (A-F) must be lowercased, otherwise the Nexus
        # API will fail to find the mod in question
        return [NxDownloadedFile.parse_from_json(d) for d in
                self._make_request(_Op.GET,
                    f'games/{game_domain}/mods/md5_search/'
                    f'{md5_hash.lower()}.json')]

    def mod_endorse(self, game_domain: str, mod_id: int):
        """Endorse a mod. May fail if a mod is endorsed without having been
        downloaded, in which case an EndorsedWithoutDownloadError is raised, or
        if a mod is endorsed too quickly after having been downloaded, in which
        case an EndorsedTooSoonError is raised.

        :param game_domain: A string with the Nexus game domain.
        :param mod_id: A Nexus mod id."""
        self._mod_endorse_shared(game_domain, mod_id, 'endorse')

    def mod_disendorse(self, game_domain: str, mod_id: int):
        """'Abstain' from endorsing a mod - aka disendorse it. May fail with
        the same errors under the same conditions as mod_endorse.

        :param game_domain: A string with the Nexus game domain.
        :param mod_id: A Nexus mod id."""
        self._mod_endorse_shared(game_domain, mod_id, 'abstain')

    # Mod Files - operations on mod files (that's referring to the 'Files'
    # tab, so the files in the main, optional, etc. sections) -----------------
    def mod_files_list(self, game_domain: str, mod_id: int,
            categories: Iterable[FileCategory] | None = None) -> NxModFiles:
        """Lists all files for a specific mod.

        :param game_domain: A string with the Nexus game domain.
        :param mod_id: A Nexus mod id.
        :param categories: Filters returned files by one or more categories. If
            this is not None, return only files in one of the specified
            categories."""
        if categories is None:
            mf_payload = None
        else:
            mf_payload = {'category': ','.join(c.value for c in categories)}
        return NxModFiles.parse_from_json(self._make_request(
            _Op.GET, f'games/{game_domain}/mods/{mod_id}/files.json',
            req_payload=mf_payload))

    def mod_files_details(self, game_domain: str, mod_id: int,
            file_id: int) -> NxFile:
        """Return details for a specified mod file, using a specified game.

        :param game_domain: A string with the Nexus game domain.
        :param mod_id: A Nexus mod id.
        :param file_id: A string with the file id."""
        return NxFile.parse_from_json(self._make_request(
            _Op.GET, f'games/{game_domain}/mods/{mod_id}/files/'
                     f'{file_id}.json'))

    def mod_files_generate_link(self, game_domain: str, mod_id: int,
            file_id: int, nxm_key: str | None = None,
            expiry: str | None = None): # FIXME wrap
        """Generate a download link for a mod file.

        Note: Non-premium users must visit the website and provide the nxm key
        and expiry created by the website in order for this to work. Premium
        users do not need to provide them.

        :param game_domain: A string with the Nexus game domain.
        :param mod_id: A Nexus mod id.
        :param file_id: A string with the file id.
        :param nxm_key: A string with the nxm key provided by the website.
        :param expiry: A string with the expiry of the nxm key."""
        if None in (nxm_key, expiry):
            mfdl_payload = None
        else:
            mfdl_payload = {'key': nxm_key, 'expires': expiry}
        return self._make_request(_Op.GET,
            f'games/{game_domain}/mods/{mod_id}/files/{file_id}/'
            f'download_link.json', req_payload=mfdl_payload)

    # Games - operations on multiple games ------------------------------------
    def games_list(self, include_unapproved: bool = False) -> list[NxGame]:
        """Returns details for all games supported by the Nexus.

        :param include_unapproved: A boolean on whether to include unapproved
            games."""
        ia_string = 'true' if include_unapproved else 'false'
        req_raw = self._make_request(_Op.GET, 'games.json',
            req_payload={'include_unapproved': ia_string})
        return [NxGame.parse_from_json(d) for d in req_raw]

    # Game - operations on a single game --------------------------------------
    def game_details(self, game_domain: str) -> NxGame:
        """Returns details for the specified game.

        :param game_domain: A string with the Nexus game domain."""
        return NxGame.parse_from_json(self._make_request(
            _Op.GET, f'games/{game_domain}.json'))

    # User - operations on the currently logged in user -----------------------
    def user_details(self) -> NxUser:
        """Checks that an API key is valid and returns the user's details."""
        return NxUser.parse_from_json(self._make_request(
            _Op.GET, 'users/validate.json'))

    def user_tracked_mods(self) -> list[NxTrackedMod]:
        """Returns a list of all the mods being tracked by the current user."""
        return [NxTrackedMod.parse_from_json(d) for d in self._make_request(
            _Op.GET, 'user/tracked_mods.json')]

    def user_track_mod(self, game_domain: str, mod_id: int):
        """Tracks this mod with the current user.

        :param game_domain: A string with the Nexus game domain.
        :param mod_id: A Nexus mod id."""
        self._make_request(_Op.POST, 'user/tracked_mods.json',
            req_payload={'domain_name': game_domain},
            req_data={'mod_id': str(mod_id)},
            req_headers={'content-type': 'application/x-www-form-urlencoded'})

    def user_untrack_mod(self, game_domain: str, mod_id: int):
        """Stop tracking this mod with the current user.

        :param game_domain: A string with the Nexus game domain.
        :param mod_id: A Nexus mod id."""
        self._make_request(_Op.DELETE, 'user/tracked_mods.json',
            req_payload={'domain_name': game_domain},
            req_data={'mod_id': str(mod_id)},
            req_headers={'content-type': 'application/x-www-form-urlencoded'})

    def user_endorsements(self) -> list[NxEndorsement]:
        """Returns a list of all endorsements for the current user."""
        return [NxEndorsement.parse_from_json(d) for d in self._make_request(
            _Op.GET, 'user/endorsements.json')]

    # Colour Schemes ----------------------------------------------------------
    def colour_schemes(self) -> list[NxColourScheme]:
        """Returns a list of all colour schemes, including the primary,
        secondary and 'darker' colours."""
        return [NxColourScheme.parse_from_json(d) for d in self._make_request(
            _Op.GET, 'colourschemes.json')]
