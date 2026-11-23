# Copyright 2008-2018 pydicom authors. See LICENSE file for details.
"""Access dicom dictionary information"""

import re
from contextvars import ContextVar
from collections.abc import Mapping

# the actual dict of {tag: (VR, VM, name, is_retired, keyword), ...}
# those with tags like "(50xx,0005)"
from pydicom._dicom_dict import DicomDictionary, RepeatersDictionary
from pydicom.misc import warn_and_log
from pydicom._private_dict import private_dictionaries
from pydicom.tag import Tag, BaseTag, TagType
from pydicom.valuerep import VR as VR_


# Generate mask dict for checking repeating groups etc.
# Map a true bitwise mask to the DICOM mask with "x"'s in it.
masks: dict[str, tuple[int, int]] = {}
for mask_x in RepeatersDictionary:
    # mask1 is XOR'd to see that all non-"x" bits
    # are identical (XOR result = 0 if bits same)
    # then AND those out with 0 bits at the "x"
    # ("we don't care") location using mask2
    mask1 = int(mask_x.replace("x", "0"), 16)
    mask2 = int("".join(["F0"[c == "x"] for c in mask_x]), 16)
    masks[mask_x] = (mask1, mask2)


def mask_match(tag: int) -> str | None:
    """Return the repeaters tag mask for `tag`.

    Parameters
    ----------
    tag : int
        The tag to check.

    Returns
    -------
    str or None
        If the tag is in the repeaters dictionary then returns the
        corresponding masked tag, otherwise returns ``None``.
    """
    for mask_x, (mask1, mask2) in masks.items():
        if (tag ^ mask1) & mask2 == 0:
            return mask_x
    return None


def add_dict_entry(
    tag: int,
    VR: str,
    keyword: str,
    description: str,
    VM: str = "1",
    is_retired: str = "",
) -> None:
    """Update the DICOM dictionary with a new non-private entry.

    Parameters
    ----------
    tag : int
        The tag number for the new dictionary entry.
    VR : str
        DICOM value representation.
    description : str
        The descriptive name used in printing the entry. Often the same as the
        keyword, but with spaces between words.
    VM : str, optional
        DICOM value multiplicity. If not specified, then ``'1'`` is used.
    is_retired : str, optional
        Usually leave as blank string (default). Set to ``'Retired'`` if is a
        retired data element.

    Raises
    ------
    ValueError
        If the tag is a private tag.

    Notes
    -----
    Does not permanently update the dictionary, but only during run-time.
    Will replace an existing entry if the tag already exists in the dictionary.

    See Also
    --------
    pydicom.examples.add_dict_entry
        Example file which shows how to use this function
    add_dict_entries
        Update multiple values at once.

    Examples
    --------

    >>> from pydicom import Dataset
    >>> add_dict_entry(0x10021001, "UL", "TestOne", "Test One")
    >>> add_dict_entry(0x10021002, "DS", "TestTwo", "Test Two", VM='3')
    >>> ds = Dataset()
    >>> ds.TestOne = 'test'
    >>> ds.TestTwo = ['1', '2', '3']

    """
    add_dict_entries({tag: (VR, VM, description, is_retired, keyword)})


def add_dict_entries(
    new_entries_dict: dict[int, tuple[str, str, str, str, str]]
) -> None:
    """Update the DICOM dictionary with new non-private entries.

    Parameters
    ----------
    new_entries_dict : dict
        :class:`dict` of form:
        ``{tag: (VR, VM, description, is_retired, keyword), ...}``
        where parameters are as described in :func:`add_dict_entry`.

    Raises
    ------
    ValueError
        If one of the entries is a private tag.

    See Also
    --------
    add_dict_entry
        Add a single entry to the dictionary.

    Examples
    --------

    >>> from pydicom import Dataset
    >>> new_dict_items = {
    ...        0x10021001: ('UL', '1', "Test One", '', 'TestOne'),
    ...        0x10021002: ('DS', '3', "Test Two", '', 'TestTwo'),
    ... }
    >>> add_dict_entries(new_dict_items)
    >>> ds = Dataset()
    >>> ds.TestOne = 'test'
    >>> ds.TestTwo = ['1', '2', '3']

    """

    if any([BaseTag(tag).is_private for tag in new_entries_dict]):
        raise ValueError(
            'Private tags cannot be added using "add_dict_entries" - '
            'use "add_private_dict_entries" instead'
        )

    # Update the dictionary itself
    DicomDictionary.update(new_entries_dict)

    # Update the reverse mapping from name to tag
    keyword_dict.update({val[4]: tag for tag, val in new_entries_dict.items()})


def add_private_dict_entry(
    private_creator: str, tag: int, VR: str, description: str, VM: str = "1"
) -> None:
    """Update the private DICOM dictionary with a new entry.

    Parameters
    ----------
    private_creator : str
        The private creator for the new entry.
    tag : int
        The tag number for the new dictionary entry. Note that the
        2 high bytes of the element part of the tag are ignored.
    VR : str
        DICOM value representation.
    description : str
        The descriptive name used in printing the entry.
    VM : str, optional
        DICOM value multiplicity. If not specified, then ``'1'`` is used.

    Raises
    ------
    ValueError
        If the tag is a non-private tag.

    Notes
    -----
    Behaves like :func:`add_dict_entry`, only for a private tag entry.

    See Also
    --------
    add_private_dict_entries
        Add or update multiple entries at once.
    """
    new_dict_val = (VR, VM, description, "")
    add_private_dict_entries(private_creator, {tag: new_dict_val})


def add_private_dict_entries(
    private_creator: str, new_entries_dict: dict[int, tuple[str, str, str, str]]
) -> None:
    """Update pydicom's private DICOM tag dictionary with new entries.

    Parameters
    ----------
    private_creator: str
        The private creator for all entries in `new_entries_dict`.
    new_entries_dict : dict
        :class:`dict` of form ``{tag: (VR, VM, description, is_retired), ...}``
        where parameters are as described in :func:`add_private_dict_entry`.

    Raises
    ------
    ValueError
        If one of the entries is a non-private tag.

    See Also
    --------
    add_private_dict_entry
        Function to add a single entry to the private tag dictionary.

    Examples
    --------
    >>> new_dict_items = {
    ...        0x00410001: ('UL', '1', "Test One"),
    ...        0x00410002: ('DS', '3', "Test Two", '3'),
    ... }
    >>> add_private_dict_entries("ACME LTD 1.2", new_dict_items)
    >>> add_private_dict_entry("ACME LTD 1.3", 0x00410001, "US", "Test Three")
    """

    if not all([BaseTag(tag).is_private for tag in new_entries_dict]):
        raise ValueError(
            "Non-private tags cannot be added using "
            "'add_private_dict_entries()' - use 'add_dict_entries()' instead"
        )

    new_entries = {
        f"{tag >> 16:04X}xx{tag & 0xff:02X}": value
        for tag, value in new_entries_dict.items()
    }
    private_dictionaries.setdefault(private_creator, {}).update(new_entries)


def get_entry(tag: TagType) -> tuple[str, str, str, str, str]:
    """Return an entry from the DICOM dictionary as a tuple.

    If the `tag` is not in the main DICOM dictionary, then the repeating
    group dictionary will also be checked.

    Parameters
    ----------
    tag : int or str or Tuple[int, int]
        The tag for the element whose entry is to be retrieved, in any of the
        forms accepted by :func:`~pydicom.tag.Tag`. Only entries in the
        official DICOM dictionary will be checked, not entries in the
        private dictionary.

    Returns
    -------
    tuple of str
        The (VR, VM, name, is_retired, keyword) from the DICOM dictionary.

    Raises
    ------
    KeyError
        If the tag is not present in the DICOM data dictionary.

    See Also
    --------
    get_private_entry
        Return an entry from the private dictionary.
    """
    # Note: tried the lookup with 'if tag in DicomDictionary'
    # and with DicomDictionary.get, instead of try/except
    # Try/except was fastest using timeit if tag is valid (usual case)
    # My test had 5.2 usec vs 8.2 for 'contains' test, vs 5.32 for dict.get
    if not isinstance(tag, BaseTag):
        tag = Tag(tag)
    try:
        return DicomDictionary[tag]
    except KeyError:
        if not tag.is_private:
            mask_x = mask_match(tag)
            if mask_x:
                return RepeatersDictionary[mask_x]
        raise KeyError(f"Tag {tag} not found in DICOM dictionary")


def dictionary_is_retired(tag: TagType) -> bool:
    """Return ``True`` if the element corresponding to `tag` is retired.

    Only performs the lookup for official DICOM elements.

    Parameters
    ----------
    tag : int or str or Tuple[int, int]
        The tag for the element whose retirement status is being checked, in
        any of the forms accepted by :func:`~pydicom.tag.Tag`.

    Returns
    -------
    bool
        ``True`` if the element's retirement status is 'Retired', ``False``
        otherwise.

    Raises
    ------
    KeyError
        If the tag is not present in the DICOM data dictionary.
    """
    return "retired" in get_entry(tag)[3].lower()


def dictionary_VR(tag: TagType) -> str:
    """Return the VR of the element corresponding to `tag`.

    Only performs the lookup for official DICOM elements.

    Parameters
    ----------
    tag : int or str or Tuple[int, int]
        The tag for the element whose value representation (VR) is being
        retrieved, in any of the forms accepted by :func:`~pydicom.tag.Tag`.

    Returns
    -------
    str
        The VR of the corresponding element.

    Raises
    ------
    KeyError
        If the tag is not present in the DICOM data dictionary.
    """
    return get_entry(tag)[0]


def _dictionary_vr_fast(tag: int) -> str:
    """Return the VR corresponding to `tag`"""
    # Faster implementation of `dictionary_VR`
    try:
        return DicomDictionary[tag][0]
    except KeyError:
        if not (tag >> 16) % 2 == 1:
            mask_x = mask_match(tag)
            if mask_x:
                return RepeatersDictionary[mask_x][0]

        raise KeyError(f"Tag {Tag(tag)} not found in DICOM dictionary")


def dictionary_VM(tag: TagType) -> str:
    """Return the VM of the element corresponding to `tag`.

    Only performs the lookup for official DICOM elements.

    Parameters
    ----------
    tag : int or str or Tuple[int, int]
        The tag for the element whose value multiplicity (VM) is being
        retrieved, in any of the forms accepted by :func:`~pydicom.tag.Tag`.

    Returns
    -------
    str
        The VM of the corresponding element.

    Raises
    ------
    KeyError
        If the tag is not present in the DICOM data dictionary.
    """
    return get_entry(tag)[1]


def dictionary_description(tag: TagType) -> str:
    """Return the description of the element corresponding to `tag`.

    Only performs the lookup for official DICOM elements.

    Parameters
    ----------
    tag : int or str or Tuple[int, int]
        The tag for the element whose description is being retrieved, in any
        of the forms accepted by :func:`~pydicom.tag.Tag`.

    Returns
    -------
    str
        The description of the corresponding element.

    Raises
    ------
    KeyError
        If the tag is not present in the DICOM data dictionary.
    """
    return get_entry(tag)[2]


def dictionary_keyword(tag: TagType) -> str:
    """Return the keyword of the element corresponding to `tag`.

    Only performs the lookup for official DICOM elements.

    Parameters
    ----------
    tag : int or str or Tuple[int, int]
        The tag for the element whose keyword is being retrieved, in any of
        the forms accepted by :func:`~pydicom.tag.Tag`.

    Returns
    -------
    str
        The keyword of the corresponding element.

    Raises
    ------
    KeyError
        If the tag is not present in the DICOM data dictionary.
    """
    return get_entry(tag)[4]


def dictionary_has_tag(tag: TagType) -> bool:
    """Return ``True`` if `tag` is in the official DICOM data dictionary.

    Parameters
    ----------
    tag : int or str or Tuple[int, int]
        The tag to check, in any of the forms accepted by
        :func:`~pydicom.tag.Tag`.

    Returns
    -------
    bool
        ``True`` if the tag corresponds to an element present in the official
        DICOM data dictionary, ``False`` otherwise.
    """
    try:
        return Tag(tag) in DicomDictionary
    except Exception:
        return False


def keyword_for_tag(tag: TagType) -> str:
    """Return the keyword of the element corresponding to `tag`.

    Parameters
    ----------
    tag : int or str or Tuple[int, int]
        The tag for the element whose keyword is being retrieved, in any of
        the forms accepted by :func:`~pydicom.tag.Tag`.

    Returns
    -------
    str
        If the element is in the DICOM data dictionary then returns the
        corresponding element's keyword, otherwise returns ``''``. For
        group length elements will always return ``'GroupLength'``.
    """
    try:
        return dictionary_keyword(tag)
    except KeyError:
        return ""


# Provide for the 'reverse' lookup. Given the keyword, what is the tag?
keyword_dict: dict[str, int] = {dictionary_keyword(tag): tag for tag in DicomDictionary}


def tag_for_keyword(keyword: str) -> int | None:
    """Return the tag of the element corresponding to `keyword`.

    Only performs the lookup for official DICOM elements.

    Parameters
    ----------
    keyword : str
        The keyword for the element whose tag is being retrieved.

    Returns
    -------
    int or None
        If the element is in the DICOM data dictionary then returns the
        corresponding element's tag, otherwise returns ``None``.
    """
    return keyword_dict.get(keyword)


def repeater_has_tag(tag: int) -> bool:
    """Return ``True`` if `tag` is in the DICOM repeaters data dictionary.

    Parameters
    ----------
    tag : int
        The tag to check.

    Returns
    -------
    bool
        ``True`` if the tag is a non-private element tag present in the
        official DICOM repeaters data dictionary, ``False`` otherwise.
    """
    return mask_match(tag) in RepeatersDictionary


REPEATER_KEYWORDS = [val[4] for val in RepeatersDictionary.values()]


def repeater_has_keyword(keyword: str) -> bool:
    """Return ``True`` if `keyword` is in the DICOM repeaters data dictionary.

    Parameters
    ----------
    keyword : str
        The keyword to check.

    Returns
    -------
    bool
        ``True`` if the keyword corresponding to an element present in the
        official DICOM repeaters data dictionary, ``False`` otherwise.
    """
    return keyword in REPEATER_KEYWORDS


# PRIVATE DICTIONARY handling
# functions in analogy with those of main DICOM dict
def get_private_entry(tag: TagType, private_creator: str) -> tuple[str, str, str, str]:
    """Return an entry from the private dictionary corresponding to `tag`.

    Parameters
    ----------
    tag : int or str or Tuple[int, int]
        The tag for the element whose entry is to be retrieved, in any of the
        forms accepted by :func:`~pydicom.tag.Tag`. Only entries in the
        private dictionary will be checked.
    private_creator : str
        The name of the private creator.

    Returns
    -------
    tuple of str
        The (VR, VM, name, is_retired) from the private dictionary.

    Raises
    ------
    KeyError
        If the tag or private creator is not present in the private dictionary.

    See Also
    --------
    get_entry
        Return an entry from the DICOM data dictionary.
    """
    if not isinstance(tag, BaseTag):
        tag = Tag(tag)

    try:
        private_dict = private_dictionaries[private_creator]
    except KeyError as exc:
        raise KeyError(
            f"Private creator '{private_creator}' not in the private dictionary"
        ) from exc
    except TypeError as exc:
        msg = (
            f"{tag.private_creator} '{private_creator}' "
            f"is not a valid private creator"
        )
        warn_and_log(msg)
        raise KeyError(msg) from exc

    # private elements are usually agnostic for
    # "block" (see PS3.5-2008 7.8.1 p44)
    # Some elements in _private_dict are explicit;
    # most have "xx" for high-byte of element
    #  so here put in the "xx" in the block position for key to look up
    group_str = f"{tag.group:04X}"
    elem_str = f"{tag.elem:04X}"
    keys = [
        f"{group_str}{elem_str}",
        f"{group_str}xx{elem_str[-2:]}",
        f"{group_str[:2]}xxxx{elem_str[-2:]}",
    ]
    keys = [k for k in keys if k in private_dict]
    if not keys:
        raise KeyError(
            f"Tag '{tag}' not in private dictionary "
            f"for private creator '{private_creator}'"
        )
    dict_entry = private_dict[keys[0]]

    return dict_entry


def private_dictionary_VR(tag: TagType, private_creator: str) -> str:
    """Return the VR of the private element corresponding to `tag`.

    Parameters
    ----------
    tag : int or str or Tuple[int, int]
        The tag for the element whose value representation (VR) is being
        retrieved, in any of the forms accepted by :func:`~pydicom.tag.Tag`.
    private_creator : str
        The name of the private creator.

    Returns
    -------
    str
        The VR of the corresponding element.

    Raises
    ------
    KeyError
        If the tag is not present in the private dictionary.
    """
    return get_private_entry(tag, private_creator)[0]


def private_dictionary_VM(tag: TagType, private_creator: str) -> str:
    """Return the VM of the private element corresponding to `tag`.

    Parameters
    ----------
    tag : int or str or Tuple[int, int]
        The tag for the element whose value multiplicity (VM) is being
        retrieved, in any of the forms accepted by :func:`~pydicom.tag.Tag`.
    private_creator : str
        The name of the private creator.

    Returns
    -------
    str
        The VM of the corresponding element.

    Raises
    ------
    KeyError
        If the tag is not present in the private dictionary.
    """
    return get_private_entry(tag, private_creator)[1]


def private_dictionary_description(tag: TagType, private_creator: str) -> str:
    """Return the description of the private element corresponding to `tag`.

    Parameters
    ----------
    tag : int or str or Tuple[int, int]
        The tag for the element whose description is being retrieved, in any
        of the forms accepted by :func:`~pydicom.tag.Tag`.
    private_creator : str
        The name of the private creator.

    Returns
    -------
    str
        The description of the corresponding element.

    Raises
    ------
    KeyError
        If the tag is not present in the private dictionary,
        or if the private creator is not valid.
    """
    return get_private_entry(tag, private_creator)[2]


# ---------------------------------------------------------------------------
# Per-read private dictionary bindings
#
# A PrivateDictionaryBinding is an immutable, per-read overlay of the global
# private dictionary.  Bindings are *not* merged into ``private_dictionaries``:
# the same process (and even the same thread) may use different definitions
# for the same private creator, while datasets returned by an earlier read
# keep interpreting their private elements through the binding they were read
# with.  A single binding object is shared by every dataset of one read
# (including all nested sequence items), so the additional storage is O(K)
# for the whole read instead of O(K*N) for N parsed datasets.
# ---------------------------------------------------------------------------

# The single, definite VRs of DICOM Standard Part 5, Table 6.2-1.
# Ambiguous VRs such as "US or SS" are deliberately not accepted.
_DEFINITE_VRS = frozenset(str(vr) for vr in VR_ if "or" not in str(vr))

# DICOM VM notation (PS3.5, 6.4): a positive integer multiplicity, an open or
# closed range of positive multiplicities, or an even range such as "2-2n".
_VM_REGEX = re.compile(r"^[1-9]\d*(?:-(?:[1-9]\d*n?|n))?$")

# Type of a validated private dictionary entry: (VR, VM, description, retired)
PrivateDictEntry = tuple[str, str, str, str]


def _validate_private_dictionary(
    private_dictionary: Mapping[str, Mapping[tuple[int, int], tuple]],
) -> dict[str, dict[tuple[int, int], PrivateDictEntry]]:
    """Validate a caller supplied `private_dictionary` and return a frozen copy.

    The copy is built from plain builtin containers (``dict`` and ``tuple``),
    so it does not keep any caller object - including callables - alive and is
    safe to share between threads and to pickle.

    Raises
    ------
    ValueError
        If any part of the configuration is invalid.  Everything that is wrong
        with the configuration is reported as :class:`ValueError`, never as
        :class:`TypeError` or :class:`KeyError`.
    """
    if not isinstance(private_dictionary, Mapping):
        raise ValueError(
            "'private_dictionary' must be a mapping of private creator names "
            "to private dictionary mappings"
        )

    frozen: dict[str, dict[tuple[int, int], PrivateDictEntry]] = {}
    for creator, entries in private_dictionary.items():
        # bool is a subclass of int, so reject it explicitly
        if not isinstance(creator, str) or isinstance(creator, bool) or not creator:
            raise ValueError(
                "Private creator names must be non-empty strings, got "
                f"{creator!r} of type {type(creator).__name__}"
            )

        if not isinstance(entries, Mapping):
            raise ValueError(
                f"The entries for private creator {creator!r} must be a mapping "
                "of (group, offset) keys to (VR, VM, description, retired) "
                f"tuples, got {type(entries).__name__}"
            )

        creator_dict: dict[tuple[int, int], PrivateDictEntry] = {}
        for key, entry in entries.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or isinstance(key, bool)
                or not all(
                    isinstance(part, int) and not isinstance(part, bool)
                    for part in key
                )
            ):
                raise ValueError(
                    f"Private dictionary keys for creator {creator!r} must be "
                    f"(group, offset) integer tuples, got {key!r}"
                )

            group, offset = key
            if not 0x0001 <= group <= 0xFFFF or group % 2 == 0:
                raise ValueError(
                    f"Private group for creator {creator!r} and key {key!r} must "
                    "be an odd integer between 0x0001 and 0xFFFF, got "
                    f"{group:#06x}"
                )

            if not 0 <= offset <= 255:
                raise ValueError(
                    f"Private element offset for creator {creator!r} and key "
                    f"{key!r} must be an integer between 0 and 255, got {offset}"
                )

            if (
                not isinstance(entry, tuple)
                or len(entry) != 4
                or any(callable(part) for part in entry)
            ):
                raise ValueError(
                    f"Private dictionary entry for {creator!r} / {key!r} must "
                    "be a (VR, VM, description, retired) tuple without callables"
                )

            vr, vm, description, retired = entry
            if not isinstance(vr, str) or isinstance(vr, bool) or vr not in _DEFINITE_VRS:
                raise ValueError(
                    f"Invalid VR {vr!r} for {creator!r} / {key!r}: must be a "
                    "single definite DICOM VR (e.g. 'UL', 'US' or 'LO'); "
                    "ambiguous VRs such as 'US or SS' are not allowed"
                )

            if not isinstance(vm, str) or isinstance(vm, bool) or not _VM_REGEX.match(vm):
                raise ValueError(
                    f"Invalid VM {vm!r} for {creator!r} / {key!r}: must follow "
                    "DICOM VM notation such as '1', '1-n', '2-2n' or '1-3'"
                )

            if not isinstance(description, str) or isinstance(description, bool):
                raise ValueError(
                    f"Invalid description for {creator!r} / {key!r}: must be a "
                    f"string, got {type(description).__name__}"
                )

            if not isinstance(retired, str) or isinstance(retired, bool):
                raise ValueError(
                    f"Invalid retired flag for {creator!r} / {key!r}: must be a "
                    f"string, got {type(retired).__name__}"
                )

            creator_dict[(group, offset)] = (vr, vm, description, retired)

        frozen[creator] = creator_dict

    return frozen


class PrivateDictionaryBinding:
    """An immutable private dictionary overlay bound to a single read.

    Instances are created by :func:`make_private_dictionary_binding` and shared
    by every :class:`~pydicom.dataset.Dataset` (and nested sequence item
    dataset) produced by the corresponding
    :func:`~pydicom.filereader.dcmread` call.

    The overlay is checked first.  Tags it does not define continue to be
    resolved through the built-in global private dictionary, and standard
    (non-private) tags are unaffected.  The object is frozen at read start:
    later changes to the caller mapping, to the global private dictionary or
    to private creator registrations do not alter any interpretation handed
    out through this binding.

    Attributes
    ----------
    overlay : dict
        The validated, frozen mapping of private creator name to a mapping of
        ``(group, offset)`` to ``(VR, VM, description, retired)`` entries.
    """

    __slots__ = ("overlay",)

    def __init__(
        self, overlay: dict[str, dict[tuple[int, int], PrivateDictEntry]]
    ) -> None:
        self.overlay = overlay

    def lookup(
        self, tag: BaseTag | int, private_creator: str
    ) -> PrivateDictEntry | None:
        """Return the overlay entry for `tag`, or ``None`` if not defined.

        Private dictionary entries are block agnostic (see PS3.5 7.8.1): only
        the private group and the low byte of the element number are used.
        """
        if not isinstance(tag, BaseTag):
            tag = Tag(tag)

        creator_entries = self.overlay.get(private_creator)
        if creator_entries is None:
            return None

        return creator_entries.get((tag.group, tag.element & 0xFF))

    def get_private_entry(
        self, tag: BaseTag | int, private_creator: str
    ) -> PrivateDictEntry | None:
        """Return the overlay entry, falling back to the built-in dictionary.

        ``None`` is returned if neither the overlay nor the built-in private
        dictionary define `tag` for `private_creator`.
        """
        entry = self.lookup(tag, private_creator)
        if entry is not None:
            return entry

        try:
            return get_private_entry(tag, private_creator)
        except KeyError:
            return None

    def private_VR(self, tag: BaseTag | int, private_creator: str) -> str | None:
        """Return the VR for `tag`, or ``None`` if nothing defines it."""
        entry = self.get_private_entry(tag, private_creator)
        return entry[0] if entry is not None else None

    def private_description(
        self, tag: BaseTag | int, private_creator: str
    ) -> str | None:
        """Return the description for `tag`, or ``None`` if undefined."""
        entry = self.get_private_entry(tag, private_creator)
        return entry[2] if entry is not None else None


def make_private_dictionary_binding(
    private_dictionary: Mapping[str, Mapping[tuple[int, int], tuple]] | None,
) -> PrivateDictionaryBinding | None:
    """Validate `private_dictionary` and return the binding for one read.

    Parameters
    ----------
    private_dictionary : mapping, optional
        A mapping of private creator name to mappings of ``(group, offset)``
        keys to ``(VR, VM, description, retired)`` tuples, as accepted by
        :func:`~pydicom.filereader.dcmread`.  ``None`` or an empty mapping
        result in ``None``, i.e. a read without a private dictionary overlay.

    Returns
    -------
    PrivateDictionaryBinding or None
        The immutable binding for the read, or ``None`` if no overlay was
        supplied.

    Raises
    ------
    ValueError
        If any part of `private_dictionary` is invalid.
    """
    if private_dictionary is None:
        return None

    try:
        frozen = _validate_private_dictionary(private_dictionary)
    except ValueError:
        raise
    except Exception as exc:
        # A malformed configuration must fail uniformly as ValueError, never
        # leak a container type error from the caller's objects
        raise ValueError(f"Invalid private_dictionary configuration: {exc}") from exc

    if not frozen:
        return None

    return PrivateDictionaryBinding(frozen)


# The binding of the dataset currently being parsed or converted.  Set for the
# duration of a read and of every raw element conversion triggered through a
# bound dataset.  Using a ContextVar (instead of a plain module global or a
# lock) lets threads use different bindings concurrently without serialising
# reads, and lets nested sequence parsing inherit the binding without passing
# a copy of the whole configuration through every sequence item.
_active_private_binding: ContextVar[PrivateDictionaryBinding | None] = ContextVar(
    "pydicom_active_private_binding", default=None
)


def get_active_private_binding() -> PrivateDictionaryBinding | None:
    """Return the private dictionary binding of the current read context."""
    return _active_private_binding.get()
