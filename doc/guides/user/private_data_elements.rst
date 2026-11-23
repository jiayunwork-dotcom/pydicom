.. _private_data_elements:

Private Data Elements
=====================

.. rubric:: Accessing or creating private data elements

.. currentmodule:: pydicom.dataset

Introduction
------------

The DICOM standard allows DICOM file creators to use `private data elements`
to store information that is not defined by the DICOM standard itself.

Private data elements are stored in a :class:`~pydicom.dataset.Dataset` just
like other data elements. When reading files with *pydicom*, they will automatically be read
and available for display.  *pydicom* knows descriptive names for some
'well-known' private data elements, but for others it may not be able to
show anything except the tag and the value.

When writing your own private data elements, the DICOM standard requires the
use of 'private creator blocks'.  *pydicom* has some convenience
functions to make creating private blocks and data elements easier.

The sections below outlines accessing and creating private blocks and data
elements using *pydicom*.

Displaying Private Data Elements in *pydicom*
---------------------------------------------

Here is an example of some private tags displayed for *pydicom's* example CT
dataset::

    >>> from pydicom import examples
    >>> ds = examples.ct
    >>> ds
    Dataset.file_meta -------------------------------
    (0002, 0000) File Meta Information Group Length  UL: 192
    (0002, 0001) File Meta Information Version       OB: b'\x00\x01'
    (0002, 0002) Media Storage SOP Class UID         UI: CT Image Storage
    (0002, 0003) Media Storage SOP Instance UID      UI: 1.3.6.1.4.1.5962.1.1.1.1.1.20040119072730.12322
    (0002, 0010) Transfer Syntax UID                 UI: Explicit VR Little Endian
    (0002, 0012) Implementation Class UID            UI: 1.3.6.1.4.1.5962.2
    (0002, 0013) Implementation Version Name         SH: 'DCTOOL100'
    (0002, 0016) Source Application Entity Title     AE: 'CLUNIE1'
    -------------------------------------------------
    (0008, 0005) Specific Character Set              CS: 'ISO_IR 100'
    (0008, 0008) Image Type                          CS: ['ORIGINAL', 'PRIMARY', 'AXIAL']
    ...
    (0009, 0010) Private Creator                     LO: 'GEMS_IDEN_01'
    (0009, 1001) [Full fidelity]                     LO: 'GE_GENESIS_FF'
    (0009, 1002) [Suite id]                          SH: 'CT01'
    ...

The last two lines in the example above show *pydicom's* display of two private
data elements.  The line preceding those shows the private creator data element
that reserves a section of tag element numbers for that creator's use.

Since the descriptions for private data elements are not part of the DICOM
standard, and are thus not necessarily unique, *pydicom* does not allow you to
access data elements using those names. This is indicated by enclosing the
text in square brackets, to make it clear it is different from DICOM
standard descriptors.

You can still access the private data elements using the tag, remembering that
data elements access by tag number return a full :class:`~pydicom.dataelem.DataElement`
instance, and the `value` attribute is needed to get the value::

    >>> ds[0x00091001].value
    'GE_GENESIS_FF'

You can also create a :class:`PrivateBlock` instance and access elements
through it::

    >>> block = ds.private_block(0x0009, 'GEMS_IDEN_01')
    >>> block[0x01]
    (0009, 1001) [Full fidelity]                     LO: 'GE_GENESIS_FF'
    >>> block[0x01].value
    'GE_GENESIS_FF'

Using the private block like this is even more useful when creating your own
private data elements, as shown in the next section.

Setting Private Data Elements with *pydicom*
--------------------------------------------

The DICOM standard requires a private creator data element to identify and
reserve a section of private tags. That name should be unique, and usually
has the company name as the first part to accomplish that.  *pydicom* provides
convenience functions to manage this::

    >>> block = ds.private_block(0x000b, "My company 001", create=True)
    >>> block.add_new(0x01, "SH", "my value")
    >>> ds
    ...
    (000b, 0010) Private Creator                     LO: 'My company 001'
    (000b, 1001) Private tag data                    SH: 'my value'
    ...

Standard Python operations like ``in`` and ``del`` can also be used when working
with block object::

    >>> 0x01 in block
    True
    >>> 0x02 in block
    False
    >>> del block[0x01]
    >>> 0x01 in block
    False

Since v3.0, there's also a convenience method to add a private tag without creating a private block first::

    >>> block = ds.add_new_private("My company 001", 0x000B, 0x01, "my value", VR.SH)
    >>> ds
    ...
    (000b, 0010) Private Creator                     LO: 'My company 001'
    (000b, 1001) Private tag data                    SH: 'my value'
    ...

Note that for known private tags you don't need to provide the VR in this function.

Removing All Private Data Elements
-----------------------------------------------

One part of anonymizing a DICOM file is to ensure that private data elements
have been removed, as there is no guarantee as to what kind of information
might be contained in them.  *pydicom* provides a convenience function
:func:`Dataset.remove_private_tags` to recursively remove private elements::

    >>> ds.remove_private_tags()

This can also be helpful during interactive sessions when exploring DICOM
files, to remove a large number of lines from the display of a dataset --
lines which may not provide useful information.

Adding new entries to the DICOM dictionary
------------------------------------------
*pydicom* contains a dictionary with all known DICOM tags from the latest DICOM standard
at release time. It also contains a dictionary with a number of known private tags
collected from various sources.
Sometimes you may encounter tags unknown to *pydicom* - either tags defined in a newer version
of the standard, or, the more common case, private tags that are not contained
in the private tags dictionary.

In this case, you can add these tags to the DICOM dictionary before reading or writing datasets
containing these tags. After that, *pydicom* will correctly handle the type of these tags, and
can display their description if needed.

For standard tags, you can use :func:`~pydicom.datadict.add_dict_entry` or
:func:`~pydicom.datadict.add_dict_entries` (to add multiple tags at once)::

    >>> add_dict_entry(tag=0x888800001, VR="SH", keyword="SomeNewTag", description="Some New Tag")

For private tags, the analogous functions are
:func:`~pydicom.datadict.add_private_dict_entry` and :func:`~pydicom.datadict.add_private_dict_entries`::

    >>> add_private_dict_entry(private_creator="ACME 1.1", tag=0x004100001, VR="DA", description="Release Date")

Note that private tags do not have a keyword, as they are not registered in the
standard DICOM data dictionary. As a private tag is defined by the tuple of private creator, group ID and tag offset,
you always have to provide the private creator to define a new private tag.

An example of how to use :func:`~pydicom.datadict.add_private_dict_entries` can
be found in :ref:`this code snippet <sphx_glr_auto_examples_metadata_processing_plot_add_dict_entries.py>`.

Per-read private dictionaries
-----------------------------

:func:`~pydicom.datadict.add_private_dict_entries` modifies a process-global
dictionary, so it cannot represent two sources that reuse the same private
creator name with different meanings - for example two vendors that both use
``"ACME"`` where offset ``0x01`` is a **UL** in one archive and a **US** in
another.  For these cases :func:`~pydicom.filereader.dcmread` accepts a
keyword-only ``private_dictionary`` argument that binds a private dictionary
to that single read only::

    >>> private_dictionary = {
    ...     "ACME": {
    ...         (0x0041, 0x01): ("UL", "1", "ACME Unsigned Long", ""),
    ...     }
    ... }
    >>> ds = dcmread("acme.dcm", private_dictionary=private_dictionary)
    >>> ds[0x00411001].VR
    'UL'

The mapping is ``{creator: {(group, offset): (VR, VM, description, retired)}}``
where:

* the creator must be a non-empty :class:`str`,
* the group must be an odd :class:`int` from ``0x0001`` to ``0xFFFF``
  (``bool`` is rejected),
* the offset must be an :class:`int` from ``0`` to ``255`` (``bool`` is
  rejected),
* the VR must be a single definite DICOM VR such as ``"UL"`` or ``"LO"`` -
  ambiguous VRs like ``"US or SS"`` are not allowed,
* the VM must follow DICOM notation, e.g. ``"1"``, ``"1-n"`` or ``"2-2n"``,
* the description and retired flag must be strings,
* the whole mapping may be empty (it is then ignored) and must not contain
  callbacks or other callable objects.

Any invalid configuration raises :class:`ValueError` before the first byte of
the input is read.

Overlay priority and standard tags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A matching entry in ``private_dictionary`` takes precedence over pydicom's
built-in private dictionary.  Private tags that the overlay does not define
continue to be resolved through the built-in dictionary, and standard
(non-private) tags are never affected.  The overlay is never written to the
process-global private dictionary, so different threads may read different
sources with different definitions concurrently without any global lock.

For explicit VR transfer syntaxes the VR stored in the file always wins; the
overlay only supplies the description.  For implicit VR little endian files
the overlay VR participates in the actual byte parsing, including deciding
whether an undefined-length private element is a sequence.

Frozen interpretation, copies and pickling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The interpretation is frozen when the read starts.  Mutating the caller's
mapping afterwards, registering global private entries, or reading another
source does not change any element already returned, including its
:attr:`~pydicom.dataelem.DataElement.VR`,
:attr:`~pydicom.dataelem.DataElement.name` and
:attr:`~pydicom.dataelem.DataElement.repval`.  This also holds for elements
inside sequences (which may define their own private creators), for values
materialised later through ``defer_size`` and for a detached
:class:`~pydicom.dataelem.DataElement` taken from the dataset.

:meth:`~pydicom.dataset.Dataset.copy`, :func:`copy.deepcopy` and
:mod:`pickle` preserve the bound interpretation; a pickled dataset reloaded in
another process keeps its VR and description without any global registration.
Whether the deferred source file or buffer is still readable later remains the
caller's responsibility, as for ordinary deferred reads.

All datasets of one read - including every nested sequence item - share one
immutable binding object, so the extra memory is the size of the validated
overlay (O(K) for K entries) rather than a per-dataset copy (no O(K*N)
growth for N parsed datasets).

With ``specific_tags``
^^^^^^^^^^^^^^^^^^^^^^

Private creator elements are still scanned when ``specific_tags`` is used, so a
selected private element can be interpreted even though its creator is not in
the returned tag set.  The creator elements themselves are not added to the
result dataset.

Writing back
^^^^^^^^^^^^

No configuration entry is needed to write a dataset read this way.  Writing an
interpreted dataset to explicit VR little endian and re-reading it *without*
any overlay restores the private values and the sequence hierarchy, because the
resolved VR is written into the file.  Custom descriptions are not persisted in
the DICOM file and no configuration hook is added for newly constructed
:class:`~pydicom.dataset.Dataset` objects.
