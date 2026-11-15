# Third-party notices

The [MIT license](LICENSE) covers PhotoCoach source. Dependencies retain their own licenses. Source installations obtain them separately; the Win64 release ZIP includes their runtime binaries and license texts under licenses/.

| Dependency | Purpose | License reference |
| --- | --- | --- |
| NumPy | Image arrays and processing | [NumPy license](https://numpy.org/doc/stable/license.html) |
| rawpy | Python RAW decoding interface | [rawpy license](https://github.com/letmaik/rawpy/blob/main/LICENSE) |
| LibRaw | Native RAW decoding used by rawpy | [LibRaw licensing](https://www.libraw.org/about#licensing) |
| PySide6 / Qt | Desktop interface | [Qt for Python licenses](https://doc.qt.io/qtforpython-6/licenses.html) |
| Pillow | Image conversion, JPEG, and ICC profiles | [Pillow license](https://pillow.readthedocs.io/en/stable/about.html#license) |
| tifffile | TIFF output | [tifffile license](https://github.com/cgohlke/tifffile/blob/master/LICENSE) |
| pytest | Test runner | [pytest license](https://docs.pytest.org/en/stable/license.html) |

The portable ZIP includes Python, PySide6/Qt and shiboken6, rawpy/LibRaw, NumPy, Pillow and tifffile. Qt/shiboken libraries remain separate DLLs so compatible modified versions can replace them. The bundle includes component copyright/license texts and source links in THIRD_PARTY_NOTICES.txt. Installing from source does not change dependency licenses.

## Demonstration photograph

The editor screenshot uses **Glacier National Park Scenic**, credited to **George Gentry / USFWS**, created August 25, 2009. The U.S. Fish and Wildlife Service marks the photograph **Public Domain** on its [source page](https://www.fws.gov/media/glacier-national-park-scenic).

The image is used as a JPEG preview fixture in the actual PhotoCoach interface. It is not a RAW camera sample. No endorsement by the photographer or the U.S. Fish and Wildlife Service is implied.
