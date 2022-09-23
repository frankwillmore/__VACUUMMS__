# RELEASE NOTES

---

## 1.1.3: Fix TIFF issues

- Large TIFF issue addressed (integer type was overflowing for a tiff sized at 4 x 1024 x 1024 x 1024, resulting in a malloc of size 0, which was successful, but useless)
- Fixed non-working FVI2TIFF. 
- Added ability to map to more than one channel in the TIFF
- Change hard-coded limits to allow for larger configurations in gfg2fvi. Extracted hard-coded values to new limits.h file.

## 1.1.2

- Now builds with apple-clang, both x86_64 and arm64
- Add examples
- Remove redundant definitions/includes that were confounding builds

## 1.1.1

- Now builds with linux, gcc for both x86_64 and arm64
- Add explicit dependency on X11
- Remove pointer-to-stack-variable bug that would *sometimes* compile and/or run OK
- Purge the malloc-wrapped-with-an-assertion that was causing release builds to segfault, while debug ran fine

## 1.1.0

- Add first pass at documentation and user guides
- Add first few working utilities

## 1.0.0 

- Introduce first pass at CMake build system generator
