# Minimal ASGST fork

Idea:
- implement a minimal version of JEP Candidate 435, with
  - stack walking at safe points
  - loom support
  - ability to restart stack walking from arbitrary locations
- but don't
  - add an API for walking native/C/C++ frames
  - add a queue implementation for safepoint stack walking
  - support for multiple profilers
  - replace jmethodIds
  - be fuzzing safe, but still tested
- essentially
  - reduce the API to its bare minimum
  - don't implement anything in the VM that can be implemented in the profiler
  - start from scratch
- aim: be the minimal possible version of the JEP that still allows the proficient
  profiler writer to implement a profiler that can do everything the JEP
  would allow, just not as easily



# Welcome to the JDK!

For build instructions please see the
[online documentation](https://openjdk.org/groups/build/doc/building.html),
or either of these files:

- [doc/building.html](doc/building.html) (html version)
- [doc/building.md](doc/building.md) (markdown version)

See <https://openjdk.org/> for more information about the OpenJDK
Community and the JDK and see <https://bugs.openjdk.org> for JDK issue
tracking.
