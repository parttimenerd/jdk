# Fork

Fork of the JDK to experiment with a new version of the AsyncGetCallTrace call (and code reuse regarding stack walking).

It should work with JMC and related tools.
For using the AsyncGetCallTrace2, we recommend using a modified [async-profiler](https://github.com/SAP/async-profiler/tree/parttimenerd_asgct2).

This is still an early draft and especially the data structures of AsyncGetCallTrace2 (and its name) are not finalized and represent more
a proof of concept. The data structures should later be more efficient,
but there current implementation made creating a working async-profiler
far easier (and required less changes to its structure).

For more information and a demo, please refer to [asgct2-demo](https://github.com/parttimenerd/asgct2-demo).


# Welcome to the JDK!

For build instructions please see the
[online documentation](https://openjdk.java.net/groups/build/doc/building.html),
or either of these files:

- [doc/building.html](doc/building.html) (html version)
- [doc/building.md](doc/building.md) (markdown version)

See <https://openjdk.java.net/> for more information about
the OpenJDK Community and the JDK.
