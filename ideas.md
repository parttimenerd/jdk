# Revamped version

##### ## Focus

- simplicity, don't add more APIs than we actually really need
- don't fuzz the API with random values



## Questions

- do we need `JavaThread ThreadsList::find_JavaThread_from_ucontext` ?

## Large Parts of the code are

- stackwalker -> remove by using ASGCT
  - maybe just have a call back called at every Java frame, with call back when walking over native frame (next native frame)
  - -> no more stack walker class or configuration
- queue implementation
  - let the profiler handle this
- no more Method* wrapping, jmethodIds should be fine
- info per frame
  - inlined?
  - pc, fp, sp
  - methodId
  - compilation level
  - frame start pc, fp, pc
    - -> in combination with methodId enough to implement incremental stack walking

## Aim

- the less changes to existing code the better
- walk at safepoints
- 