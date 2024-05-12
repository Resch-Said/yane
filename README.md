# Yet Another Neuro Evolution

Currently still in the planning phase. If someone is interested in the implementation, then the source code is the documentation

If you want to see yane in action, go to src -> examples -> gym -> Cartpole. The other examples currently work rather semi well.

!Warning: This is a slow long term project. Means, this project could be out of order for several months before further development (mainly because I'm currently busy).

## Current Plan
- Redoing and making it more simple. The aim is also to define as many parameters as possible genetically so that the user can concentrate as much as possible on implementing the fitness function.
- Switch to a server-client method, which should make it possible for "incomplete clients" in other programming languages to be able to perform certain tasks, even if they cannot perform every task. For example, if a C++ client only has the function to create a new genome, the server should be able to adapt to this and not send this client any other tasks that it cannot solve.
