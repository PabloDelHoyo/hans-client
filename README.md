# HANS client
A Python client which interacts with the HANS platform.

## Installation
It is recommended to first create a virtual environment. After activating
it, run the following commands
```
$ python3 -m pip install --upgrade pip
$ python3 -m pip install git+https://github.com/PabloDelHoyo/hans-client
```

## Usage
The structure of a client is very similar to the one followed by a game loop. All the logic which makes the agent move must be in subclass of `Loop`. These are the most important methods that you can override, but that is not compulsory:
* `setup(arg1, arg2, ...)`. 

This method is called every time a new round starts (when the participant start to answer by moving their square). It receives as many arguments as the creator of the subclass wants.

* `update(snapshot: StateSnapshot, delta: float)`.

This method is called with a fixed `delta`. `delta` is the number of seconds which have passed since the last call to it. In the future, to avoid "the spiral of hell", the number of times this method is called in a second may decrease in case a lot of work is being done for a period of time. This will have the effect of slowing down the simulation but at least the frame time (number of seconds the render method is called) will stop increasing and increasing

This method is meant to be used for the calculation of the position where the agent will be.

* `render(sync_ratio: float)`

This method is not guaranteed to be called at the same rate. That will depend on the work done in the update method. `sync_ratio` is a quantity associated to the way `update` and `render` are called at different rates. I cannot think of a reason to use it right now, so it can be safely ignored.

The main purpose of `render` is sending the position to the server.


Additionally, a subclass of `Loop` inherits two attributes:
* `round`: It contains all the useful information for a question
* `client`: It allows you to send the position.

The skeleton of a client (after doing the corresponding imports) is therefore the following one:
```python
class AgentLogic(Loop):

    def setup(self, arg1, arg2, arg3):
        self.arg1 = arg1
        # ...
        self.position = np.zeros(2)
    
    def update(self, snapshot: StateSnapshot, delta: float):
        # update position using snapshot, self.round and delta
    
    def render(self, sync_ratio: float):
        # Most of the time, you will do the following
        self.client.send_position(self.position)
```

NOTE: keep in mind that an instance of a `Loop` is destroyed when a round finishes and a fresh one will be created when a new question starts.

For running it, you must first create the thread which will execute the logic contained in the subclass of `Loop` in the follwing manner

```python
loop = LoopThread(AgentLogic, loop_kwargs={
    "arg1": "first argument",
    "arg2": "second argument",
    "arg3": "third argument"
})
```

Finally, pass the `LoopThread` to the `HansPlatform` so that it can handle it.

```python
with HansPlatform("agent name", loop) as platform:
    platform.connect("host")
    platform.listen()
```

## Examples
In the `examples/` directory, there are some examples which show different clients interacting with the platform in different ways. You can run each one of them as a standalone script but before doing that make sure that all server related settings (host, port, etc) are configured correctly.