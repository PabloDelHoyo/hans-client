# HANS client
A Python client which interacts with the HANS platform.

## Installation
It is recommended to first create a virtual environment. After activating
it, run the following commands
```bash
$ python3 -m pip install --upgrade pip
$ python3 -m pip install git+https://github.com/PabloDelHoyo/hans-client
```

If you want to install the development version, run the following:
```bash
$ python3 -m pip install git+https://github.com/PabloDelHoyo/hans-client@dev
```

## Usage
The structure of a client is very similar to the one followed by a game loop. All the logic which makes the agent move must be in a subclass of `Agent`. These are the most important methods that you can override, but that is not compulsory:
* `setup(arg1, arg2, ...)`. 

This method is called every time a new round starts (when the participant start to answer by moving their square). It receives as many arguments as the creator of the subclass wants.

* `fixed_update(delta: float, sync_ration: float)`.

This method is called with a fixed `delta`. `delta` is the number of seconds which have passed since the last call to it. To avoid running into the "spiral of hell", the number
of fixed updates per seconds is upper bounded. This will have the effect of slowing down the simulation but at least the frame time (number of seconds the render method is called) will stop increasing and increasing.


`sync_ratio` is a quantity which measures how desynchronized `update()` is from
`fixed_update()`. In real games, it used to take into account the fact that rendering
and updating might happen at different rates. For this particular application, it is
not probably useful.

* `update(delta: float)`

This method is not guaranteed to be called at the same rate. That will depend on the work done in it. `sync_ratio` is a quantity associated to the way `update` and `render` are called at different rates. I cannot think of a reason to use it right now, so it can be safely ignored.

The main purpose of `render` is sending the position to the server.

* `close()`

Called when a round finishes. It is guaranteed to be the last call

Additionally, a subclass of `Agent` inherits two attributes:
* `snapshot`: It represents the state of a session. More concretely, it contains the position of each participant.
* `round`: It contains all the useful information for a question
* `client`: It allows you to send the position.

The skeleton of a client (after doing the corresponding imports) is therefore the following one:
```python
class AgentLogic(Agent):

    def setup(self, arg1, arg2, arg3):
        self.arg1 = arg1
        # ...
        self.position = np.zeros(2)
    
    def fixed_update(self, delta: float, sync_ratio: float):
        # calculations which require a fixed timestep to be reliable
        # update position using snapshot, self.round and delta
    
    def update(self, delta: float):
        # this method will be tried to be called at fix rate but that is not guaranteed.
        # It can be used to send the position
        ...

        self.client.send_position(self.position)

    def close(self):
        # last of piece of code executed 
```

NOTE: keep in mind that an instance of a `Agent` is destroyed when a round finishes and a fresh one will be created when a new question starts.

For running it, you must first create the thread which will execute the logic contained in the subclass of `Agent` in the follwing manner

```python
loop = AgentThread(AgentLogic, agent_kwargs={
    "arg1": "first argument",
    "arg2": "second argument",
    "arg3": "third argument"
})
```

Finally, pass the `AgentThread` to the `HansPlatform` so that it can handle it.

```python
with HansPlatform("agent name", loop) as platform:
    platform.connect("host")
    platform.listen()
```

## Coroutines
This feature is inspired by the coroutines feature that Unity [has](https://docs.unity3d.com/Manual/Coroutines.html). They allow you to
spread a task across several calls of `update()`. This is possible because
the task context is saved without you having to explicitily do it. You just
have to put the code inside a coroutine function (`async def `) and use `await` to yield the execution so that the next piece of code is executed after a call of `update()`. You would
normally use coroutines in this way:

```python
from hans import Loop
import hans.coro

class TestAgent(Loop):

    def setup():
        self.start_coroutine(self.my_coroutine())
    
    async def my_coroutine(self):
        # Some code

        # Wait at least five seconds
        await hans.coro.sleep(5)

        # More code

```

Keep in mind that couroutines have some overhead so if you have something like the following
```python
while True:
    # some code
    await hans.coro.sleep(0)
```
consider moving that piece of code to `update()` or `fixed_update()`

## Logging
`hans-client` uses the built-in `logging` module to log messages. By default, it does not send the logs anywhere. In order to do that, you must configure it. All used loggers have as parent a logger called `hans`. For a very basic configuration, you can do the following.
```python
def configure_logger(level, formatter, handler=logging.StreamHandler()):
    handler.setFormatter(formatter)

    logger = logging.getLogger("hans")
    logger.setLevel(level)
    logger.addHandler(handler)

# If you want to log info messages, write
# logging.INFO
configure_logger(
    logging.DEBUG,
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
```
In order to see everything you can do with the `logging`, I recommend you read the official documentation.

## Examples
In the `examples/` directory, there are some examples which show different clients interacting with the platform in different ways. You can run each one of them as a standalone script but before doing that make sure that all server related settings (host, port, etc) are configured correctly.