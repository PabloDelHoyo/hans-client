from dataclasses import dataclass, field
from typing import Any

import heapq


def _create_queue_element_cls(include_data):
    @dataclass(order=True)
    class _wrapper:
        order_items: tuple
        data: Any = field(compare=include_data)

    return _wrapper


class PriorityQueue:
    """A priority queue implementation using heaps which is not thread safe. It uses
    the heapq modules found in the standard library. As a consequence, it implements
    a min heap"""

    def __init__(self, keys=None, compare_data=False):
        self.queue = []
        self.keys = keys if keys is not None else []
        self.queue_item_cls = _create_queue_element_cls(compare_data)

    def put(self, item):
        order_items = tuple([fn(item) for fn in self.keys])
        queue_item = self.queue_item_cls(order_items, item)
        heapq.heappush(self.queue, queue_item)

    def peek(self):
        return self.queue[0].data

    def pop(self):
        queue_item = heapq.heappop(self.queue)
        return queue_item.data

    def is_empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self.queue)
