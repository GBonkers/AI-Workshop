class Node():
    def __init__(self, state, parent, action, g=0, f=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.f = f


class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node

class AStarFrontier:
    def __init__(self):
        self.frontier = []

    def add(self, node):
        """Simply append; ordering is handled by remove()."""
        self.frontier.append(node)

    def contains_state(self, state):
        return any(n.state == state for n in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        """
        Find and pop the node with the lowest f-value.
        This implements the priority queue behavior without imports.
        """
        if self.empty():
            raise Exception("empty frontier")
        # find index of node with minimal f
        lowest_index = min(
            range(len(self.frontier)),
            key=lambda i: self.frontier[i].f
        )
        return self.frontier.pop(lowest_index)