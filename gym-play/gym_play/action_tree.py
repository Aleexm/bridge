class ActionTree:
    "An ActionTree is an intermediary node representing an opponent's move."
    def __init__(self, parent):
        self.parent = parent
        self.children = {}

    def add_o(self, o_to_child, child):
        if o_to_child not in self.children.keys():
            self.children[o_to_child] = child

    def add_child(self, action_to_child, child):
        if action_to_child not in self.children.keys():
            self.children[action_to_child] = child

    def child_after_observation(self, a):
        self.add_child(a, ActionTree(parent=self))
