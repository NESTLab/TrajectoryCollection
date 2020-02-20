from vstig import VStig

class Barrier(VStig):

    def __init__(self):
        self.vs = VStig()
        self.rdy = []

    def quorum(self, n):
        """Returns True if (i) at least n robots have set themselves ready and (ii) the VStig is done updating"""
        return len(self.rdy) >= n and self.vs.allsent()

    def put(self, r):
        """Sets robot r as ready"""
        self.vs.put(r, r, 1)
        self.rdy.append(r)
        # print("[BARRIER_PUT] {} is ready".format(r))

    def update(self, g):
        """Updates the status of the barrier according to the communication graph g"""
        self.vs.update(g)

    def ready(self):
        """Returns a list of robots who set themselves as ready"""
        return self.rdy
