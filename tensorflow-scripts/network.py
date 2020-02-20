class Network:

    def __init__(self):
        self.outq = {} # queue of broadcast messages
        self.inq = {}  # queue of received messages

    def broadcast(self,r, m):
        """Robot r broadcasts message m"""
        self.outq[r] = self.outq.get(r, []) + [m]
        # print("[NETWORK_BROADCAST] {}: {!s}".format(r,self.outq[r]))

    def route(self, g):
        """Routes the messages according to the communication graph g"""
        # For each robot and broadcast message
        for r,ms in self.outq.items():
            # For each neighbor of r
            for n in g[r]:
                # For each message
                for m in ms:
                    # Add message to receive queue of neighbor
                    self.inq[n] = self.inq.get(n, []) + [m]
                    # print("[NETWORK_ROUTE] {} -> {}: {!s}".format(r,n,self.inq[n]))
        # Clear queue of broadcast messages
        self.outq = {}

    def clear(self):
        """Clear the queue of received messages"""
        self.inq = {} # queue of received messages

    def size(self):
        """Returns the number of messages to route"""
        return len(self.outq)
