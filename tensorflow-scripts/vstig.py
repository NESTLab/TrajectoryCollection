from network import Network

class VStig:

    def __init__(self):
        self.net = Network() # communication network
        self.rs = {}         # robots and their associated data

    def put(self, r, k, v):
        """Robot r puts tuple (k,v)"""
        #
        # Every element is stored as a tuple (v,l,o) where
        # [0] v: value
        # [1] l: lamport clock
        # [2] o: owner (id of robot that put the value)
        #
        # Get values of robot r
        rvs = self.rs.get(r, {})
        # Add tuple to r's values
        if k in rvs:
            rvs[k] = (v, rvs[k][1]+1, r)
        else:
            rvs[k] = (v, 0, r)
        self.rs[r] = rvs
        # print("[VSTIG_PUT] {}: {!s}".format(r,self.rs[r]))
        # Broadcast update
        self.net.broadcast(r, (v,rvs[k][1],r,k))

    def update(self, g):
        """Updates the state of the vstig according to the communication graph g"""
        # Route all messages
        self.net.route(g)
        # Go through the input queues and process the messages
        for r,q in self.net.inq.items():
            self.process(r, q)
        # Done with current messages
        self.net.clear()

    def allsent(self):
        """Returns True if no more messages need to be sent"""
        return self.net.size() == 0

    def process(self, r, q):
        # Get values of robot r
        rvs = self.rs.get(r, {})
        # For every received message
        for m in q:
            # Is the key already in the structure?
            if m[3] in rvs:
                # Is the received value newer than the stored one?
                if m[1] > rvs[m[3]][1]:
                    # Store the new value
                    rvs[m[3]] = m[0:3]
                    # print("[VSTIG_PROCESS] Updated value: {} now stores {!s}".format(r,(m[3],rvs[m[3]])))
                    # Keep diffusing it
                    self.net.broadcast(r, m)
                # Is the received value older than the stored one?
                #elif (m[1] < rvs[m[3]][1]) or (m[1] == rvs[m[3]][1] and m[2] == rvs[m[3]][2]):
                    # Ignore it
                  #  print("[VSTIG_PROCESS] Ignored value {!s} by {}".format(m,r))
                # The received value and the stored value have the same Lamport clock -> conflict!
                #else:
                 #   print("[VSTIG_PROCESS] Conflicting value: {} stores {!s} and received {!s}".format(r,(m[3],rvs[m[3]]),m))
            # The key is not already in the structure
            else:
                # Store the new value
                rvs[m[3]] = m[0:3]
                # print("[VSTIG_PROCESS] Inserted value: {} now stores {!s}".format(r,(m[3],rvs[m[3]])))
                # Keep diffusing it
                self.net.broadcast(r, m)
        # Store values of robot r
        self.rs[r] = rvs
