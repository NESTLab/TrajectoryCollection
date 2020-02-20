from vstig import VStig

class VSWeights:

    def __init__(self):
        self.ws = VStig()
        self.ss = VStig()

    def update(self, g):
        """Update the state of the structure according to the communication graph g"""
        self.ws.update(g)
        self.ss.update(g)

    def allsent(self):
        return self.ws.allsent() and self.ss.allsent()

    def put_weights(self, r, ws):
        """Robot r puts the weights ws in this structure"""
        # print("[VSWEIGHTS_PUT_WEIGHTS] {} put {}".format(r,ws))
        self.ws.put(r, r, ws)

    def put_samples(self, r, s):
        """Robot r puts the samples s in this structure"""
        # print("[VSWEIGHTS_PUT_SAMPLES] {} put {}".format(r,s))
        self.ss.put(r, r, s)

    def weights(self):
        return { r:v[r][0] for (r,v) in self.ws.rs.items() }

    def samples(self):
        return { r:v[r][0] for (r,v) in self.ss.rs.items() }
