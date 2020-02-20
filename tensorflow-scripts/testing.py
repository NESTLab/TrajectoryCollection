from barrier import Barrier
from vsweights import VSWeights

#
# TEST GRAPH
#
# 1 - 2 - 4
# | /
# 3
#
graph = {
    # id : [id, id, ...]
    1: [2,3],
    2: [1,3,4],
    3: [1,2],
    4: [2]
}

#
# TEST READY TIMES
#
ready = {
    # step : id
    2: 1,
    4: 2,
    8: 3,
    16: 4
}

#
# TEST BARRIER
#
b = Barrier()
t = 0
print("[TEST BARRIER] Time start:", t)
while(not b.quorum(len(graph))):
    b.update(graph)
    if t in ready:
        b.put(ready[t])
    t = t + 1
print("[TEST BARRIER] Time end:", t)
print("[TEST BARRIER] Ready robots: {}".format(b.ready()))


#
# TEST WEIGHTS
#
# For simplicity we use plain lists, but it's the same with numpy lists
w = VSWeights()

t = 0
print("[TEST WEIGHTS] Time start:", t)
# Robot 1
w.put_weights(1, [
    [1.11, 1.12, 1.13], # Layer 1
    [1.21, 1.22]        # Layer 2
])
w.put_samples(1, 10)
# Keep spinning until all messages have been sent
while not w.allsent():
    w.update(graph)
    t = t + 1

# Robot 2
w.put_weights(2, [
    [2.11, 2.12, 2.13], # Layer 1
    [2.21, 2.22]        # Layer 2
])
w.put_samples(2, 20)
# Keep spinning until all messages have been sent
while not w.allsent():
    w.update(graph)
    t = t + 1

# Robot 3
w.put_weights(3, [
    [3.11, 3.12, 3.13], # Layer 1
    [3.21, 3.22]        # Layer 2
])
w.put_samples(3, 30)
# Keep spinning until all messages have been sent
while not w.allsent():
    w.update(graph)
    t = t + 1

# Robot 4
w.put_weights(4, [
    [4.11, 4.12, 4.13], # Layer 1
    [4.21, 4.22]        # Layer 2
])
w.put_samples(4, 40)
# Keep spinning until all messages have been sent
while not w.allsent():
    w.update(graph)
    t = t + 1

print("[TEST WEIGHTS] Time end:", t)
print("[TEST WEIGHTS] weights = {!s}".format(w.weights()))
print("[TEST WEIGHTS] samples = {!s}".format(w.samples()))
