# NATHALIE
#
# nathalie_network(t) =
#   dictionary(rid, list(rid, rid, ...))
#
# nathalie_participating_robots(t) =
#   list(rid, rid, ...)
#
# nathalie_not_learning_robots(t) =
#   list(rid, rid, ...)
#
# nathalie_done_learning_robots(t) =
#   list(rid, rid, ...)
#
# nathalie_weights(rid) =
#   list(numpy_list(weights), ...))
#
# nathalie_samples(rid) =
#   nsamples

# WAITING FOR QUORUM
#
# t = START_T
# q = QUORUM
# barrier = new barrier()
#
# while (not barrier.quorum(q)):
#   barrier.update(nathalie_network(t))
#   for rid in nathalie_participating_robots(t):
#     barrier.put(rid)
#   t = t + 1
#
# barrier.ready() = [ rid, ... ]

# LEARNING
#
# t = t + 1
# vsweights = vstig()
# for rid in nathalie_not_learning_robots(t):
#     barrier.put(rid)
# 
# while (not barrier.quorum(NUM_ROBOTS)):
#   barrier.update(nathalie_network(t))
#   for rid in nathalie_done_learning_robots(t):
#     barrier.put(rid)
#     vsweights.put_weights(rid, nathalie_weights(rid))
#     vsweights.put_samples(rid, nathalie_samples(rid))
#   t = t + 1

# AGGREGATE
#
# vsweights.weights()
#   dictionary(rid, [
#     numpy list for layer 1,
#     numpy list for layer 2,
#     numpy list for layer 3,
#     numpy list for layer 4,
#     numpy list for layer 5
#   ])
#
# vsweights.samples()
#   dictionary(rid, nsamples)
