# this is if the ENTRYPOINT was `python -m recsys.api`

# time podman run recsys '{"query":["this should error"], "limit": -1}' 
time podman run recsys '{"query":["hello", "world", "hello world"], "limit": 5}'
time podman run recsys '{"query":["dog", "park", "fetch", "woof"], "limit": 10}'
time podman run recsys '{"query":["mad men", "ball", "game on"], "limit": 5}'
time podman run recsys '{"query":["poker", "straight flush", "hand", "cooler"], "limit": 5}'
time podman run recsys '{"query":["no limit", "gogogo"]}'
time podman run recsys '{"query":["thread pool", "builder", "laying the ground work"]}'