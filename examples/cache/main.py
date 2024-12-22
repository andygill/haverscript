from haverscript import connect, cache, options
import time
import sys

# This enabled the cache
model = connect("mistral") | cache("cache.db")

prompt = "In one sentence, why is the sky blue?"
times = []
replies = []

# This runs a query several times, returning any cached answer first
times.append(time.time())
for i in range(int(sys.argv[1])):
    replies.append(model.chat(prompt, middleware=options(seed=i)).reply)
    times.append(time.time())

for i, (t1, t2, r) in enumerate(zip(times, times[1:], replies)):
    print(f"chat #{i}")
    print("reply:", r)
    print("fast (used cache)" if t2 - t1 < 0.5 else "slower (used LLM)")
    print()
