from haverscript import connect, fresh
import time
import sys

model = connect("mistral").cache("cache.db")

prompt = "In one sentence, why is the sky blue?"
times = []
replies = []

print(len(model.children(prompt)), "replies are in cache at start of run")

times.append(time.time())
for _ in range(int(sys.argv[1])):
    replies.append(model.chat(prompt).reply)
    times.append(time.time())

for i, (t1, t2, r) in enumerate(zip(times, times[1:], replies)):
    print(f"chat #{i}")
    print("reply:", r)
    print("time: {:.5f}".format(t2 - t1))
    print()

print(len(model.children(prompt)), "replies are in cache at end of run")
