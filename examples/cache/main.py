from haverscript import connect, fresh
import time

model = connect("mistral").cache("cache.db")

prompt = "In one sentence, why is the sky blue?"
times = []
replies = []
times.append(time.time())
replies.append(model.chat(prompt).reply)
times.append(time.time())
replies.append(model.chat(prompt).reply)
times.append(time.time())
replies.append(model.chat(prompt).check(fresh).reply)
times.append(time.time())

for i, (t1, t2, r) in enumerate(zip(times, times[1:], replies)):
    print(f"chat #{i}")
    print("reply:", r)
    print("time: {:.5f}".format(t2 - t1))
    print()

print(len(model.children(prompt)), "replies are cached")
