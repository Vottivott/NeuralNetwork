import os
def get():
    nets = sorted(os.listdir("./saved_networks"))
    for net in nets:
        yield net


g = get()

for i in range(5):
    print g.next()