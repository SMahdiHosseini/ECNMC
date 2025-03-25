import os
f = open("result_0.txt", "r")
lines = f.readlines()
f.close()

for l in range(len(lines)):
    if "### EVENT ### " in lines[l]:
        # find the number after "Queue Size: " in the previous line. The format of the previous line is "Queue Size: <number> Total Queue Size: <number> Queuing Delay: <number> packet size: <number>"
        last_size = int(lines[l-1].split("Queue Size: ")[1].split()[0])
        curr_size = int(lines[l].split("Queue Size: ")[1].split()[0])
        if (curr_size == 0) and (last_size != 0) and (curr_size != last_size):
            print("problem in line ", lines[l].split("Time: ")[1].split()[0])
            print("last size: ", last_size)
            print("curr size: ", curr_size)
            print("**********")