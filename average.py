import os
import sys

# filename = sys.argv[1]
# count = 0
# update = 0
# cumulate = 0
# gbs = 0
# total = 0
# with open(filename) as fin:
#     for line in fin:
#         if "update, cumulate, gbs" in line:
#             gg = line.split(":")[1].strip()
#             nums = gg.split(",")
#             update += int(nums[0])
#             cumulate += int(nums[1])
#             gbs += int(nums[2])
#             total += int(nums[3])
#             count+=1
# print(update/count)
# print(cumulate/count)
# print(gbs/count)
# print(total/count)

for n in range(10, 22):
    filename = "out_" + str(n)
    count = 0
    update = 0
    cumulate = 0
    gbs = 0
    total = 0
    with open(filename) as fin:
        for line in fin:
            if "update, cumulate, gbs" in line:
                gg = line.split(":")[1].strip()
                nums = gg.split(",")
                update += int(nums[0])
                cumulate += int(nums[1])
                gbs += int(nums[2])
                total += int(nums[3])
                count+=1
    # print(update/count)
    # print(cumulate/count)
    # print(gbs/count)
    # print(total/count)
    print("{},{},{},{}".format(update/count,cumulate/count,gbs/count,total/count))


