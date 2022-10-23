inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
# for data in inputs:
#     if data > 0:
#         output.append(data)
#     elif data <= 0:
#         output.append(0)


# Alternative way:
for data in inputs:
    output.append(max(0, data))

print(output)