def split_data(source_file, destination_file, n):
    with open(source_file, 'r') as source:
        lines = source.readlines()

    last_n_lines = lines[-n:]

    with open(destination_file, 'w') as destination:
        destination.writelines(last_n_lines)

    lines = lines[:-n]

    with open(source_file, 'w') as source:
        source.writelines(lines)


source_file = 'data/all.txt'
destination_file = 'data/test.txt'
n = 40000

split_data(source_file, destination_file, n)
