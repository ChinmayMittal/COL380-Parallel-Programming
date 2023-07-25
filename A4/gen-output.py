"""Multiply two matrices."""
import os
from argparse import ArgumentParser
from typing import List
import numpy as np

parser = ArgumentParser()
parser.add_argument("--filepaths", "-f", type=str, nargs="+", required=True,
                    help="Files to crosscheck.")
parser.add_argument("--element_size", "-e", dest="size", type=int, required=True,
                    help="Size of each block element in bytes.")
parser.add_argument("--output", "-o", type=str, required=True,
                    help="Output filename.")
args = parser.parse_args()
print(args)

assert len(args.filepaths) == 2
for filepath in args.filepaths:
    assert os.path.exists(filepath)

def read_matrix(filepath:str) -> List:
    with open(filepath, "rb") as file:
        # read n, m, k
        n = int.from_bytes(file.read(4), byteorder="little")
        m = int.from_bytes(file.read(4), byteorder="little")
        k = int.from_bytes(file.read(4), byteorder="little")
        blocks = list()
        for __ in range(k):
            x = int.from_bytes(file.read(4), byteorder="little")
            y = int.from_bytes(file.read(4), byteorder="little")
            block = np.array(
                [int.from_bytes(file.read(args.size), byteorder="little") for __ in range(m**2)],
                dtype=np.uint32,
            ).reshape(m, m)
            blocks.append([x, y, block])
    return n, m, k, blocks

def write_matrix(n:int, m:int, k:int, matrix:List, filepath:str) -> None:
    with open(filepath, "wb") as file:
        file.write(n.to_bytes(length=4, byteorder="little"))
        file.write(m.to_bytes(length=4, byteorder="little"))
        file.write(k.to_bytes(length=4, byteorder="little"))
        for block in matrix:
            # coordinates
            file.write(block[0].to_bytes(length=4, byteorder="little"))
            file.write(block[1].to_bytes(length=4, byteorder="little"))
            # elements
            file.write(block[2].tobytes())
    return

if __name__ == "__main__":
    #* Read the matrices.
    n1, m1, k1, mat1 = read_matrix(args.filepaths[0])
    n2, m2, k2, mat2 = read_matrix(args.filepaths[1])
    assert n1 == n2
    assert m1 == m2

    MAX = int(2**32 - 1)

    #* Matrix multiplication
    # create an empty list for output blocks.
    mat3 = dict()
    n3 = n1
    m3 = m1
    k3 = 0
    # create placeholder block.
    # iterate over the blocks of m1 and m2.
    for block1 in mat1:
        x1, y1 = block1[0], block1[1]
        for block2 in mat2:
            x2, y2 = block2[0], block2[1]
            # match m1's column and m2's row.
            # if they don't match, continue.
            if y1 != x2:
                continue
            # multiply the blocks and add to block3. Clip the values before adding.
            product = np.clip(block1[2] @ block2[2], a_min=0, a_max=MAX)
            # if the output matrix is all zeros.
            if not np.any(product):
                continue
            # add the block to mat3 with coordinates as (x1, y2).
            if (x1, y2) not in mat3.keys():
                # first block for row of mat1 and col of mat2.
                mat3[(x1, y2)] = product
                k3 += 1
            else:
                # another block found for row of mat1 and col of mat2.
                mat3[(x1, y2)] += product

    # convert mat3 to a list
    mat3 = [[key[0], key[1], val] for key, val in mat3.items()]

    #* Write the output matrix to disk.
    write_matrix(n=n3, m=m3, k=k3, matrix=mat3, filepath=args.output)
