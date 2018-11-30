

import struct
import numpy as np
loc=np.loadtxt('loc',delimiter=' ')




class Binary_STL_Writer(ASCII_STL_Writer):
    def __init__(self, stream):
        self.counter = 0
        super(Binary_STL_Writer, self).__init__(stream)
    def close(self):
        self._write_header()
    def _write_header(self):
        self.fp.seek(0)
        self.fp.write(struct.pack(BINARY_HEADER, b'Python Binary STL Writer', self.counter))
    def _write(self, face):
        self.counter += 1
        data = [
            0., 0., 0.,
            face[0][0], face[0][1], face[0][2],
            face[1][0], face[1][1], face[1][2],
            face[2][0], face[2][1], face[2][2],
            0
        ]
        self.fp.write(struct.pack(BINARY_FACET, *data))


def example():
    def get_cube(x,y,z):
        # cube corner points
        s = 1.
        p1 = (x, y, z)
        p2 = (x, y, z+s)
        p3 = (x, y+s, z)
        p4 = (x, y+s, z+s)
        p5 = (x+s, y, z)
        p6 = (x+s, y, z+s)
        p7 = (x+s, y+s, z)
        p8 = (x+s, y+s, z+s)

        # define the 6 cube faces
        # faces just lists of 3 or 4 vertices
        return [
            [p1, p5, p7, p3],
            [p1, p5, p6, p2],
            [p5, p7, p8, p6],
            [p7, p8, p4, p3],
            [p1, p3, p4, p2],
            [p2, p6, p8, p4],
        ]

    with open('cube.stl', 'wb') as fp:
        writer = Binary_STL_Writer(fp)
        for i in range(len(loc)):
            writer.add_faces(get_cube(loc[i,0],loc[i,1],loc[i,2]))
        writer.close()

if __name__ == '__main__':
    example()