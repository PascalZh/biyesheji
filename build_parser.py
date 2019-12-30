#!/usr/bin/env python3
from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef(
r"""
int parse_vlp_16(char [], float *, float *, float *, int);
int parse_rs(char [], float *, float *, float *, int);
int size_float();
""")

ffibuilder.set_source("_parser",
r"""
    int size_float()
    {
        return sizeof(float);
    }
    float unpack(char x[], int lend)
    {
        unsigned char a, b;
        a = x[0]; b = x[1];
        if (lend == 1) {
            return a | (b << 8);
        }
        if (lend == 0) {
            return b | (a << 8);
        }
    }
    static int parse_vlp_16(char buf[], \
            float *radius, float *intensity, float *azimuth, int x)
    {
        if (x != 1248)
            return 2;
        int p0, p1;
        for (int i = 0; i < 12; i++) {
            p0 = 42 + i * 100;
            if (buf[p0] != (char)0xff || buf[p0+1] != (char)0xee)
                return 3;
            azimuth[32*i] = unpack(buf+p0+2, 1) / 100.0;
            for (int j = 0; j < 16; j++) {
                p1 = p0 + 4 + 3 * j;
                radius[16*2*i+j] = unpack(buf+p1+1, 0) / 500.0;
                intensity[16*2*i+j] = (float)(unsigned char)(buf[p1+2]);
                p1 = p0 + 4 + 3 * j + 16 * 3;
                radius[16*(2*i+1)+j] = unpack(buf+p1, 1) / 500.0;
                intensity[16*(2*i+1)+j] = (float)(unsigned char)(buf[p1+2]);
            }
        }
        return 1;
    }
    static int parse_rs(char buf[], \
            float *radius, float *intensity, float *azimuth, int x)
    {
        if (x != 1248)
            return 2;
        int p0, p1;
        for (int i = 0; i < 12; i++) {
            p0 = 42 + i * 100;
            if (buf[p0] != (char)0xff || buf[p0+1] != (char)0xee)
                return 3;
            azimuth[32*i] = unpack(buf+p0+2, 0) / 100.0;
            for (int j = 0; j < 16; j++) {
                p1 = p0 + 4 + 3 * j;
                radius[16*2*i+j] = unpack(buf+p1, 0) / 500.0;
                intensity[16*2*i+j] = (float)(unsigned char)(buf[p1+2]);
                p1 = p0 + 4 + 3 * j + 16 * 3;
                radius[16*(2*i+1)+j] = unpack(buf+p1, 0) / 500.0;
                intensity[16*(2*i+1)+j] = (float)(unsigned char)(buf[p1+2]);
            }
        }
        return 1;
    }
""")

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
