#! /usr/bin/env python3

from utils import record_run_time, show_run_time, plot3d, plt
import numpy as np
import dpkt
from numpy import cos, sin, pi
from numba import njit
from mayavi import mlab
from _parser import lib, ffi

vlp16_vert_angle = np.array([
    -15, 1, -13, -3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15],
    dtype='float')
rs_vert_angle = np.array([
    -15, -13, -11, -9, -7, -5, -3, -1, +15, +13, +11, +9, +7, +5, +3, +1],
    dtype='float')

np.set_printoptions(threshold=1e6)

dt = np.dtype('f%s' % lib.size_float())


def gen_interp(gap):
    interp = np.zeros(384)
    for i in range(12):
        for k in range(31):
            if k < 16:
                interp[32*i+k+1] = interp[16*2*i]+(gap*2.304*(k+1))/55.296
            else:
                interp[32*i+k+1] = interp[16*2*i] +\
                                   (gap*2.304*((k-15)+55.296))/(2*55.296)
    return interp


def enclosure_calc_const():
    azimuth_sum = 0.0
    cnt = 0
    old_azimuth = None

    def calc_const(azimuth):
        nonlocal azimuth_sum, cnt, old_azimuth
        if azimuth_sum > 360:
            print("(%s, %s)" % (cnt, cnt / 16 * 55.296 / 1000 / 1000))
            return

        if old_azimuth is None:
            old_azimuth = azimuth
            return
        delta_azimuth = azimuth - old_azimuth
        old_azimuth = azimuth
        if delta_azimuth > 360:
            delta_azimuth -= 360
        if delta_azimuth < 0:
            delta_azimuth += 360
        azimuth_sum += delta_azimuth
        cnt += 1
    return calc_const


calc_const = enclosure_calc_const()


interp_vlp = gen_interp(0.2)
interp_rs = gen_interp(0.18)


@record_run_time
def interpolate_azimuth(data, lidar):
    if lidar == 'vlp':
        gap = 0.2
        data[:, 2] += interp_vlp
    elif lidar == 'rs':
        gap = 0.18
        data[:, 2] += interp_rs
    # print(f"before:{data[:,2]}")
    _interpolate(data, gap)
    # print(data[:,2])
    return data


@njit
def _interpolate(data, gap):
    """This function is seperated from `interpolate_azimuth` in order to use
    jit from numba. For some reasons, adding `@njit` to `interpolate_azimuth`
    doesn't work."""
    for i in range(12):
        for j in range(16):
            data[16*2*i+j][2] = data[32*i][2]
            data[16*(2*i+1)+j][2] = data[0][2] + gap*(2*i+1)


@record_run_time
def parse_packet(buf, lidar):
    # data fields: distance intensity azimuth
    buffer_in = ffi.from_buffer(buf)
    radius = ffi.new('float[]', 384)
    intensity = ffi.new('float[]', 384)
    azimuth = ffi.new('float[]', 384)

    if lidar == 'vlp':
        lib.parse_vlp_16(buffer_in, radius, intensity, azimuth, 1248)
    elif lidar == 'rs':
        lib.parse_rs(buffer_in, radius, intensity, azimuth, 1248)

    radius = np.frombuffer(ffi.buffer(radius), dtype=dt)
    intensity = np.frombuffer(ffi.buffer(intensity), dtype=dt)
    azimuth = np.frombuffer(ffi.buffer(azimuth), dtype=dt)

    return np.array([radius, intensity, azimuth]).transpose()


@record_run_time
def convert_pointcloud(data, lidar):
    '''get the point cloud, supposing x-axis is the north direction'''
    pointcloud = np.zeros((24*16, 4), dtype='float')

    alpha = pi / 180.0 * data[:, 2]
    if lidar == 'vlp':
        vert_angle = vlp16_vert_angle
    elif lidar == 'rs':
        vert_angle = rs_vert_angle
    omega = pi / 180.0 * np.tile(vert_angle, 24)

    r = data[:, 0]
    pointcloud[:, 0] = r * cos(omega) * sin(alpha)
    pointcloud[:, 1] = r * cos(omega) * cos(alpha)
    pointcloud[:, 2] = r * sin(omega)
    pointcloud[:, 3] = data[:, 1]
    return pointcloud


class Parser(object):

    def __init__(self, lidar, pcap_file):
        self.lidar = lidar
        self.pcap_file = pcap_file

        self.n_points_per_frame = 28960 if lidar == 'vlp' else 31872
        self.pc_buf = np.zeros((self.n_points_per_frame*2, 4), dtype='float')
        self.end = 0

    @record_run_time
    def parse(self, buf):
        """
        Parameters:
            buf : buffer
                content of a network frame
            lidar: str
                'vlp' for vlp16, 'rs' for rs lidar
        """
        # for rs lidar, first 42 bytes are used for other intention.
        if len(buf) == 1290:
            buf = buf[42:]

        M = self.n_points_per_frame
        data = parse_packet(buf, self.lidar)
        data = interpolate_azimuth(data, self.lidar)

        # for azimuth in data[:, 2]:
            # calc_const(azimuth)

        assert data.shape == (384, 3)

        self.pc_buf[self.end:
                    self.end + 384] = convert_pointcloud(data, self.lidar)
        self.end += 384

        if self.end > M:
            ret = self.pc_buf[0:M].copy()
            self.pc_buf[0:self.end-M] = self.pc_buf[M:self.end]
            self.end = self.end - M
            return ret
        else:
            return None

    def generator(self):
        with open(self.pcap_file, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for timestamp, buf in pcap:
                if len(buf) != 1248 and len(buf) != 1290:
                    continue
                ret = self.parse(buf)
                if ret is None:
                    continue
                yield ret


def animate_pcap(lidar, pcap_file):
    """Animate the point clouds pcap file.

    Parameters:
        lidar : str
            Possible values: 'vlp' or 'rs'.
        pcap_file : str
            File name.
    """
    parser = Parser(lidar, pcap_file)
    generator = parser.generator()
    frame = next(generator)

    obj = mlab.points3d(frame[:, 0], frame[:, 1], frame[:, 2], frame[:, 3],
                        mode='point', colormap='cool', scale_factor=10.5)
    ms = obj.mlab_source

    @mlab.animate(delay=100)
    def anim():
        for frame in generator:
            ms.x = frame[:, 0]
            ms.y = frame[:, 1]
            ms.z = frame[:, 2]
            ms.scalars = frame[:, 3]
            yield

    anim()
    mlab.show()


def test():
    """Open vlp pcap file and print out the packets"""
    parser = Parser('rs', 'data/2019-07-15-10-23-00-RS-16-Data.pcap')
    for frame in parser.generator():
        # plot3d(frame)
        # plt.show()
        pass


if __name__ == '__main__':
    # animate_pcap('rs', 'data/2019-07-15-10-23-00-RS-16-Data.pcap')
    test()
    show_run_time()
