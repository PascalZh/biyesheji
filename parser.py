#! /usr/bin/env python3

from matplotlib.widgets import Button, TextBox

from utils import *
from _parser import lib, ffi

vlp16_vert_angle = np.array([ -15, 1, -13, -3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15], dtype='float')

np.set_printoptions(threshold=1e6)
# number of points in a frame
fpnum = 28959

dt = np.dtype('f%s' % lib.size_float())


def gen_interp(gap):
    interp = np.zeros(384)
    for i in range(12):
        for k in range(31):
            if k < 16:
                interp[32*i+k+1] = interp[16*2*i]+(gap*2.304*(k+1))/55.296
            else:
                interp[32*i+k+1] = interp[16*2*i]+(gap*2.304*((k-15)+55.296))/(2*55.296)
    return interp


@record_run_time
def calc_fpnum(azimuths):
    inds = np.argsort(np.abs(azimuths-azimuths[0]))
    for i, ind in enumerate(inds):
        if ind == 2:
            break
    c = inds[:i]
    ret = c[np.argsort(c)[2]]
    #print(azimuths[0:ret])
    return ret


interp_vlp = gen_interp(0.2)
interp_rs = gen_interp(0.18)

@record_run_time
@jit
def interpolate_azimuth(data, lidar):
    '''interpolate azimuth
    '''
    if lidar == 0:
        data[:,2] += interp_vlp
    elif lidar == 1:
        gap = 0.18
        data[:,2] += interp_rs
    for i in range(12):
        for j in range(16):
            data[16*2*i+j][2] = data[32*i][2]
            data[16*(2*i+1)+j][2] = data[0][2] + gap*(2*i+1)
    return data


@record_run_time
def parse_packet(buf, lidar):
    # data fields: distance intensity azimuth
    buffer_in = ffi.from_buffer(buf)
    radius = ffi.new('float[]', 384)
    intensity = ffi.new('float[]', 384)
    azimuth = ffi.new('float[]', 384)

    if lidar == 0:
        lib.parse_vlp_16(buffer_in, radius, intensity, azimuth, 1248)
    elif lidar == 1:
        lib.parse_rs(buffer_in, radius, intensity, azimuth, 1248)

    #start = time.time()
    #span = time.time() - start
    #name = 'test'
    #if name not in run_time.keys():
    #    run_time[name] = [span, 1]
    #else:
    #    run_time[name][0] += span
    #    run_time[name][1] += 1

    radius = np.frombuffer(ffi.buffer(radius), dtype=dt)
    intensity =np.frombuffer(ffi.buffer(intensity), dtype=dt)
    azimuth =np.frombuffer(ffi.buffer(azimuth), dtype=dt)

    return np.array([radius, intensity, azimuth]).transpose()

@record_run_time
def convert_pointcloud(data):
    '''get the point cloud, supposing x-axis is the north direction'''
    pointcloud = np.zeros((24*16, 4), dtype='float')

    alpha = pi/180.0*data[:,2]
    omega = pi/180.0*np.tile(vlp16_vert_angle, 24)
    r = data[:,0]
    pointcloud[:,0] = r*cos(omega)*sin(alpha)
    pointcloud[:,1] = r*cos(omega)*cos(alpha)
    pointcloud[:,2] = r*sin(omega)
    pointcloud[:,3] = data[:,1]
    return pointcloud


pc = np.zeros((fpnum*2, 4), dtype='float')
ptr_end = 0

@record_run_time
def parse(buf, lidar):
    global fpnum, pc, ptr_end
    data = parse_packet(buf, lidar)
    data = interpolate_azimuth(data, lidar)

    assert data.shape == (384, 3)
    pc[ptr_end:ptr_end+384] = convert_pointcloud(data)
    ptr_end += 384
    if ptr_end > fpnum:
        ret = pc[0:fpnum].copy()
        pc[0:ptr_end-fpnum] = pc[fpnum:ptr_end]
        ptr_end = ptr_end - fpnum
        return ret
    else:
        return None
    #x = pc[:,0]
    #y = pc[:,1]
    #z = pc[:,2]
    #s = pc[:,3]
    #l = mlab.points3d(x, y, z, s, mode='point', colormap='cool', scale_factor=10.5)


    #@mlab.animate(delay=100)
    #def anim():
        #for i in range(100):
            #l.mlab_source.x = pc_frames[i,:,0]
            #l.mlab_source.y = pc_frames[i,:,1]
            #l.mlab_source.z = pc_frames[i,:,2]
            #l.mlab_source.scalars = pc_frames[i,:,3]
            #yield


    #anim()
    #mlab.show()

def test():
    """Open vlp pcap file and print out the packets"""
    with open('data/2015-07-23-14-37-22_Velodyne-VLP-16-Data_Downtown 10Hz Single.pcap', 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for timestamp, buf in pcap:
            if len(buf) == 1248:
                parse(buf)

    show_run_time()

if __name__ == '__main__':
    test()
