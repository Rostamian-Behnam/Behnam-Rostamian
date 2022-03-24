# %% python code for 3d printer control
import serial
import time
import numpy as np
from threading import Thread
from queue import Queue

state_sp = True
p = serial.Serial('COM4', 115200, timeout=1)
p.close()
p.open()

if state_sp:
    sp = serial.Serial('COM6', 115200, timeout=1)  # raw_data
    sp.close()
    
vib = np.array([])

speed_z= 3000
reset = b'G28 X0 Y011\n'
reset_z = b'G28 Z0\n'
reset_all = b'G28 X0 Y0 Z0\n'
beep = b'M300 S300 P1000\n'
active = 0
Z = 0
X = 0
Y = 0

ind=20

px01=px11=px21=px31=px41=px51=px61=120
px02=px12=px22=px32=px42=px52=px62=50
py01=py02= 17
py11=py12= 47
py21=py22= 77
py31=py32= 105
py41=py42= 135
py51=py52= 165
py61=py62= 198
z0=z1=z2=z3=z4=16
z5=z6=16

def reset_all_safe():
    global X, Y, Z
    z = 45
    Z += z
    p.write(indentation(z, 1500))
    p.write(reset)
    p.write(reset_z)
    z = 45
    Z += z
    p.write(indentation(z, 1500))
    X = 0
    Y = 0
    
    
def pause():
    programPause = input("Press the <ENTER> key to continue...")

def response():
    out = str()
    while p.in_waiting == 0:
        pass
    while p.in_waiting > 0:
        out = str(p.readline())
        print(out)


def indentation(z, speed):
    pathes = 'G90\nG1 Z{} F{}\nG90\n'.format(z, speed)
    pathes_byte = pathes.encode()
    out = str()
    while p.in_waiting == 0:
        pass
    while p.in_waiting > 0:
        out = str(p.readline())
        print(out)
    return pathes_byte


def motion(x, y, speed):
    pathes = 'G90\nG1 X{} Y{} F{}\nG90\n'.format(x, y, speed)
    pathes_byte = pathes.encode()
    out = str()
    while p.in_waiting == 0:
        pass
    while p.in_waiting > 0:
        out = str(p.readline())
        print(out)
    return pathes_byte


def store(trial, sh_):
    global vib, speed , z 
    print('shape:',sh_)
    print('trial:',trial)
    print('speed:',speed)
    np.save('C:\\Users\\Behnam\\Desktop\\Temp_data\\Pro\\Pro_Ro{}_tr{}_sp{}.npy'.format(sh_,trial,speed), vib)
def record():
    global active, state_sp, vib
    while state_sp:
        if active == 1 :
            try:
                raw_data = sp.readline()
                raw_data = raw_data.decode().rstrip().split('\t')
                raw_data1=raw_data[0]
                pizo_data=int(raw_data1)
                print('pizo                   ',pizo_data)
                vib = np.append(vib,pizo_data)
            except:
                print('error 1')
        else:
            vib = np.array([]) 
            
def profile(x1, y1, x2, y2, z, speed , trial):
    global active , sh_, x, y, speed_z, ind
    
    for r in range(trial):
        
        t = (ind / speed_z) * 60
        p.write(indentation(z+ind,speed_z))
        time.sleep(t)
        
        delay = abs(x - x1) + abs(y - y1)
        t = (delay / speed_z) * 60
        p.write(motion(x1, y1, speed_z))
        time.sleep(t+2) 
        t = (ind / speed_z) * 60
        p.write(indentation(z,speed_z)) 
        time.sleep(t)
        if state_sp:
            active = 1    
            sp.open()
            print("port is opend")
        start = time.time()
        # time.sleep(1)  ###########################################################
        delay = abs(x1 - x2) + abs(y1 - y2)
        t = (delay / speed) * 60
        p.write(motion(x2, y2, speed))
        time.sleep(t)
        
        # time.sleep(2) ################################################################
        if state_sp:
            store(r, sh_)
            active = 0
            sp.close()
            print("port is closed")
            print(sh_)    
            print(r)    
        t = (ind / speed_z) * 60
        p.write(indentation(z+ind,speed_z))
        time.sleep(t)
def Profile_test(z, speed, trial, obj):
    global x, y, speed_z, sh_
    x=0
    y=0
    z=0
    sh_ = 0
    if sh_ in obj:
        profile(px01, py01, px02, py02, z0, speed, trial)
        x=px02
        y=py02
        z=z0
    sh_ = 1
    if sh_ in obj:
        profile(px11, py11, px12, py12, z1, speed, trial)
        x=px12
        y=py12
        z=z1
    sh_ = 2
    if sh_ in obj:
        profile(px21, py21, px22, py22, z2, speed, trial)
        x=px22
        y=py22
        z=z2
    sh_ = 3
    if sh_ in obj:
        profile(px31, py31, px32, py32, z3, speed, trial)
        x=px32
        y=py32
        z=z3
    sh_ = 4
    if sh_ in obj:
        profile(px41, py41, px42, py42, z4, speed, trial)
        x=px42
        y=py42
        z=z4
    sh_ = 5
    if sh_ in obj:
        profile(px51, py51, px52, py52, z5, speed, trial)
        x=px52
        y=py52
        z=z5
    sh_ = 6
    if sh_ in obj:
        profile(px61, py61, px62, py62, z6, speed, trial)
        x=px62
        y=py62
        z=z6

def trials():
    global active , state_sp , z, speed , trial
    while True: 
        a = input('command:')
        if a =='reset':
            reset_all_safe()
        if a =='pro':
            p.write(indentation(10,5000))
            t = (10 / 5000) * 60
            time.sleep(t)
            p.write(motion(100,50, 5000))
            
        if a =='start':   
            speed = int(input('Speed :'))
            # z =  float(input('Z :'))
            z=1
            trial = int(input('Trial :'))
            shapenum = int(input('Number of shapes :'))
            obj=[i for i in range(shapenum)]
            # obj=[2,3]
            Profile_test(z, speed, trial,obj)

print("In main block")
msg = Queue()
t1 = Thread(target=record)
threads = [t1]
t2 = Thread(target=trials)
threads += [t2]
t1.start()
t2.start()

for tloop in threads:
    tloop.join()
