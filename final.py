import numpy as np
import cv2 as cv
face1=[]
face=''
face2=''
list_y=[]
list_x=[]
total_str=''
a=0
ve=[]
def detect(frame):
    frame = cv.GaussianBlur(frame, (5,5), 0)
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    return hsv

def detect_blue(hsv):
    b_img = cv.inRange(hsv, (104,105,108), (118,255,255)) #BLUE//
    return b_img

def detect_orange(hsv):
    b_img = cv.inRange(hsv, (0,134,221), (11,255,255)) #ORANGE//
    return b_img

def detect_green(hsv):
    b_img = cv.inRange(hsv,(64,141,172),(88,255,255))    #GREEN
    return b_img

def detect_red(hsv):
    b_img = cv.inRange(hsv, (0,178,97), (7,255,216)) #RED
    return b_img

def detect_white(hsv):
    b_img = cv.inRange(hsv, (89,0,166), (130,87,255)) #WHITE //
    return b_img

def detect_yellow(hsv):
    b_img = cv.inRange(hsv, (36,55,226), (91,234,255))  #YELLOW
    return b_img

def detect_contour(b_img):
    contours, hierachy = cv.findContours(b_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    return contours

def color(mau): ##BGR
    if mau=='b':
        return (255,0,0)        #BLUE
    if mau=='g':
        return (0,255,0)        #GREEN
    if mau=='r':
        return (0,0,255)        #RED
    if mau=='w':
        return (255,255,255)    #WHITE
    if mau=='y':
        return (0,255,255)      #YELLOW
    if mau=='o':
        return (0,69,255)       #ORANGE

def add_item(x,y,h,a,w):
    if a=='b':
        face1.append(['b',x,y,(y+h)])
        list_y.append(y+(h/2))
        ve.append([x,y,w,h])
    if a=='g':
        face1.append(['g',x,y,(y+h)])
        list_y.append(y+(h/2))
        ve.append([x,y,w,h])
    if a=='r':
        face1.append(['r',x,y,(y+h)])
        list_y.append(y+(h/2))
        ve.append([x,y,w,h])
    if a=='w':
        face1.append(['w',x,y,(y+h)])
        list_y.append(y+(h/2))
        ve.append([x,y,w,h])
    if a=='y':
        face1.append(['y',x,y,(y+h)])
        list_y.append(y+(h/2))
        ve.append([x,y,w,h])
    if a=='o':
        face1.append(['o',x,y,(y+h)])
        list_y.append(y+(h/2))
        ve.append([x,y,w,h])
    list_x.sort()
    list_y.sort()
    return face1

def remove_list(list_y):
    del list_y[0:2]
    del list_y[1:3]
    del list_y[2:4]
    return list_y

def row(list_y,face1):
    a=[]
    list_x=[]
    str_=''
    for j in face1:
        if j[2]<list_y[0]:
            list_x.append(j)
    # print(list_x)
    for i in list_x:
        face1.remove(i)
        a.append(i[1])
    a.sort()
    # print(a)
    for n in a:
        for m in list_x:
            if n==m[1]:
                str_=str_+m[0]
    del list_y[0]
    return str_


def draw_contour(frame,contours,a):
    for cnt in contours:
        Area = cv.contourArea(cnt)
        if Area > 1100 and Area<3000:
            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(frame,(x,y),(x+w,y+h),color(a),3)
            add_item(x,y,h,a,w)
            # if a=='b':             
            # print('x: ',str(x)+'  y: ',str(y))
            # print('l:  ',str((x+w))+ " w:  ",str((y+h)))
            # print('dientich:  ',(x+w)*(y+h))

def vey(ve):
    a=[]
    for i in ve:
        a.append(i[1]+(i[3]/2))
        a.sort()
    del a[0:2]
    del a[1:3]
    del a[2:4]
    return a


def vex(ve):
    a=[]
    for i in ve:
        a.append(i[0]+(i[2]/2))
        a.sort()
    del a[0:2]
    del a[1:3]
    del a[2:4]
    return a

def l_d(vx,vy):
    a=[]
    a.append((int(vx[0]),int(vy[0])))
    a.append((int(vx[0]),int(vy[2])))
    return a

def l_u(vx,vy):
    a=[]
    a.append((int(vx[0]),int(vy[2])))
    a.append((int(vx[0]),int(vy[0])))
    return a

def r_d(vx,vy):
    a=[]
    a.append((int(vx[2]),int(vy[0])))
    a.append((int(vx[2]),int(vy[2])))
    return a

def r_u(vx,vy):
    a=[]
    a.append((int(vx[2]),int(vy[2])))
    a.append((int(vx[2]),int(vy[0])))
    return a

def b_d(vx,vy):
    a=[]
    a.append((int(vx[1]),int(vy[0])))
    a.append((int(vx[1]),int(vy[2])))
    return a

def b_u(vx,vy):
    a=[]
    a.append((int(vx[1]),int(vy[2])))
    a.append((int(vx[1]),int(vy[0])))
    return a

def l_r1(vx,vy):
    a=[]
    a.append((int(vx[0]),int(vy[0])))
    a.append((int(vx[2]),int(vy[0])))
    return a

def r_l1(vx,vy):
    a=[]
    a.append((int(vx[2]),int(vy[0])))
    a.append((int(vx[0]),int(vy[0])))
    return a

def l_r2(vx,vy):
    a=[]
    a.append((int(vx[0]),int(vy[1])))
    a.append((int(vx[2]),int(vy[1])))
    return a

def r_l2(vx,vy):
    a=[]
    a.append((int(vx[2]),int(vy[1])))
    a.append((int(vx[0]),int(vy[1])))
    return a

def draw_derect(frame,start_point,end_point,color_derect,thickness):
    cv.arrowedLine(frame,start_point,end_point,color_derect,thickness)


def l_r3(vx,vy):
    a=[]
    a.append((int(vx[0]),int(vy[2])))
    a.append((int(vx[2]),int(vy[2])))
    return a

def r_l3(vx,vy):
    a=[]
    a.append((int(vx[2]),int(vy[2])))
    a.append((int(vx[0]),int(vy[2])))
    return a


def draw_derect(frame,start_point,end_point,color_derect,thickness):
    cv.arrowedLine(frame,start_point,end_point,color_derect,thickness)


def l_r3(vx,vy):
    a=[]
    a.append((int(vx[0]),int(vy[2])))
    a.append((int(vx[2]),int(vy[2])))
    return a

def r_l3(vx,vy):
    a=[]
    a.append((int(vx[2]),int(vy[2])))
    a.append((int(vx[0]),int(vy[2])))
    return a

def gcode_2(gcode):
    a=[]
    for i in gcode:
        if i=='U2':
            a.append('U')
            a.append('U')

        if i=='R2':
            a.append('R')
            a.append('R')
        if i=='F2':
            a.append('F')
            a.append('F')
        if i=='D2':
            a.append('D')
            a.append('D')
        if i=='L2':
            a.append('L')
            a.append('L')
        if i=='B2':
            a.append('B')
            a.append('B')
        if i!='U2' and i!='R2' and i!='F2' and i!='D2' and i!='L2'and i!='B2':
            a.append(str(i))
    return a


def gcode_3(gcode):
    a=[]
    for i in gcode:
        if i=="b'":
            a.append("lr3")
            a.append("l'")
            a.append("rl3")
        if i=="b":
            a.append("lr3")
            a.append("l")
            a.append("rl3")
        if i=="B":
            a.append("lr3")
            a.append("L")
            a.append("rl3")
        if i=="B'":
            a.append("lr3")
            a.append("L'")
            a.append("rl3")
        if i=="S":
            a.append("lr3")
            a.append("M'")
            a.append("rl3")
        if i=="S'":
            a.append("lr3")
            a.append("M")
            a.append("rl3")
        if i!="b" and i!="b'" and i!="B" and i!="B'" and i!="S" and i!="S'":
            a.append(str(i))
    return a

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    frame=cv.resize(frame,(500,500))
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    draw_contour(frame,detect_contour(detect_blue(detect(frame))),'b')
    draw_contour(frame,detect_contour(detect_orange(detect(frame))),'o')
    draw_contour(frame,detect_contour(detect_white(detect(frame))),'w')
    draw_contour(frame,detect_contour(detect_green(detect(frame))),'g')
    draw_contour(frame,detect_contour(detect_red(detect(frame))),'r')
    draw_contour(frame,detect_contour(detect_yellow(detect(frame))),'y')
    if len(face1)==9:
        face=row(remove_list(list_y),face1)+row(list_y,face1)+row(list_y,face1)
    else:
        face1.clear()
        list_y.clear()
    if len(face)==9 and face!=face2:
        face2=face
        total_str=total_str+face
        print(total_str) 
    if len(total_str)==54:
        break   
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


from rubik_solver import utils
gcode=utils.solve(total_str, 'Kociemba')
gcode=gcode_2(gcode)
gcode=gcode_3(gcode)
print(gcode)
# gcode.insert(0,gcode[0])

counter=0
ve22=[]
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    frame=cv.resize(frame,(500,500))
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    draw_contour(frame,detect_contour(detect_blue(detect(frame))),'b')
    draw_contour(frame,detect_contour(detect_orange(detect(frame))),'o')
    draw_contour(frame,detect_contour(detect_white(detect(frame))),'w')
    draw_contour(frame,detect_contour(detect_green(detect(frame))),'g')
    draw_contour(frame,detect_contour(detect_red(detect(frame))),'r')
    draw_contour(frame,detect_contour(detect_yellow(detect(frame))),'y')
    if len(face1)==9:
        face=row(remove_list(list_y),face1)+row(list_y,face1)+row(list_y,face1)
    else:
        face1.clear()
        list_y.clear()
    if len(face)==9 and face!=face2:
        counter=counter+1
        face2=face
        if counter==len(gcode):
            print('Finish!!')
            break 
    if len(ve)==9:
        print(ve)
        if gcode[counter]=="U":
            draw_derect(frame,(r_l1(vex(ve),vey(ve))[0][0],r_l1(vex(ve),vey(ve))[0][1]),(r_l1(vex(ve),vey(ve))[1][0],r_l1(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="U'":
            draw_derect(frame,(l_r1(vex(ve),vey(ve))[0][0],l_r1(vex(ve),vey(ve))[0][1]),(l_r1(vex(ve),vey(ve))[1][0],l_r1(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=='L':
            draw_derect(frame,(l_d(vex(ve),vey(ve))[0][0],l_d(vex(ve),vey(ve))[0][1]),(l_d(vex(ve),vey(ve))[1][0],l_d(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="L'":
            draw_derect(frame,(l_u(vex(ve),vey(ve))[0][0],l_u(vex(ve),vey(ve))[0][1]),(l_u(vex(ve),vey(ve))[1][0],l_u(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="R":
            draw_derect(frame,(r_u(vex(ve),vey(ve))[0][0],r_u(vex(ve),vey(ve))[0][1]),(r_u(vex(ve),vey(ve))[1][0],r_u(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="R'":
            draw_derect(frame,(r_d(vex(ve),vey(ve))[0][0],r_d(vex(ve),vey(ve))[0][1]),(r_d(vex(ve),vey(ve))[1][0],r_d(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="D":
            draw_derect(frame,(l_r3(vex(ve),vey(ve))[0][0],l_r3(vex(ve),vey(ve))[0][1]),(l_r3(vex(ve),vey(ve))[1][0],l_r3(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="D'":
            draw_derect(frame,(r_l3(vex(ve),vey(ve))[0][0],r_l3(vex(ve),vey(ve))[0][1]),(r_l3(vex(ve),vey(ve))[1][0],r_l3(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="M":
            draw_derect(frame,(b_d(vex(ve),vey(ve))[0][0],b_d(vex(ve),vey(ve))[0][1]),(b_d(vex(ve),vey(ve))[1][0],b_d(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="M'":
            draw_derect(frame,(b_u(vex(ve),vey(ve))[0][0],b_u(vex(ve),vey(ve))[0][1]),(b_u(vex(ve),vey(ve))[1][0],b_u(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="E":
            draw_derect(frame,(l_r2(vex(ve),vey(ve))[0][0],l_r2(vex(ve),vey(ve))[0][1]),(l_r2(vex(ve),vey(ve))[1][0],l_r2(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="E'":
            draw_derect(frame,(r_l2(vex(ve),vey(ve))[0][0],r_l2(vex(ve),vey(ve))[0][1]),(r_l2(vex(ve),vey(ve))[1][0],r_l2(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="u":
            draw_derect(frame,(r_l1(vex(ve),vey(ve))[0][0],r_l1(vex(ve),vey(ve))[0][1]),(r_l1(vex(ve),vey(ve))[1][0],r_l1(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(r_l2(vex(ve),vey(ve))[0][0],r_l2(vex(ve),vey(ve))[0][1]),(r_l2(vex(ve),vey(ve))[1][0],r_l2(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="l":
            draw_derect(frame,(l_d(vex(ve),vey(ve))[0][0],l_d(vex(ve),vey(ve))[0][1]),(l_d(vex(ve),vey(ve))[1][0],l_d(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(b_d(vex(ve),vey(ve))[0][0],b_d(vex(ve),vey(ve))[0][1]),(b_d(vex(ve),vey(ve))[1][0],b_d(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="r":
            draw_derect(frame,(b_u(vex(ve),vey(ve))[0][0],b_u(vex(ve),vey(ve))[0][1]),(b_u(vex(ve),vey(ve))[1][0],b_u(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(r_u(vex(ve),vey(ve))[0][0],r_u(vex(ve),vey(ve))[0][1]),(r_u(vex(ve),vey(ve))[1][0],r_u(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="d":
            draw_derect(frame,(l_r2(vex(ve),vey(ve))[0][0],l_r2(vex(ve),vey(ve))[0][1]),(l_r2(vex(ve),vey(ve))[1][0],l_r2(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(l_r3(vex(ve),vey(ve))[0][0],l_r3(vex(ve),vey(ve))[0][1]),(l_r3(vex(ve),vey(ve))[1][0],l_r3(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="u'":
            draw_derect(frame,(l_r1(vex(ve),vey(ve))[0][0],l_r1(vex(ve),vey(ve))[0][1]),(l_r1(vex(ve),vey(ve))[1][0],l_r1(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(l_r2(vex(ve),vey(ve))[0][0],l_r2(vex(ve),vey(ve))[0][1]),(l_r2(vex(ve),vey(ve))[1][0],l_r2(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="l'":
            draw_derect(frame,(l_u(vex(ve),vey(ve))[0][0],l_u(vex(ve),vey(ve))[0][1]),(l_u(vex(ve),vey(ve))[1][0],l_u(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(b_u(vex(ve),vey(ve))[0][0],b_u(vex(ve),vey(ve))[0][1]),(b_u(vex(ve),vey(ve))[1][0],b_u(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="r'":
            draw_derect(frame,(r_d(vex(ve),vey(ve))[0][0],r_d(vex(ve),vey(ve))[0][1]),(r_d(vex(ve),vey(ve))[1][0],r_d(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(b_d(vex(ve),vey(ve))[0][0],b_d(vex(ve),vey(ve))[0][1]),(b_d(vex(ve),vey(ve))[1][0],b_d(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="d'":
            draw_derect(frame,(r_l2(vex(ve),vey(ve))[0][0],r_l2(vex(ve),vey(ve))[0][1]),(r_l2(vex(ve),vey(ve))[1][0],r_l2(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(r_l3(vex(ve),vey(ve))[0][0],r_l3(vex(ve),vey(ve))[0][1]),(r_l3(vex(ve),vey(ve))[1][0],r_l3(vex(ve),vey(ve))[1][1]),(0,255,0),3)
        if gcode[counter]=="F'":
            draw_derect(frame,(r_l1(vex(ve),vey(ve))[0][0],r_l1(vex(ve),vey(ve))[0][1]),(r_l1(vex(ve),vey(ve))[1][0],r_l1(vex(ve),vey(ve))[1][1]),(0,69,255),3)
            draw_derect(frame,(l_d(vex(ve),vey(ve))[0][0],l_d(vex(ve),vey(ve))[0][1]),(l_d(vex(ve),vey(ve))[1][0],l_d(vex(ve),vey(ve))[1][1]),(0,69,255),3)
        if gcode[counter]=="F":
            draw_derect(frame,(l_r1(vex(ve),vey(ve))[0][0],l_r1(vex(ve),vey(ve))[0][1]),(l_r1(vex(ve),vey(ve))[1][0],l_r1(vex(ve),vey(ve))[1][1]),(0,69,255),3)
            draw_derect(frame,(r_d(vex(ve),vey(ve))[0][0],r_d(vex(ve),vey(ve))[0][1]),(r_d(vex(ve),vey(ve))[1][0],r_d(vex(ve),vey(ve))[1][1]),(0,69,255),3)
        if gcode[counter]=="f'":
            draw_derect(frame,(b_d(vex(ve),vey(ve))[0][0],b_d(vex(ve),vey(ve))[0][1]),(b_d(vex(ve),vey(ve))[1][0],b_d(vex(ve),vey(ve))[1][1]),(0,0,0),5)
            draw_derect(frame,(b_u(vex(ve),vey(ve))[0][0],b_u(vex(ve),vey(ve))[0][1]),(b_u(vex(ve),vey(ve))[1][0],b_u(vex(ve),vey(ve))[1][1]),(0,0,0),5)
            draw_derect(frame,(r_d(vex(ve),vey(ve))[0][0],r_d(vex(ve),vey(ve))[0][1]),(r_d(vex(ve),vey(ve))[1][0],r_d(vex(ve),vey(ve))[1][1]),(0,0,0),5)
            draw_derect(frame,(r_u(vex(ve),vey(ve))[0][0],r_u(vex(ve),vey(ve))[0][1]),(r_u(vex(ve),vey(ve))[1][0],r_u(vex(ve),vey(ve))[1][1]),(0,0,0),5)
            draw_derect(frame,(r_l1(vex(ve),vey(ve))[0][0],r_l1(vex(ve),vey(ve))[0][1]),(r_l1(vex(ve),vey(ve))[1][0],r_l1(vex(ve),vey(ve))[1][1]),(0,255,0),5)
            draw_derect(frame,(l_d(vex(ve),vey(ve))[0][0],l_d(vex(ve),vey(ve))[0][1]),(l_d(vex(ve),vey(ve))[1][0],l_d(vex(ve),vey(ve))[1][1]),(0,255,0),5)
        if gcode[counter]=="f":
            draw_derect(frame,(b_d(vex(ve),vey(ve))[0][0],b_d(vex(ve),vey(ve))[0][1]),(b_d(vex(ve),vey(ve))[1][0],b_d(vex(ve),vey(ve))[1][1]),(0,0,0),5)
            draw_derect(frame,(b_u(vex(ve),vey(ve))[0][0],b_u(vex(ve),vey(ve))[0][1]),(b_u(vex(ve),vey(ve))[1][0],b_u(vex(ve),vey(ve))[1][1]),(0,0,0),5)
            draw_derect(frame,(l_d(vex(ve),vey(ve))[0][0],l_d(vex(ve),vey(ve))[0][1]),(l_d(vex(ve),vey(ve))[1][0],l_d(vex(ve),vey(ve))[1][1]),(0,0,0),5)
            draw_derect(frame,(l_u(vex(ve),vey(ve))[0][0],l_u(vex(ve),vey(ve))[0][1]),(l_u(vex(ve),vey(ve))[1][0],l_u(vex(ve),vey(ve))[1][1]),(0,0,0),5)
            draw_derect(frame,(l_r1(vex(ve),vey(ve))[0][0],l_r1(vex(ve),vey(ve))[0][1]),(l_r1(vex(ve),vey(ve))[1][0],l_r1(vex(ve),vey(ve))[1][1]),(0,255,0),5)
            draw_derect(frame,(r_d(vex(ve),vey(ve))[0][0],r_d(vex(ve),vey(ve))[0][1]),(r_d(vex(ve),vey(ve))[1][0],r_d(vex(ve),vey(ve))[1][1]),(0,255,0),5)

        if gcode[counter]=="lr3":
            draw_derect(frame,(l_r1(vex(ve),vey(ve))[0][0],l_r1(vex(ve),vey(ve))[0][1]),(l_r1(vex(ve),vey(ve))[1][0],l_r1(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(l_r2(vex(ve),vey(ve))[0][0],l_r2(vex(ve),vey(ve))[0][1]),(l_r2(vex(ve),vey(ve))[1][0],l_r2(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(l_r3(vex(ve),vey(ve))[0][0],l_r3(vex(ve),vey(ve))[0][1]),(l_r3(vex(ve),vey(ve))[1][0],l_r3(vex(ve),vey(ve))[1][1]),(0,255,0),3)

        if gcode[counter]=="rl3":
            draw_derect(frame,(r_l1(vex(ve),vey(ve))[0][0],r_l1(vex(ve),vey(ve))[0][1]),(r_l1(vex(ve),vey(ve))[1][0],r_l1(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(r_l2(vex(ve),vey(ve))[0][0],r_l2(vex(ve),vey(ve))[0][1]),(r_l2(vex(ve),vey(ve))[1][0],r_l2(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(r_l3(vex(ve),vey(ve))[0][0],r_l3(vex(ve),vey(ve))[0][1]),(r_l3(vex(ve),vey(ve))[1][0],r_l3(vex(ve),vey(ve))[1][1]),(0,255,0),3)


        if gcode[counter]=="d3":
            draw_derect(frame,(l_d(vex(ve),vey(ve))[0][0],l_d(vex(ve),vey(ve))[0][1]),(l_d(vex(ve),vey(ve))[1][0],l_d(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(r_d(vex(ve),vey(ve))[0][0],r_d(vex(ve),vey(ve))[0][1]),(r_d(vex(ve),vey(ve))[1][0],r_d(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(b_d(vex(ve),vey(ve))[0][0],b_d(vex(ve),vey(ve))[0][1]),(b_d(vex(ve),vey(ve))[1][0],b_d(vex(ve),vey(ve))[1][1]),(0,255,0),3)

        if gcode[counter]=="u3":
            draw_derect(frame,(l_u(vex(ve),vey(ve))[0][0],l_u(vex(ve),vey(ve))[0][1]),(l_u(vex(ve),vey(ve))[1][0],l_u(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(r_u(vex(ve),vey(ve))[0][0],r_u(vex(ve),vey(ve))[0][1]),(r_u(vex(ve),vey(ve))[1][0],r_u(vex(ve),vey(ve))[1][1]),(0,255,0),3)
            draw_derect(frame,(b_u(vex(ve),vey(ve))[0][0],b_u(vex(ve),vey(ve))[0][1]),(b_u(vex(ve),vey(ve))[1][0],b_u(vex(ve),vey(ve))[1][1]),(0,255,0),3)
    ve.clear()
    print(gcode[counter])
    print(ve)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()        
