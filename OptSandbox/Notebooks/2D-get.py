# -*- coding: UTF-8 -*-
import time
import os
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import part
import time
import random
import string
executeOnCaeStartup()

#Parameters

folder_path = os.getcwd()

height0txtpath = "{}/a_Design_var1.txt".format(folder_path)
rstxtpath      = "{}/a_Design_var2.txt".format(folder_path)

height0=float(open(height0txtpath, "r").read().strip())
rs     =float(open(rstxtpath, "r").read().strip())

Job_name='Job-pad-2D'

# height0 = 34.05     # max height
dis_r = 2.11  # radius in top
height1 = 31.90     # heigth except the top ben area
height2 = 0.5      # incline
# rs = 80.0          # R radius in middle
z_num = 16         # element layers in bottom
kuan0,kuan1= 34.5 , 38.046    # length in y [upper, bottom]
mesh_size = 0.4  # mesh size
cpu_num = 12 # cpus cores used
open_path="" 
open_name_init="2D-init.cae"
open_name_model="2D-get-0.cae"
save_name="2D-get-out-0.cae"
output_r='output-r.txt'      #

# =============================================================================
# output_gap_cm4_cg='b_Objective_cm4_cg'+str(iteration)+'.txt'      #
# output_gap_cm4_oled='b_Objective_cm4_oled'+str(iteration)+'.txt'      #
# 
# output_gap_cm3_cg='b_Objective_cm3_cg'+str(iteration)+'.txt'      #
# output_gap_cm3_oled='b_Objective_cm3_oled'+str(iteration)+'.txt'      #
# 
# output_gap_cm2_cg='b_Objective_cm2_cg'+str(iteration)+'.txt'      #
# output_gap_cm2_oled='b_Objective_cm2_oled'+str(iteration)+'.txt'      #
# 
# output_gap_cm1_cg='b_Objective_cm1_cg'+str(iteration)+'.txt'      #
# output_gap_cm1_oled='b_Objective_cm1_oled'+str(iteration)+'.txt'      #
# 
# output_gap_c_cg='b_Objective_c_cg'+str(iteration)+'.txt'      #
# output_gap_c_oled='b_Objective_c_oled'+str(iteration)+'.txt'      #
# 
# output_gap_cp1_cg='b_Objective_cp1_cg'+str(iteration)+'.txt'      #
# output_gap_cp1_oled='b_Objective_cp1_oled'+str(iteration)+'.txt'      #
# 
# output_gap_cp2_cg='b_Objective_cp2_cg'+str(iteration)+'.txt'      #
# output_gap_cp2_oled='b_Objective_cp2_oled'+str(iteration)+'.txt'      #
# =============================================================================

output_gap='b_Objective_c_gap.txt'      #

time_init=time.time()
z_inc=height1/(z_num+1.0/3)
tole=mesh_size/100.0

def r_loc(loc1,loc2,rs):   #get cicle center
    xo1,yo1,ro1=loc1[0],loc1[1],rs
    xo2,yo2,ro2=loc2[0],loc2[1],rs
    d=sqrt((xo1-xo2)**2+(yo1-yo2)**2)
    Ao=(ro1**2-ro2**2+d**2)/(2*d)
    h=sqrt(ro1**2-Ao**2)
    xo3=xo1+Ao*(xo2-xo1)/d
    yo3=yo1+Ao*(yo2-yo1)/d
    xn1=xo3-h*(yo2-yo1)/d
    yn1=yo3+h*(xo2-xo1)/d
    xn2=xo3+h*(yo2-yo1)/d
    yn2=yo3-h*(xo2-xo1)/d
    if xn1>xn2:
        return [xn1,yn1]
    else:
        return [xn2,yn2]
def line1_2D(z0):  #side lines
    x0=loc_rx[0]-sqrt(rs**2-(z0-loc_rx[1])**2)
    return x0
def get_coord(nodesc1):  # central point
    sumx,sumy,sumz=0.0,0.0,0.0
    lennode=len(nodesc1)
    for J in range(0,lennode):
        sumx=sumx+nodesc1[J].coordinates[0]/lennode
        sumy=sumy+nodesc1[J].coordinates[1]/lennode
        sumz=sumz+nodesc1[J].coordinates[2]/lennode
    return [sumx,sumy,sumz]
def sort_label(nodesc):  # sort 
    xy_sum=[sum(n1.getFromLabel(node2).coordinates) for node2 in nodesc]
    for J in range(0,len(nodesc)-1):
        for I in range(1,len(nodesc)-J):
            if xy_sum[I-1]>xy_sum[I]:
                xy_sum[I],xy_sum[I-1]=xy_sum[I-1],xy_sum[I]
                nodesc[I],nodesc[I-1]=nodesc[I-1],nodesc[I]
    return nodesc
def up_line_2D(y_loc): #up line from y to z
    return -height2/(kuan0-dis_r)**2*y_loc**2
def create_element_2D(ns,nsl,ele_type):
    if ele_type.getText()=='QUAD4':
        return p.Element(nodes=(n1.getFromLabel(ns[nsl[0]]), n1.getFromLabel(ns[nsl[1]]), n1.getFromLabel(ns[nsl[2]]), n1.getFromLabel(ns[nsl[3]])), elemShape=ele_type).label
def get_circle(locs_r):  #center and radius
    m_nNum=len(locs_r)
    if m_nNum<3:
        return
    X1,Y1,X2,Y2,X3,Y3,X1Y1,X1Y2,X2Y1=0,0,0,0,0,0,0,0,0
    for i in range(0,m_nNum):
        X1 = X1 + locs_r[i][0];
        Y1 = Y1 + locs_r[i][1];
        X2 = X2 + locs_r[i][0]*locs_r[i][0];
        Y2 = Y2 + locs_r[i][1]*locs_r[i][1];
        X3 = X3 + locs_r[i][0]*locs_r[i][0]*locs_r[i][0];
        Y3 = Y3 + locs_r[i][1]*locs_r[i][1]*locs_r[i][1];
        X1Y1 = X1Y1 + locs_r[i][0]*locs_r[i][1];
        X1Y2 = X1Y2 + locs_r[i][0]*locs_r[i][1]*locs_r[i][1];
        X2Y1 = X2Y1 + locs_r[i][0]*locs_r[i][0]*locs_r[i][1];
    N = m_nNum;
    C = N*X2 - X1*X1;
    D = N*X1Y1 - X1*Y1;
    E = N*X3 + N*X1Y2 - (X2+Y2)*X1;
    G = N*Y2 - Y1*Y1;
    H = N*X2Y1 + N*Y3 - (X2+Y2)*Y1;
    a = (H*D-E*G)/(C*G-D*D);
    b = (H*C-E*D)/(D*D-G*C);
    c = -(a*X1 + b*Y1 + X2 + Y2)/N;
 
    A = a/(-2);
    B = b/(-2);
    R = sqrt(a*a+b*b-4*c)/2;
    return [(A,B),R]

def sort_point(loc_cg):  
    xy_sum=[loc1[0]-loc1[1] for loc1 in loc_cg]
    for J in range(0,len(loc_cg)-1):
        for I in range(1,len(loc_cg)-J):
            if xy_sum[I-1]>xy_sum[I]:
                xy_sum[I],xy_sum[I-1]=xy_sum[I-1],xy_sum[I]
                loc_cg[I],loc_cg[I-1]=loc_cg[I-1],loc_cg[I]
    return loc_cg
def dist_p(p1_loc,p2_loc,p0_loc):
    x1,y1=p1_loc[0],p1_loc[1]
    x2,y2=p2_loc[0],p2_loc[1]
    x0,y0=p0_loc[0],p0_loc[1]
    return (abs((y2-y1)*x0+(x1-x2)*y0+((x2*y1)-(x1*y2))))/(sqrt((y2-y1)**2+(x1-x2)**2))
def ranstr(num):  
    salt = ''.join(random.sample(string.ascii_letters + string.digits, num))
    return salt

loc_rx=r_loc([kuan1,0],[kuan0,height1],rs) 
openMdb(open_path+open_name_init)

#stetch
p = mdb.models['Model-1'].parts['Part-init']
s = p.features['Shell planar-1'].sketch
mdb.models['Model-1'].ConstrainedSketch(name='__edit__', objectToCopy=s)
s1 = mdb.models['Model-1'].sketches['__edit__']
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=SUPERIMPOSE)
p.projectReferencesOntoSketch(sketch=s1, upToFeature=p.features['Shell planar-1'], filter=COPLANAR_EDGES)
d[3].setValues(value=height0, )
d[2].setValues(value=height1, )
d[0].setValues(value=dis_r, )
d[5].setValues(value=kuan0-dis_r, )
s1.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['Part-init']
p.features['Shell planar-1'].setValues(sketch=s1)
del mdb.models['Model-1'].sketches['__edit__']
p.regenerate()
#devide
f, e, d = p.faces, p.edges, p.datums
t = p.MakeSketchTransform(sketchPlane=f[0], sketchPlaneSide=SIDE1, origin=(0.0,0.0,0.0))
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=98.31, gridSpacing=2.45, transform=t)
g, v, d1, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=SUPERIMPOSE)
p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
s.Line(point1=(kuan0-dis_r, 500.0), point2=(kuan0-dis_r,-100.0))
f = p.faces
pickedFaces = f.getSequenceFromMask(mask=('[#1 ]', ), )
e1, d2 = p.edges, p.datums
p.PartitionFaceBySketch(faces=pickedFaces, sketch=s)
s.unsetPrimaryObject()
del mdb.models['Model-1'].sketches['__profile__']
p.regenerate()
mdb.models['Model-1'].parts['Part-init'].writeAcisFile('Part-init.sat', 24)

#model
openMdb(open_path+open_name_model)
acis = mdb.openAcis('Part-init.sat', scaleFromFile=OFF)
mdb.models['PAD_2D'].PartFromGeometryFile(name='Part-init', geometryFile=acis, 
    combine=False, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
p = mdb.models['PAD_2D'].parts['Part-init']
#2D mesh
f = p.faces
pickedRegions = f.getByBoundingBox(xMin=-tole,xMax=1000.0,yMin=-tole,yMax=1000.0,zMin=-tole,zMax=tole)
p.setMeshControls(regions=pickedRegions, elemShape=QUAD, technique=STRUCTURED)
p.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)    ###
p.generateMesh()
e=p.elements

# size
n1=p.nodes
node_find=n1.getByBoundingBox(xMin=-tole,xMax=tole,yMin=height1-tole,yMax=height1+tole,zMin=-tole,zMax=tole)
node_start=node_find[0]
e_connect=node_start.getElements()[0]
e_nodes=e_connect.getNodes()
x_inc=get_coord(e_nodes)[0]
x_in_len=kuan0-dis_r
x_ele_num=int(round(x_in_len/x_inc/2.0,0))  
x_ele_get=x_ele_num/3-1
x_no_ele=x_ele_get*x_inc*6

p.PartFromMesh(name='Part-2', copySets=True)
p = mdb.models['PAD_2D'].parts['Part-2']
e1 = p.elements

n1=p.nodes
side_nodes=n1.getByBoundingBox(xMin=-tole,xMax=1000.0,yMin=height1-tole,yMax=height1+tole,zMin=-tole,zMax=tole)
side_nodes_label=[ v.label for v in side_nodes]
side_nodes_label=sort_label(side_nodes_label)
#first layer
side_nodes=n1.sequenceFromLabels(side_nodes_label)
side_nodes_label_new=[]
for I in range(0,len(side_nodes_label)):
    locs=side_nodes[I].coordinates
    side_nodes_label_new.append(p.Node(coordinates=(locs[0], locs[1]-z_inc/3, 0.0)).label)
    if I == 0:
        continue
    ns=[side_nodes_label_new[I],side_nodes_label[I],side_nodes_label[I-1],side_nodes_label_new[I-1]]
    ele_build=create_element_2D(ns,[0,1,2,3],QUAD4)

#second layer
side_nodes_label=side_nodes_label_new
side_nodes_label_new=[]
side_nodes=n1.sequenceFromLabels(side_nodes_label)
I,len1=0,len(side_nodes_label)
for K in range(0,x_ele_get):
    locs_right=side_nodes[I+3].coordinates
    ns=[side_nodes_label[I],side_nodes_label[I+1],side_nodes_label[I+2],side_nodes_label[I+3]]
    if locs_right[0]<x_no_ele+tole:  
        ns=[side_nodes_label[I],side_nodes_label[I+1],side_nodes_label[I+2],side_nodes_label[I+3]]
        locs_0=side_nodes[I].coordinates
        locs_1=side_nodes[I+1].coordinates
        locs_2=side_nodes[I+2].coordinates
        locs_3=side_nodes[I+3].coordinates
        ns.append(p.Node(coordinates=(locs_1[0], locs_1[1]-z_inc*2/3, 0.0)).label)
        ns.append(p.Node(coordinates=(locs_2[0], locs_2[1]-z_inc*2/3, 0.0)).label)
        if I==0:  
            ns.append(p.Node(coordinates=(locs_0[0], locs_0[1]-z_inc, 0.0)).label)
            side_nodes_label_new.append(ns[6])
        else:
            ns.append(side_nodes_label_new[-1])
        ns.append(p.Node(coordinates=(locs_3[0], locs_0[1]-z_inc, 0.0)).label)
        ele_build=create_element_2D(ns,[4,1,0,6],QUAD4)
        ele_build=create_element_2D(ns,[5,2,1,4],QUAD4)
        ele_build=create_element_2D(ns,[7,3,2,5],QUAD4)
        ele_build=create_element_2D(ns,[7,5,4,6],QUAD4)
        side_nodes_label_new.append(ns[7])
        I=I+3
for J in range(I+1,len1):
    locs=side_nodes[J].coordinates
    side_nodes_label_new.append(p.Node(coordinates=(locs[0], locs[1]-z_inc, 0.0)).label)
    ns=[side_nodes_label_new[-1],side_nodes_label[J],side_nodes_label[J-1],side_nodes_label_new[-2]]
    ele_build=create_element_2D(ns,[0,1,2,3],QUAD4)

#bot layer
for K in range(0,z_num-1):
    side_nodes_label=side_nodes_label_new
    side_nodes_label_new=[]
    side_nodes=n1.sequenceFromLabels(side_nodes_label)
    for I in range(0,len(side_nodes_label)):
        locs=side_nodes[I].coordinates
        side_nodes_label_new.append(p.Node(coordinates=(locs[0], locs[1]-z_inc, 0.0)).label)
        if I == 0:
            continue
        ns=[side_nodes_label_new[I],side_nodes_label[I],side_nodes_label[I-1],side_nodes_label_new[I-1]]
        ele_build=create_element_2D(ns,[0,1,2,3],QUAD4)

# post prepare
a = mdb.models['PAD_2D'].rootAssembly
a.features['PART-1-1'].suppress()
a.Instance(name='Part-2-1', part=p, dependent=ON)
n1 = a.instances['BOT-1'].nodes
nodes1a = n1.getByBoundingBox(xMin=-tole,xMax=1000.0,yMin=-tole,yMax=tole,zMin=-tole,zMax=tole)
nodes1b = n1.getByBoundingBox(xMin=-tole,xMax=tole,yMin=-tole,yMax=1000.0,zMin=-tole,zMax=tole)
n2 = a.instances['PANEL-1'].nodes
nodes2 = n2.getByBoundingBox(xMin=-tole,xMax=tole,yMin=-tole,yMax=1000.0,zMin=-tole,zMax=tole)
n3 = a.instances['Part-2-1'].nodes
nodes3 = n3.getByBoundingBox(xMin=-tole,xMax=tole,yMin=-tole,yMax=1000.0,zMin=-tole,zMax=tole)
a.Set(nodes=nodes1a+nodes1b+nodes2+nodes3, name='FIX_X_NODE')
#R area
n1=p.nodes
nodes_r_all=n1.getByBoundingBox(xMin=kuan0-dis_r-tole,xMax=1000.0,yMin=height0-dis_r-tole,yMax=1000.0,zMin=-tole,zMax=tole)
nodes_r_label=[]
for node0 in nodes_r_all:
    locs=node0.coordinates
    dist=sqrt((locs[0]-kuan0+dis_r)**2+(locs[1]-height0+dis_r)**2)
    if abs(dist-dis_r)<=tole:
        nodes_r_label.append(node0.label)

p.Set(nodes=n1.sequenceFromLabels(nodes_r_label),name='SET-R-AREA')

#bottom fix
a = mdb.models['PAD_2D'].rootAssembly
a.regenerate()
n1 = a.instances['Part-2-1'].nodes
nodes1 = n1.getByBoundingBox(xMin=-tole,xMax=1000.0,yMin=-tole,yMax=tole,zMin=-tole,zMax=tole)
a.Set(nodes=nodes1, name='Set-pad-bot')
region = a.sets['Set-pad-bot']
mdb.models['PAD_2D'].DisplacementBC(name='BC-pad-bot', 
    createStepName='Initial', region=region, u1=SET, u2=UNSET, ur3=UNSET, 
    amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)

# geometry
n1=p.nodes
for node0 in n1:
    cood0=node0.coordinates
    if cood0[1]<height1:
        x_now1=line1_2D(cood0[1])
        x_scale=x_now1/kuan0
        xn=cood0[0]*x_scale
        yn=cood0[1]+up_line_2D(cood0[0])*cood0[1]/height1
        nodes0=n1.sequenceFromLabels([node0.label])
        p.editNode(nodes=nodes0, coordinate1=xn, coordinate2=yn)
    else:
        z_dec=up_line_2D(cood0[0]) 
        nodes0=n1.sequenceFromLabels([node0.label])
        p.editNode(nodes=nodes0, offset2=z_dec)

a.regenerate
p = mdb.models['PAD_2D'].parts['Part-2']
session.viewports['Viewport: 1'].setValues(displayedObject=p)

#mesh mat
p = mdb.models['PAD_2D'].parts['Part-2']
elemType1 = mesh.ElemType(elemCode=CPE4R, elemLibrary=STANDARD, 
    secondOrderAccuracy=OFF, hourglassControl=DEFAULT, distortionControl=DEFAULT)
e1 = p.elements
pickedRegions =(e1, )
p.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))

region = p.Set(elements=e1, name='Set-all')
p.SectionAssignment(region=region, sectionName='Section-16-PAD_MOD15', 
    offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

#save
mdb.saveAs(pathName=open_path+save_name)

a = mdb.models['PAD_2D'].rootAssembly
job1=mdb.Job(name=Job_name, model='PAD_2D', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=cpu_num, 
    numDomains=cpu_num, numGPUs=0)
mdb.jobs[Job_name].writeInput(consistencyChecking=OFF)
#submit
os.system("abaqus job="+Job_name+" cpus="+str(cpu_num)+" int")

#post
from odbAccess import *
from abaqusConstants import *
import string
o3 = session.openOdb(name=open_path+Job_name+'.odb')
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
odb = session.odbs[Job_name+'.odb']

## R radius
f=open(open_path+output_r,'w')
xy_r=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("PART-2-1.SET-R-AREA", ))
locs_r=[ (0,0) for i in range(0,len(xy_r)/2)]
time_r=[ p_r[0] for p_r in xy_r[0] ]
node_rc=len(xy_r)/2
for I in range(0,len(time_r)):
    locs_r=[(xy_r[xyi][I][1],xy_r[xyi+node_rc][I][1]) for xyi in range(0,node_rc)]
    [loc_center,loc_radius]=get_circle(locs_r)
    f.write(str(time_r[I])+'\t'+str(loc_center)+'\t'+str(loc_radius)+'\n')
f.close

# =============================================================================
# ##Gap-4
# xy_contact=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('CPRESS   General_Contact_Domain', NODAL), ), nodeSets=("SET-CONTACT", ))
# contact_time=1000.0
# for frame_c in range(0,len(xy_contact[0])):
#     press_c=sum([p_c[frame_c][1] for p_c in xy_contact])
#     if press_c>0.0:
#         contact_time=time_r[frame_c]
#         break
# xy_cg=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-CG", ))
# xy_panel=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-PANEL", ))
# len_cg,len_panel=len(xy_cg)/2,len(xy_panel)/2
# loc_cg=[(xy_cg[xyi][frame_c-4][1],xy_cg[xyi+len_cg][frame_c-4][1]) for xyi in range(0,len_cg)]
# loc_panel_0=[(xy_panel[xyi][frame_c-4][1],xy_panel[xyi+len_panel][frame_c-4][1]) for xyi in range(0,len_panel)]
# min_x_cg=min([p_cg[0] for p_cg in loc_cg])
# loc_panel=[p_panel for p_panel in loc_panel_0 if p_panel[0]>min_x_cg]
# loc_cg=sort_point(loc_cg)
# loc_panel=sort_point(loc_panel)
# distance_panel=[0 for loc1 in loc_panel]  
# dist_temp=[0 for loc1 in range(0,len_cg-1)] 
# 
# f=open(open_path+output_gap_cm4_cg,'w')
# for I in range(0,len(loc_cg)):
#     f.write(str(loc_cg[I])+'\n')
# f.close
# f=open(open_path+output_gap_cm4_oled,'w')
# for J in range(0,len(loc_panel)):
#     f.write(str(loc_panel[J])+'\n')
# f.close
# 
# 
# ##Gap-3
# xy_contact=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('CPRESS   General_Contact_Domain', NODAL), ), nodeSets=("SET-CONTACT", ))
# contact_time=1000.0
# for frame_c in range(0,len(xy_contact[0])):
#     press_c=sum([p_c[frame_c][1] for p_c in xy_contact])
#     if press_c>0.0:
#         contact_time=time_r[frame_c]
#         break
# xy_cg=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-CG", ))
# xy_panel=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-PANEL", ))
# len_cg,len_panel=len(xy_cg)/2,len(xy_panel)/2
# loc_cg=[(xy_cg[xyi][frame_c-3][1],xy_cg[xyi+len_cg][frame_c-3][1]) for xyi in range(0,len_cg)]
# loc_panel_0=[(xy_panel[xyi][frame_c-3][1],xy_panel[xyi+len_panel][frame_c-3][1]) for xyi in range(0,len_panel)]
# min_x_cg=min([p_cg[0] for p_cg in loc_cg])
# loc_panel=[p_panel for p_panel in loc_panel_0 if p_panel[0]>min_x_cg]
# loc_cg=sort_point(loc_cg)
# loc_panel=sort_point(loc_panel)
# distance_panel=[0 for loc1 in loc_panel]  
# dist_temp=[0 for loc1 in range(0,len_cg-1)] 
# 
# f=open(open_path+output_gap_cm3_cg,'w')
# for I in range(0,len(loc_cg)):
#     f.write(str(loc_cg[I])+'\n')
# f.close
# f=open(open_path+output_gap_cm3_oled,'w')
# for J in range(0,len(loc_panel)):
#     f.write(str(loc_panel[J])+'\n')
# f.close
# 
# ##Gap-2
# xy_contact=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('CPRESS   General_Contact_Domain', NODAL), ), nodeSets=("SET-CONTACT", ))
# contact_time=1000.0
# for frame_c in range(0,len(xy_contact[0])):
#     press_c=sum([p_c[frame_c][1] for p_c in xy_contact])
#     if press_c>0.0:
#         contact_time=time_r[frame_c]
#         break
# xy_cg=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-CG", ))
# xy_panel=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-PANEL", ))
# len_cg,len_panel=len(xy_cg)/2,len(xy_panel)/2
# loc_cg=[(xy_cg[xyi][frame_c-2][1],xy_cg[xyi+len_cg][frame_c-2][1]) for xyi in range(0,len_cg)]
# loc_panel_0=[(xy_panel[xyi][frame_c-2][1],xy_panel[xyi+len_panel][frame_c-2][1]) for xyi in range(0,len_panel)]
# min_x_cg=min([p_cg[0] for p_cg in loc_cg])
# loc_panel=[p_panel for p_panel in loc_panel_0 if p_panel[0]>min_x_cg]
# loc_cg=sort_point(loc_cg)
# loc_panel=sort_point(loc_panel)
# distance_panel=[0 for loc1 in loc_panel]  
# dist_temp=[0 for loc1 in range(0,len_cg-1)] 
# 
# f=open(open_path+output_gap_cm2_cg,'w')
# for I in range(0,len(loc_cg)):
#     f.write(str(loc_cg[I])+'\n')
# f.close
# f=open(open_path+output_gap_cm2_oled,'w')
# for J in range(0,len(loc_panel)):
#     f.write(str(loc_panel[J])+'\n')
# f.close
# 
# ##Gap-1
# xy_contact=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('CPRESS   General_Contact_Domain', NODAL), ), nodeSets=("SET-CONTACT", ))
# contact_time=1000.0
# for frame_c in range(0,len(xy_contact[0])):
#     press_c=sum([p_c[frame_c][1] for p_c in xy_contact])
#     if press_c>0.0:
#         contact_time=time_r[frame_c]
#         break
# xy_cg=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-CG", ))
# xy_panel=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-PANEL", ))
# len_cg,len_panel=len(xy_cg)/2,len(xy_panel)/2
# loc_cg=[(xy_cg[xyi][frame_c-1][1],xy_cg[xyi+len_cg][frame_c-1][1]) for xyi in range(0,len_cg)]
# loc_panel_0=[(xy_panel[xyi][frame_c-1][1],xy_panel[xyi+len_panel][frame_c-1][1]) for xyi in range(0,len_panel)]
# min_x_cg=min([p_cg[0] for p_cg in loc_cg])
# loc_panel=[p_panel for p_panel in loc_panel_0 if p_panel[0]>min_x_cg]
# loc_cg=sort_point(loc_cg)
# loc_panel=sort_point(loc_panel)
# distance_panel=[0 for loc1 in loc_panel]  
# dist_temp=[0 for loc1 in range(0,len_cg-1)] 
# 
# f=open(open_path+output_gap_cm1_cg,'w')
# for I in range(0,len(loc_cg)):
#     f.write(str(loc_cg[I])+'\n')
# f.close
# f=open(open_path+output_gap_cm1_oled,'w')
# for J in range(0,len(loc_panel)):
#     f.write(str(loc_panel[J])+'\n')
# f.close
# =============================================================================

##Gap-0
xy_contact=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('CPRESS   General_Contact_Domain', NODAL), ), nodeSets=("SET-CONTACT", ))
contact_time=1000.0
for frame_c in range(0,len(xy_contact[0])):
    press_c=sum([p_c[frame_c][1] for p_c in xy_contact])
    if press_c>0.0:
        contact_time=time_r[frame_c]
        break
xy_cg=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-CG", ))
xy_panel=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-PANEL", ))
len_cg,len_panel=len(xy_cg)/2,len(xy_panel)/2
loc_cg=[(xy_cg[xyi][frame_c][1],xy_cg[xyi+len_cg][frame_c][1]) for xyi in range(0,len_cg)]
loc_panel_0=[(xy_panel[xyi][frame_c][1],xy_panel[xyi+len_panel][frame_c][1]) for xyi in range(0,len_panel)]
min_x_cg=min([p_cg[0] for p_cg in loc_cg])
loc_panel=[p_panel for p_panel in loc_panel_0 if p_panel[0]>min_x_cg]
loc_cg=sort_point(loc_cg)
loc_panel=sort_point(loc_panel)
distance_panel=[0 for loc1 in loc_panel]  
dist_temp=[0 for loc1 in range(0,len_cg-1)] 

# =============================================================================
# f=open(open_path+output_gap_c_cg,'w')
# for I in range(0,len(loc_cg)):
#     f.write(str(loc_cg[I])+'\n')
# f.close
# f=open(open_path+output_gap_c_oled,'w')
# for J in range(0,len(loc_panel)):
#     f.write(str(loc_panel[J])+'\n')
# f.close
# =============================================================================

f=open(open_path+output_gap,'w')
for J in range(0,len(loc_panel)):
    for I in range(0,len(loc_cg)-1):
        dist_temp[I]=dist_p(loc_cg[I],loc_cg[I+1],loc_panel[J])
        distance_panel[J]=min(dist_temp)
f.write(str(max(distance_panel))+'\n')
f.close

# =============================================================================
# ##Gap+1
# xy_contact=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('CPRESS   General_Contact_Domain', NODAL), ), nodeSets=("SET-CONTACT", ))
# contact_time=1000.0
# for frame_c in range(0,len(xy_contact[0])):
#     press_c=sum([p_c[frame_c][1] for p_c in xy_contact])
#     if press_c>0.0:
#         contact_time=time_r[frame_c]
#         break
# xy_cg=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-CG", ))
# xy_panel=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-PANEL", ))
# len_cg,len_panel=len(xy_cg)/2,len(xy_panel)/2
# loc_cg=[(xy_cg[xyi][frame_c+1][1],xy_cg[xyi+len_cg][frame_c+1][1]) for xyi in range(0,len_cg)]
# loc_panel_0=[(xy_panel[xyi][frame_c+1][1],xy_panel[xyi+len_panel][frame_c+1][1]) for xyi in range(0,len_panel)]
# min_x_cg=min([p_cg[0] for p_cg in loc_cg])
# loc_panel=[p_panel for p_panel in loc_panel_0 if p_panel[0]>min_x_cg]
# loc_cg=sort_point(loc_cg)
# loc_panel=sort_point(loc_panel)
# distance_panel=[0 for loc1 in loc_panel]  
# dist_temp=[0 for loc1 in range(0,len_cg-1)] 
# 
# f=open(open_path+output_gap_cp1_cg,'w')
# for I in range(0,len(loc_cg)):
#     f.write(str(loc_cg[I])+'\n')
# f.close
# f=open(open_path+output_gap_cp1_oled,'w')
# for J in range(0,len(loc_panel)):
#     f.write(str(loc_panel[J])+'\n')
# f.close
# 
# ##Gap+2
# xy_contact=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('CPRESS   General_Contact_Domain', NODAL), ), nodeSets=("SET-CONTACT", ))
# contact_time=1000.0
# for frame_c in range(0,len(xy_contact[0])):
#     press_c=sum([p_c[frame_c][1] for p_c in xy_contact])
#     if press_c>0.0:
#         contact_time=time_r[frame_c]
#         break
# xy_cg=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-CG", ))
# xy_panel=session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), )), ), nodeSets=("SET-C-PANEL", ))
# len_cg,len_panel=len(xy_cg)/2,len(xy_panel)/2
# loc_cg=[(xy_cg[xyi][frame_c+2][1],xy_cg[xyi+len_cg][frame_c+2][1]) for xyi in range(0,len_cg)]
# loc_panel_0=[(xy_panel[xyi][frame_c+2][1],xy_panel[xyi+len_panel][frame_c+2][1]) for xyi in range(0,len_panel)]
# min_x_cg=min([p_cg[0] for p_cg in loc_cg])
# loc_panel=[p_panel for p_panel in loc_panel_0 if p_panel[0]>min_x_cg]
# loc_cg=sort_point(loc_cg)
# loc_panel=sort_point(loc_panel)
# distance_panel=[0 for loc1 in loc_panel]  
# dist_temp=[0 for loc1 in range(0,len_cg-1)] 
# 
# f=open(open_path+output_gap_cp2_cg,'w')
# for I in range(0,len(loc_cg)):
#     f.write(str(loc_cg[I])+'\n')
# f.close
# f=open(open_path+output_gap_cp2_oled,'w')
# for J in range(0,len(loc_panel)):
#     f.write(str(loc_panel[J])+'\n')
# f.close
# =============================================================================
