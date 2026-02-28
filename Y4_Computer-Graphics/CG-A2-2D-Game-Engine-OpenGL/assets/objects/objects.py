import numpy as np

def CreateCircle(center, radius, colour, points = 10, offset = 0, semi = False):
    vertices = [center[0], center[1], center[2], colour[0], colour[1], colour[2]]
    indices = []

    if semi == True:
        for i in range(points+1):
            vertices += [
                center[0] + radius * np.cos(float(i * np.pi)/points),
                center[1] + radius * np.sin(float(i * np.pi)/points),
                center[2],
                colour[0],
                colour[1],
                colour[2],
                ]
            
            ind1 = i+1
            ind2 = i+2 if i != points else 1
            indices += [0 + offset, ind1 + offset, ind2 + offset]
    else:
        for i in range(points):
            vertices += [
                center[0] + radius * np.cos(float(i * 2* np.pi)/points),
                center[1] + radius * np.sin(float(i * 2* np.pi)/points),
                center[2],
                colour[0],
                colour[1],
                colour[2],
                ]
            
            ind1 = i+1
            ind2 = i+2 if i != points-1 else 1
            indices += [0 + offset, ind1 + offset, ind2 + offset]

    return (vertices, indices)    

obj1Verts, obj1Inds = CreateCircle([0,0,0],1, [0.769,0.769,0.769], 20)
obj1Props = {
    'vertices' : np.array(obj1Verts, dtype = np.float32),
    
    'indices' : np.array(obj1Inds, dtype = np.uint32),

    'position' : np.array([0, 0, 0], dtype = np.float32),

    'rotation_z' : 0.0,

    'scale' : np.array([30, 30, 1], dtype = np.float32),

    'sens' : 125,

    'velocity' : np.array([0, 0, 0], dtype = np.float32),
    'accleration':np.array([0, 0, 0], dtype = np.float32)
}


def CreateCircle(center, radius, color, segments, index_offset):
    x, y, z = center
    vertices = []
    indices = []
    angle_step = 2 * np.pi / segments

    for i in range(segments):
        theta = i * angle_step
        next_theta = (i + 1) * angle_step


        x1 = x + radius * np.cos(theta)
        y1 = y + radius * np.sin(theta)


        x2 = x + radius * np.cos(next_theta)
        y2 = y + radius * np.sin(next_theta)


        vertices += [
            x, y, z, *color,  
            x1, y1, z, *color, 
            x2, y2, z, *color   
        ]

        indices += [
            index_offset, index_offset + 1, index_offset + 2
        ]
        index_offset += 3

    return vertices, indices

def CreateRectangle(center, width, height, color, index_offset):
    x, y, z = center
    vertices = [
        x - width / 2, y - height / 2, z, *color,  
        x + width / 2, y - height / 2, z, *color,  
        x - width / 2, y + height / 2, z, *color, 
        x + width / 2, y + height / 2, z, *color  
    ]
    indices = [
        index_offset, index_offset + 1, index_offset + 2,
        index_offset + 2, index_offset + 1, index_offset + 3
    ]
    return vertices, indices

def CreateTriangle(center, size, color, index_offset):
    x, y, z = center
    vertices = [
        x, y + size, z, *color, 
        x - size / 2, y - size / 2, z, *color, 
        x + size / 2, y - size / 2, z, *color   
    ]
    indices = [
        index_offset, index_offset + 1, index_offset + 2
    ]
    return vertices, indices

def CreateAngryBird():
    vertices = []
    indices = []

    body_verts = [
        0.0,  1.0,  0.0,  1.0,  1.0,  0.0, 
       -1.0, -1.0,  0.0,  1.0,  1.0,  0.0, 
        1.0, -1.0,  0.0,  1.0,  1.0,  0.0  
    ]
    body_inds = [0, 1, 2]
    vertices += body_verts
    indices += body_inds


    left_eye_verts, left_eye_inds = CreateCircle([-0.3, 0.2, 0.1], 0.2, [1.0, 1.0, 1.0], 20, len(vertices) // 6)
    vertices += left_eye_verts
    indices += left_eye_inds


    right_eye_verts, right_eye_inds = CreateCircle([0.3, 0.2, 0.1], 0.2, [1.0, 1.0, 1.0], 20, len(vertices) // 6)
    vertices += right_eye_verts
    indices += right_eye_inds


    left_pupil_verts, left_pupil_inds = CreateCircle([-0.3, 0.2, 0.15], 0.1, [0.0, 0.0, 0.0], 10, len(vertices) // 6)
    vertices += left_pupil_verts
    indices += left_pupil_inds


    right_pupil_verts, right_pupil_inds = CreateCircle([0.3, 0.2, 0.15], 0.1, [0.0, 0.0, 0.0], 10, len(vertices) // 6)
    vertices += right_pupil_verts
    indices += right_pupil_inds


    beak_verts, beak_inds = CreateTriangle([0.0, -0.2, 0.1], 0.3, [1.0, 0.6, 0.0], len(vertices) // 6)
    vertices += beak_verts
    indices += beak_inds


    left_brow_verts, left_brow_inds = CreateRectangle([-0.4, 0.5, 0.1], 0.4, 0.1, [1.0, 0.0, 0.0], len(vertices) // 6)
    vertices += left_brow_verts
    indices += left_brow_inds

    right_brow_verts, right_brow_inds = CreateRectangle([0.4, 0.5, 0.1], 0.4, 0.1, [1.0, 0.0, 0.0], len(vertices) // 6)
    vertices += right_brow_verts
    indices += right_brow_inds

    belly_verts, belly_inds = CreateCircle([0.0, -0.8, 0.1], 0.5, [1.0, 1.0, 1.0], 20, len(vertices) // 6)
    vertices += belly_verts
    indices += belly_inds

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

def CreatePlayer():

    vertices, indices = CreateCircle([0.0, 0.0, 0.0], 1.0, [220/255, 183/255, 139/255], 50, 0)

    eye_verts1, eye_inds1 = CreateCircle([0.4, -0.5, 0.05], 0.3, [1,1,1], 20, len(vertices)/6)
    vertices += eye_verts1
    indices += eye_inds1

    eye_verts2, eye_inds2 = CreateCircle([-0.4, -0.5, 0.05], 0.3, [1,1,1], 20, len(vertices)/6)
    vertices += eye_verts2
    indices += eye_inds2

    eye_verts3, eye_inds3 = CreateCircle([-0.4, -0.5, 0.10], 0.12, [0,0,0], 10, len(vertices)/6)
    vertices += eye_verts3
    indices += eye_inds3

    eye_verts4, eye_inds4 = CreateCircle([0.4, -0.5, 0.10], 0.12, [0,0,0], 10, len(vertices)/6)
    vertices += eye_verts4
    indices += eye_inds4

    eye_verts5, eye_inds5 = CreateCircle([0.0, 0.0, 0.2], 1.0, [1,0,0], 25, len(vertices)/6, True)
    vertices += eye_verts5
    indices += eye_inds5

    eye_verts6, eye_inds6 = CreateCircle([0.0, 0.95, 0.3], 0.3, [0.9,0.9,0.9], 20, len(vertices)/6)
    vertices += eye_verts6
    indices += eye_inds6

    return vertices, indices

def CreateBackground():
    grassColour = [0,1,0]
    waterColour = [0,0,1]

    vertices = [
        -500.0, 500.0, -0.9, grassColour[0], grassColour[1], grassColour[2],
        -400.0, 500.0, -0.9, grassColour[0], grassColour[1], grassColour[2],
        -400.0, -500.0, -0.9, grassColour[0], grassColour[1], grassColour[2],
        -500.0, -500.0, -0.9, grassColour[0], grassColour[1], grassColour[2],

        500.0, 500.0, -0.9, grassColour[0], grassColour[1], grassColour[2],
        400.0, 500.0, -0.9, grassColour[0], grassColour[1], grassColour[2],
        400.0, -500.0, -0.9, grassColour[0], grassColour[1], grassColour[2],
        500.0, -500.0, -0.9, grassColour[0], grassColour[1], grassColour[2],

        -400.0, 500.0, -0.9, waterColour[0], waterColour[1], waterColour[2],
        400.0, 500.0, -0.9, waterColour[0], waterColour[1], waterColour[2],
        400.0, -500.0, -0.9, waterColour[0], waterColour[1], waterColour[2],
        -400.0, -500.0, -0.9, waterColour[0], waterColour[1], waterColour[2],
    ]

    indices = [
        0,1,2, 0,3,2,
        8,9,10, 8,11,10,
        4,5,6, 4,7,6
    ]

    return vertices, indices



def Createdoor():

    vertices = []
    indices = []

    vert,ind=CreateRectangle([0.0, 0.0, 0.0], 50, 50, [92/255, 64/255, 51/255],0)
    vertices+=vert
    indices+=ind 

    vert,ind=CreateRectangle([0.0, 10.0, 10.0], 35, 15, [0/255, 0/255, 0/255],len(vertices)//6)
    vertices+=vert
    indices+=ind 

    vert,ind=CreateRectangle([0.0, -10.0, 10.0], 35, 15, [0/255, 0/255, 0/255],len(vertices)//6)
    vertices+=vert
    indices+=ind 

    vert,ind=CreateCircle([20,0,5], 5,[1,1,0], 30, len(vertices)//6)

    vertices+=vert
    indices+=ind 
    return vertices,indices

doorverts,doorInds=Createdoor()
doorprops={
    'vertices' : np.array(doorverts, dtype = np.float32),   
    'indices' : np.array(doorInds, dtype = np.uint32),
    'position' : np.array([0, 0, 0], dtype = np.float32),
    'rotation_z' : 0.0,
    'scale' : np.array([1, 1, 1], dtype = np.float32),
}

playerVerts, playerInds = CreateAngryBird()
playerProps = {
    'vertices' : np.array(playerVerts, dtype = np.float32),
    
    'indices' : np.array(playerInds, dtype = np.uint32),

    'position' : np.array([0, 0, 0], dtype = np.float32),

    'rotation_z' : 0.0,

    'scale' : np.array([30, 30, 1], dtype = np.float32),

    'sens' : 125,

    'velocity' : np.array([0, 0, 0], dtype = np.float32),
    'acceleration':np.array([0, 0, 0], dtype = np.float32),

    'lvl1_platform' : np.array([0,0,0],dtype=np.float32)
}

backgroundVerts, backgroundInds = CreateBackground()
backgroundProps = {
    'vertices' : np.array(backgroundVerts, dtype = np.float32),
    
    'indices' : np.array(backgroundInds, dtype = np.uint32),

    'position' : np.array([0, 0, 0], dtype = np.float32),

    'rotation_z' : 0.0,

    'scale' : np.array([1, 1, 1], dtype = np.float32),

    'boundary' : [500.0, -500.0, 500.0, 500.0],

    'river_banks': [-400.0, 400.0]
}



def CreateRedPig(center, radius, segments, index_offset=0):
    vertices = []
    indices = []


    body_verts, body_inds = CreateCircle(center, radius, [0.0, 0.8, 0.0], segments, index_offset)  # Changed to green
    vertices += body_verts
    indices += body_inds
    index_offset += segments + 1


    left_eye_verts, left_eye_inds = CreateCircle(
        [center[0] - radius * 0.3, center[1] + radius * 0.4, center[2] + 0.1],
        radius * 0.2, [0.0, 0.0, 0.0], segments, index_offset
    )
    vertices += left_eye_verts
    indices += left_eye_inds
    index_offset += segments + 1


    right_eye_verts, right_eye_inds = CreateCircle(
        [center[0] + radius * 0.3, center[1] + radius * 0.4, center[2] + 0.1],
        radius * 0.2, [0.0, 0.0, 0.0], segments, index_offset
    )
    vertices += right_eye_verts
    indices += right_eye_inds
    index_offset += segments + 1


    left_pupil_verts, left_pupil_inds = CreateCircle(
        [center[0] - radius * 0.25, center[1] + radius * 0.35, center[2] + 0.2],  # Center pupil better
        radius * 0.08, [0.0, 0.0, 0.0], segments, index_offset  
    )
    vertices += left_pupil_verts
    indices += left_pupil_inds
    index_offset += segments + 1


    right_pupil_verts, right_pupil_inds = CreateCircle(
        [center[0] + radius * 0.25, center[1] + radius * 0.35, center[2] + 0.2],  # Center pupil better
        radius * 0.08, [0.0, 0.0, 0.0], segments, index_offset  
    )
    vertices += right_pupil_verts
    indices += right_pupil_inds
    index_offset += segments + 1


    nose_verts, nose_inds = CreateCircle(
        [center[0], center[1] - radius * 0.2, center[2] + 0.15],  
        radius * 0.15, [1.0, 0.6, 0.7], segments, index_offset  
    )
    vertices += nose_verts
    indices += nose_inds
    index_offset += segments + 1


    left_nostril_verts, left_nostril_inds = CreateCircle(
        [center[0] - radius * 0.05, center[1] - radius * 0.25, center[2] + 0.2],
        radius * 0.05, [0.0, 0.0, 0.0], segments, index_offset  
    )
    vertices += left_nostril_verts
    indices += left_nostril_inds
    index_offset += segments + 1


    right_nostril_verts, right_nostril_inds = CreateCircle(
        [center[0] + radius * 0.05, center[1] - radius * 0.25, center[2] + 0.2],
        radius * 0.05, [0.0, 0.0, 0.0], segments, index_offset  
    )
    vertices += right_nostril_verts
    indices += right_nostril_inds

    return vertices, indices


# Example enemy properties
enemyVerts, enemyInds = CreateRedPig([0, 0, 0], 30, 20)
enemyProps = {
    'vertices': np.array(enemyVerts, dtype=np.float32),
    'indices': np.array(enemyInds, dtype=np.uint32),
    'position': np.array([0, 0, 0], dtype=np.float32),
    'rotation_z': 0.0,
    'scale': np.array([1, 1, 1], dtype=np.float32),
    'velocity': np.array([0, 0, 0], dtype=np.float32)
}

#CreateRectangle(center, width, height, color, index_offset):
entryVerts,entryinds= CreateRectangle([0,0,0], 50, 50 , [0,0,0], 0)
entryProps={
    'vertices':np.array(entryVerts,dtype=np.float32),
    'indices':np.array(entryinds, dtype=np.uint32),
    'position': np.array([0, 0, 0], dtype=np.float32),
    'rotation_z': 0.0,
    'scale': np.array([1, 1, 1], dtype=np.float32),
    'velocity': np.array([0, 0, 0], dtype=np.float32)
}

exitverts, exitinds= CreateRectangle([0,0,0], 50, 50 , [0,0,0], 0)
exitProps={
    'vertices':np.array(exitverts,dtype=np.float32),
    'indices':np.array(exitinds, dtype=np.uint32),
    'position': np.array([0, 0, 0], dtype=np.float32),
    'rotation_z': 0.0,
    'scale': np.array([1, 1, 1], dtype=np.float32),
    'velocity': np.array([0, 0, 0], dtype=np.float32)
}   





def CreateBackground2():
    brownColour = [51/255,36/255,33/255]
    skyColour = [135/255,206/255,235/255]

    vertices = [
        -500.0, 500.0, -0.9, skyColour[0], skyColour[1], skyColour[2],
        -400.0, 500.0, -0.9, skyColour[0], skyColour[1], skyColour[2],
        -400.0,300.0,-0.9     , skyColour[0], skyColour[1], skyColour[2],
        -500.0,300.0,-0.9,  skyColour[0], skyColour[1], skyColour[2],

        -500, 300.0,-0.9,   brownColour[0], brownColour[1], brownColour[2],
        -400,300.0,-0.9,    brownColour[0], brownColour[1], brownColour[2],
        -400,-500,-0.9,      brownColour[0], brownColour[1], brownColour[2],
        -500,-500,-0.9,      brownColour[0], brownColour[1], brownColour[2],

        500.0, 500.0, -0.9, skyColour[0], skyColour[1], skyColour[2],
        400.0, 500.0, -0.9, skyColour[0], skyColour[1], skyColour[2],
        400.0,-300.0,-0.9     , skyColour[0], skyColour[1], skyColour[2],
        500.0,-300.0,-0.9,  skyColour[0], skyColour[1], skyColour[2],

        500, -300.0,-0.9,   brownColour[0], brownColour[1], brownColour[2],
        400,-300.0,-0.9,    brownColour[0], brownColour[1], brownColour[2],
        400,-500,-0.9,      brownColour[0], brownColour[1], brownColour[2],
        500,-500,-0.9,      brownColour[0], brownColour[1], brownColour[2],

        -400,-500,-0.9,     skyColour[0], skyColour[1], skyColour[2],
        400,-500,-0.9,        skyColour[0], skyColour[1], skyColour[2],

    ]
    indices = [
        0,1,2, 
        0,3,2,
        4,5,6,
        4,7,6,
        8,9,10,
        8,10,11,
        12,13,14,
        12,14,15,
        1,9,16,
        16,17,9
    ]


    return vertices, indices

backgroundVerts2, backgroundInds2 = CreateBackground2()
backgroundProps2 = {
    'vertices' : np.array(backgroundVerts2, dtype = np.float32),
    
    'indices' : np.array(backgroundInds2, dtype = np.uint32),

    'position' : np.array([0, 0, 0], dtype = np.float32),

    'rotation_z' : 0.0,

    'scale' : np.array([1, 1, 1], dtype = np.float32),

    'boundary' : [500.0, -500.0, 500.0, 500.0],

    'river_banks': [-400.0, 400.0]
}



sunVerts, sunInds = CreateCircle([400,400,0], 50, [1,1,0], 20, 0)
sunProps = {
    'vertices' : np.array(sunVerts, dtype = np.float32),
    
    'indices' : np.array(sunInds, dtype = np.uint32),

    'position' : np.array([0, 0, 0], dtype = np.float32),

    'rotation_z' : 0.0,

    'scale' : np.array([1, 1, 1], dtype = np.float32),

}



def createcloud():
    vertices = []
    indices = []
    vert, ind = CreateCircle([0.0, 0.0, 0.0], 25.0, [128/255, 128/255, 128/255], 50, len(vertices)//6)
    vertices=vertices+vert
    indices=indices+ind
    vert, ind = CreateCircle([-25.0, 0.0, 0.0], 15.0, [128/255, 128/255, 128/255], 50, len(vertices)//6)
    vertices=vertices+vert
    indices=indices+ind  
    vert, ind = CreateCircle([25.0, 0.0, 0.0], 15.0, [128/255, 128/255, 128/255], 50, len(vertices)//6)
    vertices=vertices+vert
    indices=indices+ind 


    return vertices, indices

cloudverts, cloudInds = createcloud()
cloudprops={
    'vertices' : np.array(cloudverts, dtype = np.float32),
    
    'indices' : np.array(cloudInds, dtype = np.uint32),

    'position' : np.array([0, 0, 0], dtype = np.float32),

    'rotation_z' : 0.0,

    'scale' : np.array([1, 1, 1], dtype = np.float32),

    'velocity':np.array([0.0, 0.0, 0.0], dtype = np.float32),

} 



def CreateBackground3():
    brownColour = [196/255,164/255,132/255]
    skyColour = [0/255,0/255,128/255]

    vertices = [
        -500.0, 500.0, -0.9, skyColour[0], skyColour[1], skyColour[2],
        -200.0, 500.0, -0.9, skyColour[0], skyColour[1], skyColour[2],
        -200.0,-300.0,-0.9     , skyColour[0], skyColour[1], skyColour[2],
        -500.0,-300.0,-0.9,  skyColour[0], skyColour[1], skyColour[2],

        -500, -300.0,-0.9,   brownColour[0], brownColour[1], brownColour[2],
        -200,-300.0,-0.9,    brownColour[0], brownColour[1], brownColour[2],
        -200,-500,-0.9,      brownColour[0], brownColour[1], brownColour[2],
        -500,-500,-0.9,      brownColour[0], brownColour[1], brownColour[2],

        500.0, 500.0, -0.9, skyColour[0], skyColour[1], skyColour[2],
        200.0, 500.0, -0.9, skyColour[0], skyColour[1], skyColour[2],
        200.0,-300.0,-0.9     , skyColour[0], skyColour[1], skyColour[2],
        500.0,-300.0,-0.9,  skyColour[0], skyColour[1], skyColour[2],

        500, -300.0,-0.9,   brownColour[0], brownColour[1], brownColour[2],
        200,-300.0,-0.9,    brownColour[0], brownColour[1], brownColour[2],
        200,-500,-0.9,      brownColour[0], brownColour[1], brownColour[2],
        500,-500,-0.9,      brownColour[0], brownColour[1], brownColour[2],

        -400,-500,-0.9,     skyColour[0], skyColour[1], skyColour[2],
        400,-500,-0.9,        skyColour[0], skyColour[1], skyColour[2],

    ]
    indices = [
        0,1,2, 
        0,3,2,
        4,5,6,
        4,7,6,
        8,9,10,
        8,10,11,
        12,13,14,
        12,14,15,
        1,9,16,
        16,17,9
    ]




    return vertices, indices

backgroundVerts3, backgroundInds3 = CreateBackground3()
backgroundProps3 = {
    'vertices' : np.array(backgroundVerts3, dtype = np.float32),
    
    'indices' : np.array(backgroundInds3, dtype = np.uint32),

    'position' : np.array([0, 0, 0], dtype = np.float32),

    'rotation_z' : 0.0,

    'scale' : np.array([1, 1, 1], dtype = np.float32),

    'boundary' : [500.0, -500.0, 500.0, 500.0],

    'river_banks': [-400.0, 400.0]
}

def createtree():
    vertices = []
    indices = []
    vert, ind = CreateCircle([0, 300.0, 0.0], 100.0, [144/255, 238/255, 144/255], 50, len(vertices)//6)
    vertices=vertices+vert
    indices=indices+ind
    vert, ind = CreateCircle([-100, 275, 0.0], 75.0, [144/255, 238/255, 144/255], 50, len(vertices)//6)
    vertices=vertices+vert
    indices=indices+ind  
    vert, ind = CreateCircle([100.0, 275.0, 0.0], 75.0, [144/255, 238/255, 144/255], 50, len(vertices)//6)
    vertices=vertices+vert
    indices=indices+ind 
    vert,ind =CreateRectangle([0,-200,0], 100, 1000,  [51/255,36/255,33/255], len(vertices)//6)
    vertices=vertices+vert
    indices=indices+ind 


    return vertices, indices 



treevert,treeind=createtree()
treeprop={
    'vertices' : np.array(treevert, dtype = np.float32),
    
    'indices' : np.array(treeind, dtype = np.uint32),

    'position' : np.array([0, 0, 0], dtype = np.float32),

    'rotation_z' : 0.0,

    'scale' : np.array([1, 1, 1], dtype = np.float32),
  
}


def CreateRectangle1(center, width, height, color, index_offset):
    x, y, z = center
    vertices = [
        x - width / 2, y - height , z, *color,  # Bottom left
        x + width / 2, y - height , z, *color,  # Bottom right
        x - width / 2, y          , z, *color,  # Top left
        x + width / 2, y , z, *color   # Top right
    ]
    indices = [
        index_offset, index_offset + 1, index_offset + 2,
        index_offset + 2, index_offset + 1, index_offset + 3
    ]
    return vertices, indices


vinevert, vinind=CreateRectangle1([0,0,0.2],10,700,[0,0,0],0)
vineprop={
    'vertices' : np.array(vinevert, dtype = np.float32),
    
    'indices' : np.array(vinind, dtype = np.uint32),

    'position' : np.array([0, 0, 0], dtype = np.float32),

    'rotation_z' : 0.0,

    'scale' : np.array([1, 1, 1], dtype = np.float32),

}




def CreateKey():
    # Create a small rectangle for the key body
    key_body_verts, key_body_inds = CreateRectangle([0, 0, 0], 10, 5, [1, 1, 0], 0)  # Yellow key
    key_head_verts, key_head_inds = CreateCircle([0, 5, 0], 3, [1, 1, 0], 10, len(key_body_verts) // 6)

    vertices = key_body_verts + key_head_verts
    indices = key_body_inds + key_head_inds

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

# Define Key Object Properties
keyVerts, keyInds = CreateKey()
keyProps = {
    'vertices': np.array(keyVerts, dtype=np.float32),
    'indices': np.array(keyInds, dtype=np.uint32),
    'position': np.array([0, 0, 0], dtype=np.float32),
    'rotation_z': 0.0,
    'scale': np.array([1, 1, 1], dtype=np.float32),
}

#def CreateCircle(center, radius, color, segments, index_offset):
def CreateCrescentMoon(center, outer_radius, inner_radius, offset, segments=30,color=[1.0, 1.0, 0.0]):
    vertices = []
    indices = []
    vert, ind =CreateCircle([400, 400.0, 0.0], 50.0, [246/255,241/255,213/255], 50, len(vertices)//6)
    vertices=vertices+vert
    indices=indices+ind
    vert, ind =CreateCircle([430, 400.0, 5.0], 40.0, [0/255,0/255,128/255], 50, len(vertices)//6)
    vertices=vertices+vert
    indices=indices+ind  

    return vertices, indices



moonVerts, moonInds = CreateCrescentMoon([0, 0, 0], outer_radius=50, inner_radius=40, offset=20, segments=30,color=[246/255,241/255,213/255])
moonProps = {
    'vertices': np.array(moonVerts, dtype=np.float32),
    'indices': np.array(moonInds, dtype=np.uint32),
    'position': np.array([0, 0, 0], dtype=np.float32),
    'rotation_z': 0.0,
    'scale': np.array([1, 1, 1], dtype=np.float32)
}
