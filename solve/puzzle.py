
import cv2
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import math
from sklearn.cluster import MiniBatchKMeans
from matplotlib.backends.backend_pdf import PdfPages
import imageio,os
import base64
import shutil


def solve(temp_path,image):

    game,colors = prep(temp_path,image)

    if not game:
        return 'data:image/png;base64,ERROR'

    with imageio.get_writer(temp_path+'/gif.gif', mode='I', fps=2) as gif:
        steps = dfs([],game)
        
        #np.save('res',np.array(steps,dtype='object'))
        #np.save('colors',np.array(colors,dtype='object'))
        
        draw_game(game,colors,gif=gif)
        for step in steps:
            draw_game(step,colors,gif=gif)
        #print(len(steps))
        #print("FINISHHHHHHH")
    file = open(temp_path+"/gif.gif", "rb")    
    data = open(temp_path+"/gif.gif", "rb").read()
    encoded = base64.b64encode(data)
    file.close()
    shutil.rmtree(temp_path)
    #print(encoded)
    return  'data:image/png;base64,' + str(encoded.decode("utf-8"))


def prep(temp_path,image):


    (dh, dw) = (313, 657+40)

    # Read the main image
    img_rgb = cv2.imread(os.path.join(temp_path,image))
    img_rgb_o = cv2.imread(os.path.join(temp_path,image))

    #img_rgb_o = cv2.cvtColor(img_rgb_o, cv2.COLOR_BGR2LAB)
    print(f"HEY WHATS UP + {os.path.join(temp_path,image)}")

    template = cv2.imread('tem.png')

    img_rgb = cv2.resize(img_rgb, (dh, dw), interpolation = cv2.INTER_AREA)
    img_rgb_o = cv2.resize(img_rgb_o, (dh, dw), interpolation = cv2.INTER_AREA)

    img_rgb = cv2.copyMakeBorder(
                     img_rgb, 
                     50, 
                     50, 
                     50, 
                     50, 
                     cv2.BORDER_CONSTANT, 
                     value=(255,255,255)
                  )

    img_rgb_o = cv2.copyMakeBorder(
                     img_rgb_o, 
                     50, 
                     50, 
                     50, 
                     50, 
                     cv2.BORDER_CONSTANT, 
                     value=(255,255,255)
                  )

    img_rgb = cv2.Canny(img_rgb, 100, 600)
    template = cv2.Canny( template, 100, 600)

    result = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)

    #plt.imshow(img_rgb)
    #plt.show()
    # 
    #plt.imshow(template)
    #plt.show()
    #
    #plt.imshow(result)
    #plt.show()

    yloc, xloc = np.where(result >= 0.42)
    #print(np.amax(result))

    points = []
    for (x,y) in zip(xloc,yloc):
        flag = True
        for (x2,y2) in points:        
            dist = math.hypot(x-x2, y-y2)
            if dist < 40:
                flag = False
        if flag:
            points.append((x,y))

    n_colors = len(points)
    print(f"COLORS: {len(points)}")

    colors = []

    for (x,y) in points:
        stick = []
        color = 0
        cv2.circle(img_rgb_o, (x,y), 5, (int(0),int(0),int(0)),2)
        for i in range(4):
    
            off = 43
    
            (xc,yc,zc) = img_rgb_o[y+i*16+20,x+off]
    
            colors.append((xc,yc,zc))
            color = colors.index((xc,yc,zc))
           
            cv2.circle(img_rgb_o, (x+off,y+i*16+20), 5, (int(xc),int(yc),int(zc)),2)
            #cv2.rectangle(img_rgb_o, (x,y+i*23+20), (x+30,y+70), (int(xc),int(yc),int(zc)),2)
            #plt.imshow(img_rgb_o)

    print(f'COLORS: {len(colors)}')

    x, y, z = [i for i,j,k in colors],[j for i,j,k in colors],[k for i,j,k in colors]

    colors = [ (i,j,k) for i,j,k in colors ]

    #img_rgb_o = cv2.cvtColor(img_rgb_o, cv2.COLOR_LAB2BGR)

    cv2.imwrite('out.png', img_rgb_o)

    out = cv2.imread('out.png')
    #plt.imshow(out)
    #plt.show()

    if len(colors) == 0:
        return False,[]

    clt = MiniBatchKMeans(n_clusters = n_colors)
    labels = clt.fit_predict(colors)
    labels  = labels.reshape(n_colors,4)
    #print(labels)
    game = [ [labels[j,i] for i in range(4)] for j in range(n_colors) ]

    colors = clt.cluster_centers_
    #print(colors)
    #colors = [ 
    #    cv2.cvtColor( np.array( [[(x,y,z)]] ,dtype='uint8') , cv2.COLOR_LAB2BGR)[0,0].tolist() for x,y,z in colors
    #]

    game.append([])
    game.append([])
        
    valid = validate(game)
    print(f'valid ? {valid}')
    if not valid:
        return False,[]
    #draw_game(game)
    return game,colors

def validate(game):
    d = {}
    for i in game:
        for j in i:
            if not (j in d):
                d[j] = 0
            d[j] += 1
            if d[j] > 4:
                print(d)
                print(j)
                print("WRONG")
                print(d[j])
                return False
    return True


def draw_game(game,colors,pp=None,gif=None):
    blank_image = np.zeros((200,50*14+5*14,3), np.uint8)
    for j in range(len(game)):
        stick = game[j]
        if stick == []:
            continue;
        offset=4-len(stick) 
        for i in range(len(stick)):
            xc,yc,zc = colors[stick[i]]
            colors[stick[i]]
            cv2.rectangle(blank_image, (50*j+5,(i+offset)*50), (50+50*j,200), (int(zc),int(yc),int(xc)),-1)  
    #plt.imshow(blank_image)
    if pp : pp.savefig()
    if gif : gif.append_data(blank_image)
    #plt.show()


def copy_game(game):
    new_game = [ [ item for item in stick] for stick in game  ]    
    return new_game

def adj(game):
    moves = []
    for i in range(len(game)):
        if(len(game[i]) == 0):
            continue;
        flag = True
        prev_item = game[i][0]
        for item in game[i]:
            flag = flag and item == prev_item
        if flag and len(game[i]) > 1:
            continue;
        if (len(game[i]) == 4 and game[i][0] == game[i][1] and game[i][1] == game[i][2] and game[i][2] == game[i][3]):
            continue;
        move = game[i][0]
        for j in range(len(game)):
            new_game = copy_game(game)
            if (j != i and len(new_game[j]) != 4 and ( len(new_game[j]) == 0 or game[j][0] == move )):
                new_game[j].insert(0,move)
                new_game[i].pop(0)
                moves.append(new_game)
    return moves


def dfs(visited, node_first): 
    
    aux = 0
    queue = [node_first]
    
    steps = [[]]
    while len(queue) != 0:
        if aux % 1500 == 0:
            print("BIGSTEP")
            print(len(queue))
            print(len(steps))
        aux += 1
        node = queue.pop()
        step = steps.pop()
        prev_len=1
        if check(node):
            print("finish")
            print(len(steps))
            return step
        if not validate(node):
            print("validate?")
            #print(node)
            #draw_game(node)
            return False
        if hashhash(node) not in visited:
            visited.append(hashhash(node))
            neighbourhood = adj(node)            
            for neighbour in neighbourhood:
                queue.append(neighbour) 
                steps.append(step + [neighbour])
    return False

def hashhash(game):
    return ','.join( [ '.'.join([ str(i) for i in stick]) for stick in game ] )

def check(game):
    for stick in game:
        if len(stick) == 0:
            continue;
        if len(stick) != 4 or stick[0] != stick[1] or stick[1] != stick[2] or stick[2] != stick[3] : 
            return False            
    return True
