#!/usr/bin/env python

""" Poly.py script to generate deflected polygon images and
kaleidoscopic images (a series of overlaid polygons).
Polygons are deflected according to the algorithm described
by Miyashita.   Useful for creating a large number of unique
abstract stimuli """


# Definitions --
# Kaleido = overlaid set of 'poly's which are randomly deflected polygons
#   Each overlaid 'poly' should be smaller than the previous to avoid hiding it
#   This doesn't always work due to the randomness of making a 'poly'
#
# Poly = randomly deflected polygon for making unique memory stimuli
#   The algorithm starts with a random simple shape: triangle, square, pentagon, hexagon
#   A 'deflection' involved finding the mid point of a side and randomly moving it in or
#   out a random number of degrees of deflection.
#   Note that this doubles the number of sides on each iteration
#   After all the sides are defined, the polygon is 'filled' to make a shape
#   The fill color is randomly selected (or possibly kept constant over a set of shapes)


# Note -- we may need a new function for making highly similar stimuli
#   E.g., to make a 'sister' image from a target that matches both color and keeps
#   the deflection values roughly similar

__version__="Make_poly.py v. 2.0 PJR 6/2012"

from PIL import Image, ImageDraw
import random, math, getopt, sys, string
    

def random_color():
    """ Generate a random (R,G,B) tuple where at least one
    color is >128 (so it's not too dark)."""
    
    r=g=b=0
    while r<128 and b<128 and g<128:
        r=int(random.random()*256)
        g=int(random.random()*256)
        b=int(random.random()*256)
    return (r,g,b)


class poly:

    init_square=[-100,-100,-100,100,100,100,100,-100]
    init_pentagon=[0,-100,-95,-31,-56,81,56,81,95,-31]
    init_hexagon=[0,-100,-87,-50,-87,50,0,100,87,50,87,-50]

    def __init__(self,poly_type,scale,size,color,num_deflect,pv_list=[],deflect_list=[]):
        (self.x, self.y)=size
        self.color=color
        self.cx=self.x/2
        self.cy=self.y/2
        self.deflects=[]

        if poly_type=='r':
            self.init=random.choice(['s','p','h'])
        else:
            self.init=poly_type

        if  self.init=='s':
            self.pv=poly.init_square[0:]
        elif self.init=='p':
            self.pv=poly.init_pentagon[0:]
        elif self.init=='h':
            self.pv=poly.init_hexagon[0:]

        self.nv=len(self.pv)/2

        # Scale the initial polygon to the screen size & scale
        self.cx=self.x/2
        self.cy=self.y/2
        self.scale=scale
        for i in range(0,len(self.pv)):
            self.pv[i]=self.pv[i]*scale
            if i%2==0:
                self.pv[i]=self.pv[i]+self.cx
            else:
                self.pv[i]=self.pv[i]+self.cy

        if pv_list==[]:
            self.recurse(num_deflect,deflect_list)
        else:
            self.pv=[]
            for i in pv_list:
                self.pv.append(i)
            self.deflects=[]
            for i in deflect_list:
                self.deflects.append(i)

        self.nv=len(self.pv)/2  
        
              
    def distort(self,source_poly,distance):
        for i in range(0,len(source_poly.deflects)):
            deflection=source_poly.deflects[i]
            angle=(deflection + random.uniform(distance/2,distance)*random.choice([-1,1]))
            self.deflect(angle)
            
    def draw(self,im):
        im.polygon(self.pv,fill=self.color)

    def deflect(self,angle=0):
        # spread points
        for i in range(int(self.nv*2),0,-2):
            self.pv.insert(i,0)
            self.pv.insert(i,0)
        self.pv.insert(len(self.pv),self.pv[0])
        self.pv.insert(len(self.pv),self.pv[1])
        self.nv=len(self.pv)/2

        # deflect midpoints
        # This produces a specific deflection angle
        # This produces random deflection angles
        if angle==0:
            ga=random.uniform(-150,150)
            # ga=100
        else:
            ga=angle
            # print ga    
        self.deflects.append(ga)
        for i in range(1, int(self.nv), 2):
            mx=(self.pv[(i-1)*2]+self.pv[(i+1)*2])/2
            my=(self.pv[(i-1)*2+1]+self.pv[(i+1)*2+1])/2
            dx=self.pv[(i+1)*2]-self.pv[(i-1)*2]
            dy=self.pv[(i+1)*2+1]-self.pv[(i-1)*2+1]
            theta=math.atan2(dy,dx)
            self.pv[i*2]=mx+ga*math.sin(theta)
            self.pv[i*2+1]=my-ga*math.cos(theta)
        del self.pv[len(self.pv)-2:len(self.pv)]
        self.nv=len(self.pv)/2

        #check for off-screen points
        for i in range(0,len(self.pv)):
            if self.pv[i]<0:
                self.pv[i]=0;
            elif i%2==0:
                if self.pv[i]>=self.x:
                    self.pv[i]=self.x-1
            elif self.pv[i]>=self.y:
                self.pv[i]=self.y-1
                
    def recurse(self,n,deflect_list=[]):
        for i in range(0,n):
            if deflect_list:
                self.deflect(deflect_list[i])
            else:
                self.deflect()

    def print_size(self):
        mx=0
        my=0
        for i in range(0,self.nv*2,2):
            if abs(self.pv[i]-self.cx)>mx:
                mx=abs(self.pv[i]-self.cx)
            if abs(self.pv[i+1]-self.cy)>my:
                my=abs(self.pv[i+1]-self.cy)
        print("Scale %f, max X %d, max Y %d " % (self.scale,mx,my))
        

    def desc(self):
        print("Image size (%d,%d), Scale: %f " % (self.x,self.y,self.scale))
        print("Color ",self.color," init_shape %s, nv %d" % (self.init,self.nv))
        print("Deflections: ",self.deflects)

    def save_txt(self,output_file):
        output_file.write("DEFLECTS:\n")
        for i in self.deflects:
            output_file.write("%s, " % i)
        output_file.write("\n\n")
        output_file.write("PV:\n")
        for i in self.pv:
            output_file.write('%s,' % i)
        output_file.write('\n\n***\n')

                    
class kaleido:

    def __init__(self,poly_type='h',size=(640,480),npoly=2,num_deflect=4,scale=1.5,zoom=0.7,color_list=[],\
                 load_txt=0,copy_from=0,distort=0,deflects=[]):
        self.distort=distort
        self.deflect=num_deflect
        if load_txt!=0:
            self.load_fromfile(load_txt)
        else: # create kaleido
            if copy_from!=0:
                self.npoly=copy_from.npoly
                self.init=copy_from.init
                self.size=copy_from.size
                self.zoom=copy_from.zoom
                #change self.colors list here to adjust number of colors
                self.colors=[copy_from.colors[0],copy_from.colors[1],copy_from.colors[2]]
            else:
                self.npoly=npoly
                self.init=poly_type
                self.size=size
                self.zoom=zoom
                # if color list is not set, then make them random
                #adjust length of self.colors change number of colors (i think must also be reflected in number of npoly below)
                if color_list==[]:
                    for i in range(0,npoly):
                        self.colors.append(random_color())
                else:
                    self.colors=[color_list[0],color_list[1],color_list[2]]
                
            self.deflects=[]
            self.pvs=[]
            self.polyList=[]

            if distort:
                # Create a distortion of a the source polygon
                for i in range(0,self.npoly):
                    # first create a 0-deflect polygon:
                    if self.colors != [] :
                        self.polyList.append(poly(self.init,scale,self.size,self.colors[i],0))
                    else:
                        self.polyList.append(poly(self.init,scale,self.size,random_color(),0))
                    scale=scale*zoom
                    # then use the copy_from & distort parameters to make deflections
                    self.polyList[-1].distort(copy_from.polyList[i],distort)

                    # once created, calculate distortion value
                    self.similarity(copy_from)
                    
            else:
                # Generate a new polygon
                print(self.npoly, self.colors)
                for i in range(0,self.npoly):
                    if deflects:
                        # deflect list is set, so use that instead of random
                        # note that deflect list should be a list of lists or this fails
                        self.polyList.append(poly(self.init,scale,self.size,self.colors[i],
                                                  num_deflect,deflect_list=deflects[i]))
                    else:
                        self.polyList.append(poly(self.init,scale,self.size,self.colors[i],num_deflect))
                    scale=scale*zoom

        # end create kaleido
       
        
    def resize(self,gdist):
        "Resize the whole image to the given frame"
        cx=self.polyList[0].cx
        cy=self.polyList[0].cy
        max_x=min_x=cx
        max_y=min_y=cy
        for p in self.polyList:
            for i in range(0,p.nv):
                if p.pv[i*2]>max_x:
                    max_x=p.pv[i*2]
                if p.pv[i*2]<min_x:
                    min_x=p.pv[i*2]
                if p.pv[i*2+1]>max_y:
                    max_y=p.pv[i*2+1]
                if p.pv[i*2+1]<min_y:
                    min_y=p.pv[i*2+1]
        adist=max(max_x-cx,cx-min_x,max_y-cy,cy-min_y)
        sf=gdist/adist
        print(sf,adist,gdist,max_x,min_x,max_y,min_y)
        for p in self.polyList:
            for i in range(0,p.nv):
                p.pv[i*2]=(p.pv[i*2]-cx)*sf+cx
                p.pv[i*2+1]=(p.pv[i*2+1]-cy)*sf+cy


    def draw(self,im):
        for p in self.polyList:
            p.draw(im)

    def desc(self):
        num=1
        for p in self.polyList:
            print("Poly ",num)
            p.desc()
            num=num+1

    def fix_size(self):
        gdist=min((size[0]/2)*fix_size,(size[1]/2)*fix_size)
        if verbose:
            print("Fixing size at",gdist)
        k.resize(gdist)

    def print_size(self):
        print("%d polygons" % self.npoly)
        for p in self.polyList:
            p.print_size()

    def similarity(self,other):
        self.r=0.0
        for p in range(0,len(self.polyList)):
            for a in range(0,len(self.polyList[p].deflects)):
                difference=self.polyList[p].deflects[a]-\
                             other.polyList[p].deflects[a]
                self.r=self.r+difference*difference ### Squares the difference
        # No need to print, since it's saved in self.r
        #print "SUM OF SQUARES:"
        print(self.r)


    def save_img(self,num=1):
        surface=Image.new('RGB',size)
        im=ImageDraw.Draw(surface)
        self.draw(im)
        if self.distort==0:
            self.descriptive_filename="%s_%dp_%dnd_%dnv" % (self.init,self.npoly,self.deflect,self.polyList[0].nv)
        else:
            self.descriptive_filename="%s_%dp_%dnv_%d" % (self.init,self.npoly,self.polyList[0].nv, self.r)

        surface.save("%s-%d.jpg" % \
                     (self.descriptive_filename,num))
    
    def save_txt(self,suffix=1):
        f=open("%s-%s.kdf" % (self.descriptive_filename, suffix),"w")
        f.write("poly_type=%s;size=%s;npoly=%s;deflect=%s;scale=%s;zoom=%s;colors=%s\n" % (self.init, self.size, self.npoly , len(self.polyList[0].deflects), scale, self.zoom, self.colors ))
        f.write("***\n")
        for p in self.polyList:
            p.save_txt(f)
        f.write("\n")
        f.close()



    def load_fromfile(self,filename):
        self.filename=filename
        self.f=open(filename)
        self.p=self.f.read()
        self.f.close()
        parse_options=re.compile("poly_type=(\S);size=\((\d*),\s(\d*)\);npoly=(\d);deflect=(\d);scale=(\d*.\d*);zoom=(\d*.\d*);colors=\[\((\d*),\s(\d*),\s(\d*)\),\s\((\d*),\s(\d*),\s(\d*)\)\]")
        self.preoptions=parse_options.findall(self.p)
        self.options=self.preoptions[0]
        self.init=self.options[0]
        self.size=(int(self.options[1]), int(self.options[2]))
        self.npoly=int(self.options[3])
        self.deflect=int(self.options[4])
        self.scale=string.atof(self.options[5])
        self.zoom=string.atof(self.options[6])
        self.colors=[(int(self.options[7]),int(self.options[8]),int(self.options[9])), (int(self.options[10]),int(self.options[11]),int(self.options[12]))]

        pattern='DEFLECTS:\n(-*\d*.\d*)'
        for i in range(0, self.deflect-1):
            pattern=pattern + ',\s(-*\d*.\d*)'
        parse_deflects=re.compile(pattern)
        parse_pvs=re.compile("PV:\n(\S*)\n")

        pv_list=parse_pvs.findall(self.p)
        deflect_list=parse_deflects.findall(self.p)

        self.polyList=[]
        for i in range(0,self.npoly):
            self.polyList.append(poly(self.init,scale,self.size,self.colors[i],0,
                                      map(string.atof,string.split(pv_list[i],',')[:-1]),
                                      map(string.atof,deflect_list[i])))

     
###################################################

# Utility functions
def many_kaleidos(filename):
    f=open(filename)
    d=f.readlines()
    f.close()

    count=0
    for l in d:
        angles=map(string.atof,string.split(l))
        k=kaleido(deflects=[angles[0:4],angles[4:8]])
        k.save_img(count)
        count=count+1


def find_pair(source_k,distance,tolerance=.10):
    d1=kaleido(copy_from=source_k,distort=distance)

    print("D1 distance",d1.r)
    done=0
    while not done:
        d2=kaleido(copy_from=source_k,distort=distance)
        source_dist=d2.r
        d2.similarity(d1)
        print("D2 distance",source_dist," D1-D2 distance",d2.r)

        t1=abs((d1.r-source_dist)/d1.r)
        t2=abs((d2.r-(2*d1.r))/d2.r)

        print("T1 %.2f%%, T2 %.2f%%" % (t1,t2))
        if t1<tolerance and t2<tolerance:
            done=1

    return (d1,d2)

        
def make_family(prototype,distance,n,output_dir):
    if os.path.exists(output_dir):
        os.chdir(output_dir)
    else:
        os.mkdir(output_dir)
        os.chdir(output_dir)

    for i in range(0,n):
        m=kaleido(copy_from=prototype,distort=distance)
        m.save_img(i)
        m.save_txt(i)

    os.chdir("..")

def make_sister(prototype,distance,name):  #orginial version that works
    sister=kaleido(copy_from=prototype,distort=distance)
    print (prototype.colors, sister.colors)

    surface=Image.new('RGB',size)
    im=ImageDraw.Draw(surface)
    sister.draw(im)
    surface.save("%s_sis.jpg" % name) 

    if verbose:
       sister.desc()
    print (prototype.colors, sister.colors)
    
    
##########################
# class kaleido:
def main(poly_type='h', size=(640,480), npoly=2, num_deflect=4,
         scale=1.5, zoom=0.7,color_list=[], load_txt=0,
         copy_from=0, distort=0, deflects=[], folder='./normal', 
         name='image', num=1, schema=0):
    import os
#def main(color_list=[], stem=""):

    if verbose:
        print ("Size=(%d,%d), Scale=%f, Zoom=%f, Poly=%d, Deflect=%d" % \
              (size[0], size[1], scale, zoom, npoly, deflect))
        print ("Generating %d images as %sXXX.jpg" % (num, name))
        if same_color:
            print ("Color scheme held constant")
    
    prog_num = schema * num
    for i in range(1, num+1):  #used to say: for i in range(0,num) : change to renumber
        if not same_color or color_list==[]:
            k = kaleido(poly_init_type,size,npoly,deflect,scale,zoom)
            for p in k.polyList:
                color_list.append(p.color)
        else:
            k=kaleido(poly_init_type,size,npoly,deflect,scale,zoom,color_list)
        if fix_size>0:
            gdist=min((size[0]/2)*fix_size,(size[1]/2)*fix_size)
            if verbose:
                print ("Fixing size at",gdist)
            k.resize(gdist)
        surface=Image.new('RGB', size)
        im=ImageDraw.Draw(surface)
        k.draw(im)
        

        os.makedirs(folder, exist_ok=True)
        path_normal = os.path.join(folder, "normal")

        os.makedirs(path_normal, exist_ok=True)
        n = prog_num + i
        surface.save("%s/%s-%03d.jpg" % (path_normal, name, n))
        if verbose:
            k.desc()
        # Adding a sister paired image
        os.makedirs(os.path.join(folder, "sister"), exist_ok=True)
        path_sister = os.path.join(folder, "sister")
        make_sister(k, 10, "%s/%s-%03d" % (path_sister, name, n)) #change the 'distance' variable here



if __name__ == '__main__':
    ###################
    # Default variables
    
    size=(640, 480)          # Size of canvas to draw to
    poly_init_type='r'      # Starting polygon to deflect from, defaults to 'r' = random
    npoly=2                 # Number of overlaid polygons
    deflect=4               # Number of side deflection rounds
    scale=1.5               # Scale of image within canvas
    zoom=0.70               # Percentage size of each overlaid
    num=1                   # Number of images to make
    out='kaleido'           # Filename stem to write to
    verbose=0               # Whether to print information while making
    same_color=0            # Make all the generated images the same color scheme
    fix_size=0              # Not needed to change


    # Reset random number generator
    random.seed()

    # Configuring to override defaults above
    num=1         # Makes 12 images
    same_color=1   # All with same color scheme
    npoly=3        # 3 overlaid polygons #change when changing numcolors
    deflections=10
    
    # Main can be called with a color list if you want to set the color scheme
    # main( [ (255,0,0),(0,255,0),(0,0,255) ] )

    # This calls the main program to make the images with a random color scheme
    # Change the value of the number inside range() to make multiple sets, each with "num" kscopes, changing the color scheme between but not within sets
    n_schemes = 6
    n_images = 100
    
    for schema in range(n_schemes):
        color_list = random_color(), random_color(), random_color()
        folder = "schema_%01d" % (schema+1)
        main(size=size, npoly=npoly, num_deflect=deflect, scale=scale, zoom=zoom,
                    color_list=color_list, deflects=deflections, folder=folder, num=n_images, schema=schema)





