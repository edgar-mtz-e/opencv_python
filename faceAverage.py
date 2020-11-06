import dlib
import os
import cv2
import numpy as np
import math
import sys
import random #for delaunay

# Read points from text files in directory
def readPoints(path) :
    # Create an array of array of points.
    pointsArray = [];
    
    #List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        
        if filePath.endswith(".txt"):
            
            #Create an array of points.
            points = [];            
            
            # Read points from filePath
            with open(os.path.join(path, filePath)) as file :
                for line in file :
                    x, y = line.split()
                    points.append((int(x), int(y)))
            
            # Store array of points
            pointsArray.append(points)
            
    return pointsArray;



# Read all jpg images in folder.
def readImages(path) :
    
    #Create array of array of images.
    imagesArray = [];
    
    #List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
       
        if filePath.endswith(".jpg") and "_" not in filePath: 
            # Read image found.
            img = cv2.imread(os.path.join(path,filePath));
            print(filePath)
            # Convert to floating point
            img = np.float32(img)/255.0;

            # Add to array of images
            imagesArray.append(img);
            
    return imagesArray;
                
# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.

def similarityTransform(inPoints, outPoints) :
    s60 = math.sin(60*math.pi/180);
    c60 = math.cos(60*math.pi/180);  
  
    inPts = np.copy(inPoints).tolist();
    outPts = np.copy(outPoints).tolist();
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0];
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1];
    
    inPts.append([np.int(xin), np.int(yin)]);
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0];
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1];
    
    outPts.append([np.int(xout), np.int(yout)]);
    
    tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False);
    
    return tform;


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect);
   
    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]));

   
    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList();

    #print ("triangleList")
    #print (triangleList)
    
    # Find the indices of triangles in the points array

    delaunayTri = []
    
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        #print("{} {} {}".format(pt1, pt2, pt3))
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
                         
                        #print ("pt[j][0]{} - points[k][0] {}={}  and  pt[j][1] {}- points[k][1]{} = {} so appending k:{}".format(pt[j][0],points[k][0],pt[j][0] - points[k][0],  pt[j][1], points[k][1], pt[j][1] - points[k][1] , k))
            #print (ind           )
            if len(ind) == 3:
                 #print ("len(ind) == 3 : {}".format(ind))
                 delaunayTri.append((ind[0], ind[1], ind[2]))
        

    #print (delaunayTri)
    #print (len(delaunayTri))
    return delaunayTri


def constrainPoint(p, w, h) :
    p =  ( min( max( p[0], 0 ) , w - 1 ) , min( max( p[1], 0 ) , h - 1 ) )
    return p;

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

##    print("tin")
##    print(t1)
##    print("tout")
##    print(t2)
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    
    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)   
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);
    
    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect


# Read all jpg images in folder.
def GetLandmarksInPath(path) :    
    #List all files in the directory and generate text files one by one
    for filePath in sorted(os.listdir(path)):       
        if filePath.endswith(".jpg") and "_" not in filePath: 
            print(filePath)
            GetImgLandmarks(os.path.join(path,filePath))

            

def GetImgLandmarks(im):
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    img = cv2.imread(im)
    dets = detector(img)
    #output face landmark points inside retangle    
    #http://dlib.net/python/#dlib.point
    for k, d in enumerate(dets):
        shape = predictor(img, d)#shape is points datatype
    vec = np.empty([68, 2], dtype = int)
    with open(im+".txt", "w") as text_file:    
        for b in range(68):
            vec[b][0] = shape.part(b).x
            vec[b][1] = shape.part(b).y
            print("{} {}".format(vec[b][0],vec[b][1]), file=text_file)
    #print(vec)
    cv2.imshow('Landmark_'+im,AnnotateLandmarks(img,np.matrix([[p.x, p.y] for p in predictor(img, d).parts()])));



def averageFaces(path):    
    # Dimensions of output image
    w = 600;
    h = 600;

    # Read points for all images
    allPoints = readPoints(path);
    
    # Read all images
    images = readImages(path);
   
    # Eye corners
    eyecornerDst = [ (np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3)) ];
    
    imagesNorm = [];
    pointsNorm = [];
    
    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ]);
    
    # Initialize location of average points to 0s
    pointsAvg = np.array([(0,0)]* ( len(allPoints[0]) + len(boundaryPts) ), np.float32());
    
    n = len(allPoints[0]);

    numImages = len(images)
    
    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    
    for i in range(0, numImages):

        points1 = allPoints[i];

        # Corners of the eye in input image
        eyecornerSrc  = [ allPoints[i][36], allPoints[i][45] ] ;
        
        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst);
        
        # Apply similarity transformation
        img = cv2.warpAffine(images[i], tform, (w,h));

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68,1,2));        
        
        points = cv2.transform(points2, tform);
        
        points = np.float32(np.reshape(points, (68, 2)));
        
        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundaryPts, axis=0)
        
        # Calculate location of average landmark points.
        pointsAvg = pointsAvg + points / numImages;
        
        pointsNorm.append(points);
        imagesNorm.append(img);
    

    
    # Delaunay triangulation
    #print ("Average Points")
    #print (pointsAvg)
    
    rect = (0, 0, w, h);
    dt = calculateDelaunayTriangles(rect, np.array(pointsAvg));

    #print ("Delaunay Triangles")
    #print (dt)
    
    # Output image
    output = np.zeros((h,w,3), np.float32());

    # Warp input images to average image landmarks
    for i in range(0, len(imagesNorm)) :
        img = np.zeros((h,w,3), np.float32());
        # Transform triangles one by one
        for j in range(0, len(dt)) :
            tin = []; 
            tout = [];            
            for k in range(0, 3) :                
                pIn = pointsNorm[i][dt[j][k]];
                pIn = constrainPoint(pIn, w, h);
                
                pOut = pointsAvg[dt[j][k]];
                pOut = constrainPoint(pOut, w, h);
                
                tin.append(pIn);
                tout.append(pOut);
                
            warpTriangle(imagesNorm[i], img, tin, tout);
        
        
       
        # Add image intensities for averaging
        output = output + img;
        print(i);
        #cv2.imshow(str(i), output);

    # Divide by numImages to get average
    output = output / numImages;

    # Display result
    cv2.imshow('image', output);
    cv2.imwrite(path +"_result.jpg", output*255)
    cv2.waitKey(0);



#https://stackoverflow.com/questions/37210655/opencv-detect-face-landmarks-ear-chin-ear-line
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#This is using the Dlib Face Detector . Better result more time taking
def GetAccurateLandmarks(im):
    rects = detector(im, 1)
    rect = rects[0]    
    fwd = int(rect.width())
    if len(rects) == 0:
        return None,None
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]),fwd


def AnnotateLandmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def drawPoints(imgPath,imgLandMarks):
    img = cv2.imread(imgPath);
    with open(imgLandMarks) as file :
        points = [];
        i=1
        for line in file :
            x, y = line.split()
            pos =  (int(x), int(y))
            points.append(pos)
            cv2.putText(img, str(i), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
            cv2.circle(img, pos, 3, color=(0, 255, 255))
            i=i+1
        cv2.imshow('Result',img)





#####################FACE MORPHING


        
def getDelaunayTri(allPoints, images):    
    # Dimensions of output image
    w = 600;
    h = 600;
      
    # Eye corners
    eyecornerDst = [ (np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3)) ];
    
    imagesNorm = [];
    pointsNorm = [];
    
    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ]);
    
    # Initialize location of average points to 0s
    pointsAvg = np.array([(0,0)]* ( len(allPoints[0]) ), np.float32());
    
    n = len(allPoints[0]);

    numImages = 2;
    
    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    
    for i in range(0, numImages):

        points1 = allPoints[i];

        # Corners of the eye in input image
        eyecornerSrc  = [ allPoints[i][36], allPoints[i][45] ] ;
        
        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst);
        
        # Apply similarity transformation
        img = cv2.warpAffine(images[i], tform, (w,h));

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68,1,2));        
        
        points = cv2.transform(points2, tform);
        
        points = np.float32(np.reshape(points, (68, 2)));
        
        # Append boundary points. Will be used in Delaunay Triangulation
        #points = np.append(points, boundaryPts, axis=0)
        
        # Calculate location of average landmark points.
        pointsAvg = pointsAvg + points / numImages;
        
        pointsNorm.append(points);
        imagesNorm.append(img);
       
    rect = (0, 0, w, h);
    return calculateDelaunayTriangles(rect, np.array(pointsAvg));
        
# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :

    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        if rectContains(r, pt1) and rectContains(r, pt2) and rectContains(r, pt3) :
            #        cv2.CV_AA
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

# Draw voronoi diagram
def draw_voronoi(img, subdiv) :

    ( facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
        
        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
        ifacets = np.array([ifacet])
        #        cv2.CV_AA
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        #cv2.cv.CV_FILLED
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), -1, cv2.LINE_AA, 0)

def delaunayVoronoi(imgageFile):
        # Define window names
    win_delaunay = "Delaunay Triangulation"
    win_voronoi = "Voronoi Diagram"

    # Turn on animation while drawing triangles
    animate = True
    
    # Define colors for drawing.
    delaunay_color = (255,255,255)
    points_color = (0, 0, 255)

    # Read in the image.
    img = cv2.imread(imgageFile);
    
    # Keep a copy around
    img_orig = img.copy();
    
    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])
    
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect);

    # Create an array of points.
    points = [];
    
    # Read in the points from a text file
    with open(imgageFile+".txt") as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))

    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)
        
        # Show animation
        if animate :
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay( img_copy, subdiv, (255, 255, 255) );
            cv2.imshow(win_delaunay, img_copy)
            cv2.waitKey(100)

    # Draw delaunay triangles
    draw_delaunay( img, subdiv, (255, 255, 255) );

    # Draw points
    for p in points :
        draw_point(img, p, (0,0,255))

    # Allocate space for voronoi Diagram
    img_voronoi = np.zeros(img.shape, dtype = img.dtype)

    # Draw voronoi diagram
    draw_voronoi(img_voronoi,subdiv)

    # Show results
    cv2.imshow(win_delaunay,img)
    cv2.imshow(win_voronoi,img_voronoi)
    cv2.waitKey(0)

            
# Read points from text file
def readFilePoints(path) :
    # Create an array of points.
    points = [];
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))
    return points

def draw_point(img, p, color ) :
    #cv2.cv.CV_FILLED == -1
    cv2.circle( img, p, 2, color, -1, cv2.LINE_AA, 0 )
        
# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []
    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);
    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)
    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2
    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


#MORPHS 2 IMAGES
def morph(img1Name,img2Name, alpha):
    try:
        # Read images
        img1 = cv2.imread(img1Name);
        img2 = cv2.imread(img2Name);
        # Convert Mat to float data type
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        # Read array of corresponding points
        points1 = readFilePoints(img1Name + '.txt')
        points2 = readFilePoints(img2Name + '.txt')
        points = [];

        # Compute weighted average point coordinates
        for i in range(0, len(points1)):
            x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
            y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
            points.append((x,y))

        # Allocate space for final output
        imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

        allPoints = [];
        allPoints.append(points1)
        allPoints.append(points2)

        images = [];
        images.append(img1)
        images.append(img2)
        tri = getDelaunayTri(allPoints,images);
        #print (tri);
        for p in tri:
            x = int(p[0])
            y = int(p[1])
            z = int(p[2])
            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [ points[x], points[y], points[z] ]
            # Morph one triangle at a time.
            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)
            
    ##    # Read triangles from tri.txt
    ##    with open("tri.txt") as file :
    ##        for line in file :
    ##            x,y,z = line.split()
    ##            
    ##            x = int(x)
    ##            y = int(y)
    ##            z = int(z)
    ##            
    ##            t1 = [points1[x], points1[y], points1[z]]
    ##            t2 = [points2[x], points2[y], points2[z]]
    ##            t = [ points[x], points[y], points[z] ]
    ##
    ##            # Morph one triangle at a time.
    ##            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)


        # Display Result
        cv2.imshow("Morphed Face " + str(alpha) , np.uint8(imgMorph))
    except:
        print('An error occured.')



if __name__ == '__main__' :
    #averageFaces('presidents/')
    #TestGetLandmarks("ronald-regan.jpg");
    #drawPoints("presidents/george-h-bush.jpg","presidents/george-h-bush.jpg.txt")
    #GetImgLandmarks("img/aranet/1.jpg");

    path ="img/aranet/";
    #morphing
    alphas = np.linspace(0, 1, num=3)
    for i in alphas:
        morph(path+"1.jpg",path+"6.jpg", i )


    #to generate mix of faces    
##    GetLandmarksInPath(path);
##    averageFaces(path);


    #show delaunay
    #delaunayVoronoi(path+"6.jpg")
