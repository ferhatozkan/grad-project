"""#$ python proje_ica_deneme.py --shape-predictor shape_predictor_5_face_landmarks.dat
from PIL import Image
image = Image.open('frame.jpg', 'r')      

            width, height = image.size
            pixel_values = list(image.getdata())  
            pixel_values = numpy.array(pixel_values).reshape((width, height, 3))
            #cv2.imwrite("face%d.jpg" % numff, pixel_values)
            total=None
            total=0
            width=round((width/5)*3)
            for x in range(width):
                average = pixel_values[x]
                for y in range(height):
                    mid = average[y]
                    a=mid[2]
                    mid[2]=mid[0]
                    mid[0]=a
                    if (mid [2] > 95 and mid [1] > 40 and mid [0] > 20 and (max(mid[0],mid[1],mid[2])-min(mid[0],mid[1],mid[2])) >15 and math.fabs(mid[2]-mid[1])>15 and mid[2]>mid[1] and mid[2]>mid[1]):
                        total=mid+total
                    else:
                        excep+=1
                        pixel_values[x][y][0]=255
                        pixel_values[x][y][1]=255
                        pixel_values[x][y][2]=255
                        
            #cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)
            #cv2.imwrite("renk%d.jpg" % numff, pixel_values)
            total=total/((width*height)-excep)
    

            red.append(total[0])
            green.append(total[1])
            blue.append(total[2])