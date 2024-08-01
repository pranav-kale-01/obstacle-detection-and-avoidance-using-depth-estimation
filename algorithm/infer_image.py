import os  
import torch 
import time
import disparity
import cv2
import pyttsx3

# Inference
model = torch.hub.load( 'ultralytics/yolov5', 'yolov5x' )

# Disparity 
disp_model = disparity.Disparity()

while True: 
    # taking image name as input
    img_name = input( "Enter image name : ")
    if img_name == "exit": 
        break

    # starting timer for inference timing
    start = time.time() 

    # getting inference of image
    im_path = f'D:\\Data_Science_programs\\inference_code\\input\\{img_name}.jpg'

    # checking if file does not exists then continuing the loop 
    if not os.path.exists( im_path ):
        print("file does not exists!")
        continue

    results = model(im_path)

    # showing results 
    result_data =  results.pandas().xyxy[0].to_dict(orient = "records") 
    
    bound_boxes = []

    for d in result_data:
        bound_boxes.append( [d['xmin'], d['ymin'], d['xmax'], d['ymax'], d['name']] )     

    # list of bounding boxes 
    # print( bound_boxes )

    # getting disparity filter from midas 
    disp_model.generate_image( im_path )

    # loading disparity image 
    disparity_image_path = f"D:\\Data_Science_programs\\inference_code\\output\\{img_name}.png"
    img = cv2.imread( disparity_image_path )

    # now once the image is generated plotting the bounding boxes on disparity image 
    # for i in bound_boxes: 
    #     cv2.rectangle( img, (int(i[0]), int( i[1] ) ), (int(i[2]), int(i[3 ])), (0,0,255), 1)
    # cv2.imwrite( disparity_image_path, img )
 
    # now that bounding boxes are plotted finding out the color depth of the object 
    # by the color from five different points of the image and assigning the highest color
    bound_box_colors = []

    # for drawing circles
    # radius = 5
    # color = (255, 0, 0)
    # thickness = 2

    for i in bound_boxes:
        x_padding = (i[2] * 0.02) 
        y_padding = (i[3] * 0.02)

        colors = []
    
        # getting color of bounding box's center    
        colors.append( img[ int( int(i[1]+i[3]) * 0.5 ), int( int(i[0]+i[2]) * 0.5 ) ] )

        # getting color from top left section 
        colors.append(  img[ int( i[1] + y_padding ), int( i[0] + x_padding )] )

        # getting color from bottom left section 
        colors.append( img[int(i[3] - y_padding ), int( i[0] + x_padding )] )

        # getting color from top right section 
        colors.append( img[int( i[1] + y_padding), int( i[2] - x_padding )] )

        # getting color from bottom right section 
        colors.append( img[int( i[3] - y_padding ), int( i[2] - x_padding )] )

        # cv2.circle(img, ( int( int(i[0] + i[2]) * 0.5 ) , int( int(i[1] + i[3]) * 0.5 )), radius, color, thickness)
        # cv2.circle(img, ( int( i[0] + x_padding ) , int( i[1] + y_padding ) ), radius, color, thickness)
        # cv2.circle(img, ( int( i[0] + x_padding ) , int(i[3] - y_padding ) ), radius, color, thickness)
        # cv2.circle(img, ( int( i[2] - x_padding ) , int( i[1] + y_padding) ), radius, color, thickness)
        # cv2.circle(img, ( int( i[2] - x_padding ) , int( i[3] - y_padding ) ), radius, color, thickness)
        
        # finding the brightest color
        brightest_color = (0,0,0)
        highest_sum = 0

        for i in colors: 
            v1, v2, v3= i 

            color_sum = int( v1 )  + int( v2 ) + int( v3 )
            if highest_sum < color_sum : 
                highest_sum = color_sum 
                brightest_color = i

        # print( "brightest color is : ", brightest_color )
        bound_box_colors.append( brightest_color ) 

        # avg_color = [0, 0, 0]
        # finding the average color
        # for i in colors: 
        #     v1, v2, v3 = i
        #     avg_color[0] += v1
        #     avg_color[1] += v2 
        #     avg_color[2] += v3 

        # avg_color = (avg_color[0] / 5, avg_color[1] / 5, avg_color[2] / 5 ) 
        # print( "average color is : ", avg_color )

    # finding the brightest bounding box out of all bounding boxes 
    brightest_color = (0,0,0)
    bound_box_ind = 0
    highest_sum = 0

    for ind, i in enumerate( bound_box_colors ):
        v1, v2, v3= i 
    
        color_sum = int( v1 )  + int( v2 ) + int( v3 )
        if highest_sum < color_sum : 
            # assigning current sum as highest
            highest_sum = color_sum 

            # assigning currnet color as brightest color
            brightest_color = i

            # assigning current bound box index 
            bound_box_ind = ind 

    # showing the final resulting image
    # cv2.imshow( "disparity image", img )
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    # showing result in text format
    bb = bound_boxes[bound_box_ind]
    # text = f"Bounding box {bound_box_ind} is the brightest having class {bb[4]}"

    # loading object detection image
    object_detection_image = cv2.imread( im_path )

    # drawing the brightest bounding box on image
    object_detection_image = cv2.rectangle( object_detection_image, (int( bb[0] ), int( bb[1] )), (int( bb[2]), int( bb[3] )), (0,0,255), 2)
    object_detection_image = cv2.putText(object_detection_image, bb[4], (int( bb[0]), int(bb[1]) - 5 ), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # calculating inference time
    end = time.time() 
    print("inference time - ", end-start )

    height, width = img.shape[:2]

    # finding bounding boxes center
    bound_box_center = int( ( bb[0] + bb[2] ) / 2 )
    text = ""

    # checking where the object is from the user (the objects direction)
    if( bound_box_center >= int( width*0.4) and bound_box_center <= int( width*0.6 ) ): 
        text = f"{bb[4]} infront of you!"

    elif ( bound_box_center >= int( width*0.6) and bound_box_center <= int( width*0.8 ) ):
        text = f"{bb[4]} on slightly right of you!"
    
    elif ( bound_box_center >= int( width*0.2) and bound_box_center <= int( width*0.4 ) ):
        text = f"{bb[4]} on slightly left of you!"

    elif ( bound_box_center <= int( width*0.2) ):
        text = f"{bb[4]} on left of you!"

    else:
        text = f"{bb[4]} on right of you!"

    # converting text to speech
    # engine = pyttsx3.init("sapi5")
    # engine.setProperty('rate', 110 )     # setting up new voice rate
    # engine.setProperty('voice', engine.getProperty("voices")[0]  )
    # engine.say(text)
    # engine.runAndWait() 
    

    # svaing the ouptut image
    output_image_path = f'D:\\Data_Science_programs\\inference_code\\output\\{img_name}.jpg'
    cv2.imwrite( output_image_path, object_detection_image )

    # # showing the final resulting image
    # cv2.imshow( "object detection image", object_detection_image )
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()