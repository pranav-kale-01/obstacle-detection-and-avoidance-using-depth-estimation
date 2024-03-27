"""Perform Inference On Video Stream taken from a saved video or webcam
"""

import time
import cv2
import pyttsx3
from pynput.keyboard import Key, Listener
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import traceback
import onnxruntime as ort
import numpy as np
import ffmpeg

from ultralytics import YOLO
from midas.hubconf import transforms as midas_transforms


class Infer:
    def __init__(self, webcam=False, vid_name=1, frame_delay=30):
        """
        Initializes the object classifier with specified settings.

        This function initializes the object classifier by setting up various parameters and paths.
        It also loads the required models using the 'load_models' function.

        Parameters:
            self (object): The instance of the class.
            webcam (bool, optional): Flag indicating whether webcam mode is enabled or not. Default is False.
            vid_name (int, optional): The name or identifier of the video. Default is 15.
            frame_delay (int, optional): The delay between frames in video processing. Default is 24.

        Returns:
            None

        Raises:
            None
        """

        # flag that terminates the continous obstacle detection
        self.close_loop = False

        # flag that pauses the obstacle detection while running other modules
        self.detect_obstacles = True

        self.frame_count = 1
        self.video_name = vid_name
        self.frame_delay = frame_delay

        # images and videos
        self.disparity_image_path = "./output\\current_frame"
        self.video_support_image = "./video_support/current_frame"
        self.output_image_path = "./output/current_frame"
        video_path = f"./input/videoplayback.webm"

        # for object detection
        det_model_dir = "./config/ch_PP-OCRv3_det_infer"
        rec_model_dir = "./config/ch_PP-OCRv3_rec_infer"

        self.infer_ocr = PaddleOCR(
            use_angle_cls=True, det_model_dir=det_model_dir, rec_model_dir=rec_model_dir
        )

        # choose codec according to format needed
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video = cv2.VideoWriter("output/video7+1.mp4", fourcc, 1, (1920, 1080))

        # loading required models
        self.load_models()

        # binding the keys for running the text recognition and currency identification modules
        listener = Listener(on_press=self.handle_keypress)
        listener.start()

        # checking if webcam mode is on
        if webcam:
            self.load_from_webcam()
        else:
            self.load_from_video(video_path)

    def check_rotation(self, path_video_file):
        # this returns meta-data of the video file in form of a dictionary
        meta_dict = ffmpeg.probe(path_video_file)

        # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
        # we are looking for
        rotateCode = None
        if int(meta_dict["streams"][0]["tags"]["rotate"]) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict["streams"][0]["tags"]["rotate"]) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(meta_dict["streams"][0]["tags"]["rotate"]) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

        return rotateCode

    def correct_rotation(self, frame, rotateCode):
        return cv2.rotate(frame, rotateCode)

    def handle_keypress(self, key):
        try:
            if key == Key.esc:
                # gracefully terminating the execution
                self.close_loop = True
                exit()

            elif key.char == "a":
                # running the text reading module when the key "a" is pressed
                self.detect_obstacles = False

                # calling the ocr module
                self.perform_ocr()

                self.detect_obstacles = True

            elif key.char == "b":
                # running the currency identification module when the key "b" is pressed
                self.detect_obstacles = False

                print("run the currency identification module")

                self.detect_obstacles = True
        except:
            pass

    def read_txt_aloud(self, text):
        try:
            # setting up text to speech
            self.engine = pyttsx3.init("sapi5")
            self.engine.setProperty("rate", 110)

            self.engine.say(text)
            self.engine.runAndWait()

        except Exception as e:
            print(e)
            traceback.print_exc()

    def perform_ocr(self):
        try:
            img_path = "./output/current_frame.jpg"

            # inference
            result = self.infer_ocr.ocr(img_path)

            # saving the results
            save_path = self.output_image_path + "_test.jpg"

            image = cv2.imread(img_path)
            extracted_text = ""

            # for debugging
            if len(result) != 0:
                boxes = [line[0] for line in result[0]]
                txts = [line[1][0] for line in result[0]]

                # appending the current line to text string
                for txt in txts:
                    extracted_text = extracted_text + " " + txt

                scores = [line[1][1] for line in result[0]]
                im_show = draw_ocr(
                    image, boxes, txts, scores, font_path="./config/en_standard.ttf"
                )
                im_show = Image.fromarray(im_show)
                im_show.save(save_path)

            # reading the text aloud
            self.read_txt_aloud(extracted_text)

        except Exception as e:
            print(e)
            traceback.print_exc()

    def load_models(self):
        """
        Loads various models and initializes text-to-speech engine.

        This function loads the following models:
        - Object detection model using YOLOv5 architecture with the 'yolov5x' variant.
        - Disparity model for depth estimation.

        Additionally, it sets up the text-to-speech engine using the 'sapi5' platform and sets the speech rate to 110.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None

        Raises:
            None
        """

        # Loading object detection model
        self.model = YOLO("newbestyolov8.pt")
        self.model_version = 8

        # Disparity
        self.transform = midas_transforms().swin256_transform
        self.ort_sess = ort.InferenceSession("test.onnx")

    def load_from_webcam(self):
        """
        Loads input from the webcam for object classification.

        This function opens the webcam stream and sets it as the input stream for object classification.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None

        Raises:
            None
        """
        # getting input from webcam
        self.inp_stream = cv2.VideoCapture(0)

    def load_from_video(self, video_path):
        """
        Loads input from a video file for object classification.

        This function opens the specified video file and sets it as the input stream for object classification.

        Parameters:
            self (object): The instance of the class.
            video_path (str): The path to the video file.

        Returns:
            None

        Raises:
            None
        """
        self.inp_stream = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_MSMF)

        # check if video requires rotation
        # self.inp_stream = self.check_rotation(video_path)

    def find_brightest_color(self, colors):
        """
        Finds the brightest color from a list of colors.

        This function iterates through a list of colors and determines the brightest color based on the sum of RGB values.
        It returns the index of the color, the sum of RGB values, and the brightest color itself.

        Parameters:
            self (object): The instance of the class.
            colors (list): A list of colors, where each color is represented as a tuple of three RGB values.

        Returns:
            tuple: A tuple containing the index of the brightest color, the sum of RGB values, and the brightest color.

        Raises:
            None
        """

        # finding the brightest color
        brightest_color = (0, 0, 0)
        highest_sum = 0
        bound_box_ind = 0

        for ind, i in enumerate(colors):
            color_sum = i
            if highest_sum < color_sum:
                # assigning current sum as highest
                highest_sum = color_sum

                # assigning current color as brightest color
                brightest_color = i

                # assigning current bound box index
                bound_box_ind = ind

        return (bound_box_ind, highest_sum, brightest_color)

    def run(self):
        # infinite loop
        while not self.close_loop:
            if self.detect_obstacles:
                # getting required frame
                ret, frame = self.inp_stream.read()

                if not ret:
                    print("unable to load the video")
                    self.video.release()
                    break

                if ( self.frame_count < self.frame_delay or self.frame_count % self.frame_delay != 0 ):
                    self.frame_count += 1
                    continue

                # starting timer for inference timing
                start = time.time()

                # writing the extracted images
                cv2.imwrite(self.output_image_path + ".jpg", frame)

                results = self.model(frame)
                objects = results[0].boxes.xyxy

                # list of bounding boxes
                bound_boxes = []
                for i, object in enumerate(objects):
                    object = object.tolist()
                    object.append( results[0].names[int(results[0].boxes.cls[i].item())] )
                    bound_boxes.append(object)

                # retuning if no bounding boxes
                if len(bound_boxes) == 0:
                    self.frame_count += 1
                    continue

                # calculating inference time
                # end = time.time()
                # print("object detection - ", end - start)
                # self.frame_count += 1

                # getting disparity filter from midas
                input_batch = self.transform(frame).to("cpu")
                outputs = self.ort_sess.run(None, {"input": input_batch.numpy()})

                # converting output
                out_img = outputs[0].T
                out_img = cv2.resize(
                    out_img,
                    (frame.shape[0], frame.shape[1]),
                    interpolation=cv2.INTER_CUBIC,
                )
                out_img = (
                    ((out_img - out_img.min()) / (out_img.max() - out_img.min()) * 255)
                    .astype(np.uint8)
                    .T
                )
                
                _, encoded_img = cv2.imencode(".png", out_img)
                img = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

                # now once the image is generated plotting the bounding boxes on disparity image
                for i in bound_boxes:
                    # displaying the bounding box
                    cv2.rectangle(
                        img,
                        (int(i[0]), int(i[1])),
                        (int(i[2]), int(i[3])),
                        (0, 0, 255),
                        1,
                    )

                    # also displaying the label
                    cv2.putText(
                        img,
                        i[4],
                        (int(i[0]), int(i[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )

                cv2.imwrite(self.disparity_image_path + ".png", img)

                # now that bounding boxes are plotted finding out the color depth of the object
                # by the color from five different points of the image and assigning the highest color
                bound_box_colors = []

                # getting window height and width
                height, width = img.shape[:2]

                x_padding = i[2] * 0.02
                y_padding = i[3] * 0.02

                for i in bound_boxes:
                    # list of colors of bounding box
                    colors = []

                    # getting color of bounding box's center
                    colors.append(
                        img[
                            int(int(i[1] + i[3]) * 0.5), int(int(i[0] + i[2]) * 0.5)
                        ].tolist()
                    )

                    # getting color from top left section
                    colors.append(
                        img[
                            int(i[1] + y_padding),
                            (
                                int(i[0] + x_padding)
                                if i[0] + x_padding < width
                                else width - 1
                            ),
                        ].tolist()
                    )

                    # getting color from bottom left section
                    colors.append(
                        img[
                            int(i[3] - y_padding),
                            (
                                int(i[0] + x_padding)
                                if i[0] + x_padding < width
                                else width - 1
                            ),
                        ].tolist()
                    )

                    # getting color from top right section
                    colors.append(
                        img[int(i[1] + y_padding), int(i[2] - x_padding)].tolist()
                    )

                    # getting color from bottom right section
                    colors.append(
                        img[int(i[3] - y_padding), int(i[2] - x_padding)].tolist()
                    )

                    # finding the brightest color
                    _, _, brightest_color = self.find_brightest_color(colors)
                    bound_box_colors.append(brightest_color)

                # finding the brightest bounding box out of all bounding boxes
                bound_box_ind, highest_sum, _ = self.find_brightest_color(
                    bound_box_colors
                )

                # showing result in text format
                bb = bound_boxes[bound_box_ind]

                # loading object detection image
                object_detection_image = cv2.imread(self.output_image_path + ".jpg")

                # boundary lines
                object_detection_image = cv2.line(
                    object_detection_image,
                    (int(width * 0.1), 0),
                    (int(width * 0.1), height),
                    (0, 255, 0),
                    2,
                )
                object_detection_image = cv2.line(
                    object_detection_image,
                    (int(width * 0.9), 0),
                    (int(width * 0.9), height),
                    (0, 255, 0),
                    2,
                )
                object_detection_image = cv2.line(
                    object_detection_image,
                    (0, int(height * 0.1)),
                    (width, int(height * 0.1)),
                    (0, 255, 0),
                    2,
                )
                object_detection_image = cv2.line(
                    object_detection_image,
                    (0, int(height * 0.9)),
                    (width, int(height * 0.9)),
                    (0, 255, 0),
                    2,
                )
                 
                # finding bounding boxes center
                bound_box_center = [int((bb[0] + bb[2]) / 2), int((bb[1] + bb[3]) / 2)]

                # ignoring the object if it has a brightesness below a certain threshold value
                if highest_sum <= 650:
                    # saving the output image
                    cv2.imwrite(
                        self.video_support_image + ".jpg", object_detection_image
                    )

                    self.frame_count += 1
                    continue

                # ingoring if the object is in the 5% of the frame from all 4 sides of the frame
                if (
                    bound_box_center[0] < int(width * 0.1)
                    or bound_box_center[0] > int(width * 0.9)
                    or bound_box_center[1] < int(height * 0.1)
                    or bound_box_center[1] > int(height * 0.9)
                ):
                    # saving the output image

                    cv2.imwrite(
                        self.video_support_image + ".jpg", object_detection_image
                    )

                    self.frame_count += 1
                    continue

                # drawing the brightest bounding box on image
                object_detection_image = cv2.rectangle(
                    object_detection_image,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    (0, 0, 255),
                    2,
                )
                object_detection_image = cv2.putText(
                    object_detection_image,
                    bb[4],
                    (int(bb[0]), int(bb[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

                # saving the output image
                cv2.imwrite(self.video_support_image + ".jpg", object_detection_image)
                self.video.write(object_detection_image)

                # output_text
                output_text = ""

                # checking where the object is from the user (the object's direction)
                if bound_box_center[0] >= int(width * 0.4) and bound_box_center[
                    0
                ] <= int(width * 0.6):
                    output_text = f"{bb[4]} in front"

                elif bound_box_center[0] >= int(width * 0.6) and bound_box_center[
                    0
                ] <= int(width * 0.8):
                    output_text = f"{bb[4]} slight right"

                elif bound_box_center[0] >= int(width * 0.2) and bound_box_center[
                    0
                ] <= int(width * 0.4):
                    output_text = f"{bb[4]} slight left"

                elif bound_box_center[0] <= int(width * 0.2):
                    output_text = f"{bb[4]} left"

                else:
                    output_text = f"{bb[4]} right"

                if self.detect_obstacles:
                    # converting text to speech
                    self.read_txt_aloud(output_text)
                    pass

                self.frame_count += 1

        self.video.release()


# driver code
if __name__ == "__main__":
    # creating object of Infer
    inf = Infer(webcam=False, vid_name=2, frame_delay=30)
    inf.run()
