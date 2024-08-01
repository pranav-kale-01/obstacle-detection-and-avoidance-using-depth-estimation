import os
import torch
import disparity_utils
import numpy as np 
from midas.model_loader import default_models, load_model

class Disparity:
    def __init__(self): 
        """
        Initializes the object classifier with default settings.

        This function initializes the object classifier by setting up various parameters and loading the model.
        It sets the output path, model type, optimization flag, height, square flag, grayscale flag, model weights,
        and device based on the availability of CUDA.
        Finally, it loads the model using the specified device and other parameters.

        Parameters:
            self (object): The instance of the class.

        Returns:
            None

        Raises:
            None
        """

        self.output_path = "output"
        self.model_type = "dpt_swin2_tiny_256"
        self.optimize = False 
        self.height=None
        self.square=False
        self.grayscale=True
        self.model_weights = default_models[self.model_type]

        # select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # reshaped_weight = torch.reshape(weight, (384, 768))

        # loading the model
        self.model, self.transform, _, _ = load_model(self.device, self.model_weights, self.model_type, self.optimize, self.height, self.square)
        
    def process(self, device, model, image, target_size ):
        """
        Processes an image using a given model.

        This function takes an image as input and processes it using a provided model.
        It performs the following steps:
        1. Converts the image to a torch tensor and sends it to the specified device.
        2. Feeds the tensor through the model to obtain a prediction.
        3. Interpolates the prediction to the specified target size using bicubic interpolation.
        4. Converts the interpolated prediction to a NumPy array and returns it.

        Parameters:
            self (object): The instance of the class.
            device: The device to use for processing (e.g., "cpu" or "cuda").
            model: The model to use for processing the image.
            image: The input image to process, represented as a NumPy array.
            target_size (tuple): The target size of the processed image, specified as (width, height).

        Returns:
            numpy.ndarray: The processed image as a NumPy array.

        Raises:
            None
        """
        
        sample = torch.from_numpy(image).to(device).unsqueeze(0)
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return prediction

    def generate_image( self, input_image ):
        """
        Generates a disparity image from an input image.

        This function generates a disparity image from the provided input image.
        It creates an output folder if the output path is specified.
        If an input image is provided, it reads the image, applies transformations, and computes the disparity.
        Finally, it saves the output disparity image if the output path is specified.

        Parameters:
            self (object): The instance of the class.
            input_image (str): The path to the input image.

        Returns:
            None

        Raises:
            None
        """

        # create output folder
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)

        if input_image is not None:
            # getting input image
            original_image_rgb = disparity_utils.read_image(input_image)  
            image = self.transform({"image": original_image_rgb})["image"]

            # compute
            with torch.no_grad():
                prediction = self.process(self.device, self.model, image, original_image_rgb.shape[1::-1] )

            # savting the output
            if self.output_path is not None:
                filename = os.path.join( self.output_path, os.path.splitext(os.path.basename(input_image))[0])

                # writting the image
                disparity_utils.write_depth(filename, prediction, self.grayscale, bits=2)
                