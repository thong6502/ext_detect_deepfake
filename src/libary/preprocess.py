import os
import sys
import time
import cv2
import dlib
import yaml
import datetime
import glob
import concurrent.futures
import numpy as np
from tqdm import tqdm
from pathlib import Path
from imutils import face_utils
from skimage import transform as trans

def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts


def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        Returns: cropped image, mask (if any), and bounding box coordinates
        """
        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # Calculate the inverse transformation to map back to original image
        M_inv = cv2.invertAffineTransform(M)

        # Define the corners of the output image in the warped space
        height, width = target_size
        corners = np.array([
            [0, 0],           # Top-left
            [width, 0],       # Top-right
            [width, height],  # Bottom-right
            [0, height]       # Bottom-left
        ], dtype=np.float32)

        # Transform corners back to the original image space
        corners_orig = cv2.transform(np.array([corners]), M_inv)[0]

        # Calculate bounding box coordinates (x_min, y_min, x_max, y_max)
        x_coords = corners_orig[:, 0]
        y_coords = corners_orig[:, 1]
        bbox = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]

        # Warp the image
        img_cropped = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img_cropped = cv2.resize(img_cropped, (outsize[1], outsize[0]))

        if mask is not None:
            mask_cropped = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask_cropped = cv2.resize(mask_cropped, (outsize[1], outsize[0]))
            return img_cropped, mask_cropped, bbox
        else:
            return img_cropped, None, bbox

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cropped_faces = []
    landmarks = []
    bboxs = []
    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        for face in faces:            
            landmark = get_keypts(rgb, face, predictor, face_detector)

            # Align and crop the face
            cropped_face, _ , bbox = img_align_crop(rgb, landmark, outsize=(res, res), mask=mask)
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
            
            # Extract the all landmark from the aligned face
            face_align = face_detector(cropped_face, 1)
            if len(face_align) == 0:
                continue
            landmark = predictor(cropped_face, face_align[0])
            landmark = face_utils.shape_to_np(landmark)

            cropped_faces.append(cropped_face)
            bboxs.append(bbox)
            landmarks.append(landmark)

        return cropped_faces, landmarks, bboxs
    
    else:
        return None, None, None



if __name__ == "__main__":
    from pprint import pprint

    face_detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_81_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        print(f"Predictor path does not exist: {predictor_path}")
        sys.exit()
    face_predictor = dlib.shape_predictor(predictor_path)

    img_path = "1.jpg"
    img = cv2.imread(img_path)
    faces, landmarks, bboxs = extract_aligned_face_dlib(face_detector, face_predictor, img, res=256, mask=None)

    if faces is not None:
        for i, face in enumerate(faces):
            cv2.imwrite(f"face{i}.jpg", face)

