# Thư viện cần thiết
import cv2
import torch
import numpy as np
from .model import make_model
from deep_sort_realtime.deepsort_tracker import DeepSort
# from facenet_pytorch import MTCNN
import os
import dlib
from .libary.preprocess import extract_aligned_face_dlib
from collections import defaultdict

from .config import TRANSFORM
from pprint import pprint
from PIL import Image
from tqdm import tqdm
import sys
import json
from cvzone import putTextRect, cornerRect

def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tracker = DeepSort(max_age=30, n_init=2)

    face_detector = dlib.get_frontal_face_detector()
    predictor_path = os.path.join(os.path.dirname(__file__), "libary", "shape_predictor_81_face_landmarks.dat")
    if not os.path.exists(predictor_path):
        print(f"Predictor path does not exist: {predictor_path}")
        sys.exit()
    face_predictor = dlib.shape_predictor(predictor_path)

    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoint", "best.pt")
    # Initialize the model
    model = make_model()
    # Load model weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    
    return model, tracker, face_detector, face_predictor


class RealTimeVisionSystem:
    def __init__(self, 
                classifier : make_model,
                tracker : DeepSort,
                face_detector : dlib.get_frontal_face_detector,
                face_predictor : dlib.shape_predictor,
                movie_path : str,
                threshold_fake : float,
                num_frame: int,
                stride : int,
                mode :str):
        
        """
        args:
            move_path (str): Path to the video file to process
            mode (str): Either 'fixed_num_frames' or 'fixed_stride'.
            num_frames (int): Number of frame to extract from video file
            stride: (int): Number of frame to skip between each frame extracted
        """
        self.base_file = os.path.basename(movie_path).split('.')[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sử dụng đường dẫn tương đối
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_videos_folder = os.path.join(base_dir, "static", "videos")
        self.output_props_folder = os.path.join(base_dir, "static", "props")
        self.output_images_folder = os.path.join(base_dir, "static", "images", self.base_file)

        os.makedirs(self.output_images_folder, exist_ok=True)
        os.makedirs(self.output_props_folder, exist_ok=True)
        os.makedirs(self.output_videos_folder, exist_ok=True)
        
        self.classifier = classifier
        self.classifier.eval()
        self.class_transform = TRANSFORM
        self.tracker = tracker
        self.face_detector = face_detector
        self.face_predictor = face_predictor
        self.threshold_fake = threshold_fake

        self.classifier.to(self.device)
        
        
        self.cap = cv2.VideoCapture(movie_path)
        self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        fps_video = int(self.cap.get(cv2.CAP_PROP_FPS))
        width_frame = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height_frame = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_size = (int(width_frame), int(height_frame))
        video_path = f"{self.output_videos_folder}/{os.path.basename(movie_path)}"
        self.video_writer = cv2.VideoWriter(video_path, fourcc, fps_video, frame_size)

        if mode == "fixed_num_frames":
            #Trích xuất các frame đều nhau từ video với số frame bằng num_frame
            self.frames_idx = np.linspace(0, self.total_frame - 1, num_frame, dtype=int,)
        elif mode == "fixed_stride":
            #Trích xuất các frame đều nhau từ video với stride bằng stride
            self.frames_idx = np.arange(0, self.total_frame, stride, dtype=int)


    #Region of Interest
    def predict_class(self, image_roi):
        face_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB)
        face_roi = Image.fromarray(face_roi)
        face_roi = TRANSFORM(face_roi).unsqueeze(0).to(self.device)
        data_dict = {
            'image': face_roi,
            'label': None,
            'mask':None
            }
        with torch.no_grad():
            pred_dict = self.classifier(data_dict)
        score = pred_dict['cls']
        probs = torch.softmax(score, dim=1)
        max_prob, pred = torch.max(probs, dim=1)
        max_prob = np.round(max_prob.item() ,2)
        fake_prob = np.round(probs[0][0].cpu().numpy(), 2)
        real_prob = np.round(probs[0][1].cpu().numpy(), 2)
        return self.get_imagenet_label(pred.item()), max_prob, fake_prob, real_prob


    def get_imagenet_label(self, class_idx):
        labels = { 0: "FAKE", 1: "REAL" }
        return labels.get(class_idx)
    
    def draw_frame(self, frame, bboxs, classes, confidence):
        RED = (0,0,255)
        GREEN = (0,255,0)
        BLUE = (255,0,0)
        # Chuyển bboxs, classes, confidence thành danh sách nếu là float hoặc numpy array
        bboxs = np.atleast_2d(bboxs)  
        classes = np.atleast_1d(classes)  
        confidence = np.atleast_1d(confidence) 
        for bbox, classes, confidence in zip(bboxs, classes, confidence):
            if classes == "FAKE":
                color = RED
            else :
                color = GREEN
            xmin, ymin, xmax, ymax = map(int, bbox)
            w, h = xmax - xmin, ymax - ymin
            # text = f"{classes}: {confidence * 100:.2f}%"
            text = f"{classes}"
            frame = cornerRect(
                        frame,
                        (xmin, ymin, w, h),
                        l = 30,
                        t = 5,
                        rt = 1,
                        colorR=color,
                        colorC=color
                    )
            frame, _ = putTextRect(
                            frame, text, (xmin, ymin - 20),  # Image and starting position of the rectangle
                            scale=2.5, thickness=2,  # Font scale and thickness
                            colorT=(255, 255, 255), colorR=color,  # Text color and Rectangle color
                            font=cv2.FONT_HERSHEY_PLAIN,  # Font type
                            offset=10,  # Offset of text inside the rectangle
                            border=1, colorB=(0, 0, 0)  # Border thickness and color
                        )
        return frame
    
    def save_face(self, face_croped, track_id):
        path = f"{self.output_images_folder}/{track_id}.jpg"
        cv2.imwrite(path, face_croped)

    def run(self):
        bboxs_ = []
        classes_ = []
        confidence_ = []
        arr_confirmed = []
        track_info = defaultdict(list)
        for cnt_frame in tqdm(range(self.total_frame), desc="processing", unit="frames", colour="cyan"):
            ret, frame = self.cap.read()
            org_frame = frame.copy()
            if not ret:
                break
            if cnt_frame in self.frames_idx:
                cropped_faces, landmarks, bboxs =  extract_aligned_face_dlib(self.face_detector, self.face_predictor, frame)
                if cropped_faces is not None:
                    bbs = []
                    fake_probs = []
                    bboxs_, classes_, confidence_ = [], [], []
                    for i, (face_roi, bbox) in enumerate(zip(cropped_faces, bboxs)):
                        classes, confidence, fake_prob, real_prob = self.predict_class(face_roi)
                        frame = self.draw_frame(frame, bbox, classes, confidence)

                        bboxs_.append(bbox)
                        classes_.append(classes)
                        confidence_.append(confidence)

                        xmin, ymin, xmax, ymax = map(int, bbox)
                        w, h = xmax - xmin, ymax - ymin
                        bbs.append(([xmin, ymin, w, h], confidence, classes))
                        fake_probs.append(fake_prob)

                    tracks = self.tracker.update_tracks(bbs, frame = org_frame)
                    for track, fake_prob in zip(tracks, fake_probs):
                        track_info[track.track_id].append(fake_prob)
                        if not track.is_confirmed():
                            continue
                        
                        if track.track_id not in arr_confirmed:
                            ltrb = track.to_ltrb()
                            xmin, ymin, xmax, ymax = map(int, ltrb)
                            face_cropped = org_frame[ymin:ymax, xmin:xmax]
                            dets = self.face_detector(face_cropped, 0)
                            for k, d in enumerate(dets):
                                shape = self.face_predictor(face_cropped, d)
                                # landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
                                for num in range(shape.num_parts):
                                    cv2.circle(face_cropped, (shape.parts()[num].x, shape.parts()[num].y), 2, (255,255,255), -1)
                            self.save_face(face_cropped, track.track_id)
                            
                        arr_confirmed.append(track.track_id)
            else:
                frame = self.draw_frame(frame, bboxs_, classes_, confidence_)            
            self.video_writer.write(frame)
        track_info = {k: round(np.mean(v) * 100, 2) for k, v in track_info.items() if k in arr_confirmed and v}
        with open(f"{self.output_props_folder}/{self.base_file}.json", "w") as f:
            json.dump(track_info, f, default=float)

        self.cap.release()
        self.video_writer.release()
        video_file = f"{self.output_videos_folder}/{self.base_file}.mp4"
        prob_file = f"{self.output_props_folder}/{self.base_file}.json"
        imgs_folder = f"{self.output_images_folder}"



        sorted_track_info = dict(sorted(track_info.items(), key=lambda x: x[1], reverse=True))
        result = "REAL"
        for k, v in sorted_track_info.items():
            if v > self.threshold_fake[1] * 100:
                result = "FAKE"
                break
            elif self.threshold_fake[0] * 100 <= v <= self.threshold_fake[1] * 100:
                result = "CẢNH BÁO !, Có khả năng fake"
                break
        
        result_data = {
            "video_file": video_file,
            "prob_file": prob_file,
            "imgs_folder": imgs_folder,
            "result": result
        }

        pprint(result_data, indent=4)

        return video_file, prob_file, imgs_folder, result

# Khởi chạy hệ thống
if __name__ == "__main__":
    model, tracker, face_detector, face_predictor = get_model()
    # Sử dụng đường dẫn tương đối cho video mẫu
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    org_video = os.path.join(base_dir, "videos", "r3.mp4")
    process_video = RealTimeVisionSystem(model, tracker, face_detector, face_predictor, org_video, 0.6, 32, None, "fixed_num_frames")
    video_file, prob_file, imgs_folder, result = process_video.run()
