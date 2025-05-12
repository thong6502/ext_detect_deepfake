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
import concurrent.futures
from functools import lru_cache

from .config import TRANSFORM
from pprint import pprint
from PIL import Image
from tqdm import tqdm
import sys
import json
from cvzone import putTextRect, cornerRect

@lru_cache(maxsize=1)
def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    model.eval()  # Đặt model ở chế độ evaluation
    
    return model, face_detector, face_predictor


class RealTimeVisionSystem:
    def __init__(self, 
                classifier : make_model,
                face_detector : dlib.get_frontal_face_detector,
                face_predictor : dlib.shape_predictor,
                movie_path : str,
                threshold_fake : float,
                num_frame: int,
                stride : int,
                mode :str,
                batch_size: int = 4):  # Thêm batch_size
        
        """
        args:
            move_path (str): Path to the video file to process
            mode (str): Either 'fixed_num_frames' or 'fixed_stride'.
            num_frames (int): Number of frame to extract from video file
            stride: (int): Number of frame to skip between each frame extracted
            batch_size (int): Số lượng khuôn mặt xử lý cùng lúc
        """
        self.base_file = os.path.basename(movie_path).split('.')[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
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
        # Tắt gradient calculation để tăng tốc
        self.use_grad = torch.no_grad()
        self.use_grad.__enter__()
        
        self.class_transform = TRANSFORM
        self.tracker = self.reset_tracker()
        self.face_detector = face_detector
        self.face_predictor = face_predictor
        self.threshold_fake = threshold_fake

        self.classifier.to(self.device)
        
        
        self.cap = cv2.VideoCapture(movie_path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Tối ưu buffer
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

        # Khởi tạo thread pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        # Cache kết quả phân loại
        self.class_cache = {}

    def __del__(self):
        # Đảm bảo giải phóng tài nguyên
        if hasattr(self, 'use_grad'):
            self.use_grad.__exit__(None, None, None)
        if hasattr(self, 'executor'):
            self.executor.shutdown()
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'video_writer'):
            self.video_writer.release()

    #Region of Interest
    def predict_class(self, image_roi):
        # Kiểm tra cache trước
        image_hash = hash(image_roi.tobytes())
        if image_hash in self.class_cache:
            return self.class_cache[image_hash]
            
        face_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB)
        face_roi = Image.fromarray(face_roi)
        face_roi = TRANSFORM(face_roi).unsqueeze(0).to(self.device)
        data_dict = {
            'image': face_roi,
            'label': None,
            'mask': None
            }
        
        score = self.classifier(data_dict)['cls']
        probs = torch.softmax(score, dim=1)
        max_prob, pred = torch.max(probs, dim=1)
        max_prob = np.round(max_prob.item(), 2)
        fake_prob = np.round(probs[0][0].cpu().numpy(), 2)
        real_prob = np.round(probs[0][1].cpu().numpy(), 2)
        
        result = (self.get_imagenet_label(pred.item()), max_prob, fake_prob, real_prob)
        # Lưu kết quả vào cache
        self.class_cache[image_hash] = result
        return result

    def predict_batch(self, faces):
        """Xử lý nhiều khuôn mặt cùng lúc"""
        if not faces:
            return []
            
        batch_inputs = []
        for face in faces:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = TRANSFORM(face_pil).unsqueeze(0)
            batch_inputs.append(face_tensor)
            
        # Ghép các tensor thành một batch
        batch_tensor = torch.cat(batch_inputs).to(self.device)
        
        data_dict = {
            'image': batch_tensor,
            'label': None,
            'mask': None
        }
        
        scores = self.classifier(data_dict)['cls']
        probs = torch.softmax(scores, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        
        results = []
        for i in range(len(faces)):
            pred = preds[i].item()
            max_prob = np.round(max_probs[i].item(), 2)
            fake_prob = np.round(probs[i][0].cpu().numpy(), 2)
            real_prob = np.round(probs[i][1].cpu().numpy(), 2)
            results.append((self.get_imagenet_label(pred), max_prob, fake_prob, real_prob))
            
        return results

    def get_imagenet_label(self, class_idx):
        labels = { 0: "FAKE", 1: "REAL" }
        return labels.get(class_idx)
    
    def draw_frame(self, frame, bboxs, classes, confidence):
        RED = (0,0,255)
        GREEN = (0,255,0)
        BLUE = (255,0,0)
        
        # Không vẽ nếu không có bounding box
        if len(bboxs) == 0:
            return frame
            
        # Chuyển bboxs, classes, confidence thành danh sách nếu là float hoặc numpy array
        bboxs = np.atleast_2d(bboxs)  
        classes = np.atleast_1d(classes)  
        confidence = np.atleast_1d(confidence) 
        
        for bbox, cls, conf in zip(bboxs, classes, confidence):
            if cls == "FAKE":
                color = RED
            else:
                color = GREEN
            xmin, ymin, xmax, ymax = map(int, bbox)
            w, h = xmax - xmin, ymax - ymin
            text = f"{cls}"
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
                            frame, text, (xmin, ymax + 30),  # Image and starting position of the rectangle
                            scale=2.5, thickness=2,  # Font scale and thickness
                            colorT=(255, 255, 255), colorR=color,  # Text color and Rectangle color
                            font=cv2.FONT_HERSHEY_PLAIN,  # Font type
                            offset=10,  # Offset of text inside the rectangle
                            border=1, colorB=(0, 0, 0)  # Border thickness and color
                        )
        return frame
    
    def save_face(self, face_croped, track_id):
        path = f"{self.output_images_folder}/{track_id}.jpg"
        # Kiểm tra hình ảnh có rỗng không trước khi lưu
        if face_croped is not None and face_croped.size > 0 and not np.all(face_croped == 0):
            try:
                # Sử dụng flag IMWRITE_JPEG_QUALITY để tối ưu hóa kích thước
                cv2.imwrite(path, face_croped, [cv2.IMWRITE_JPEG_QUALITY, 90])
            except Exception as e:
                print(f"Lỗi khi lưu ảnh: {e}")
        else:
            print(f"Không thể lưu ảnh cho track_id {track_id}: Hình ảnh trống")

    def reset_tracker(self):
        """
        Reset lại tracker để tránh việc tích lũy thông tin tracking từ các lần chạy trước
        """
        return DeepSort(max_age=30, n_init=2)
        
    def process_face_batch(self, faces, bboxs):
        """Xử lý một batch các khuôn mặt"""
        if not faces:
            return [], [], []
            
        # Xử lý prediction theo batch
        batch_results = self.predict_batch(faces)
        
        bboxs_ = []
        classes_ = []
        confidence_ = []
        bbs = []
        fake_probs = []
        
        for i, ((cls, conf, fake_prob, real_prob), bbox) in enumerate(zip(batch_results, bboxs)):
            bboxs_.append(bbox)
            classes_.append(cls)
            confidence_.append(conf)
            
            xmin, ymin, xmax, ymax = map(int, bbox)
            w, h = xmax - xmin, ymax - ymin
            bbs.append(([xmin, ymin, w, h], conf, cls))
            fake_probs.append(fake_prob)
            
        return bbs, fake_probs, (bboxs_, classes_, confidence_)
        
    def run(self):
        # Reset tracker trước khi bắt đầu xử lý video mới
        self.reset_tracker()
        
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
                # Trích xuất khuôn mặt
                cropped_faces, landmarks, bboxs = extract_aligned_face_dlib(self.face_detector, self.face_predictor, frame)
                
                if cropped_faces is not None:
                    # Xử lý theo batch
                    face_batches = [cropped_faces[i:i+self.batch_size] for i in range(0, len(cropped_faces), self.batch_size)]
                    bbox_batches = [bboxs[i:i+self.batch_size] for i in range(0, len(bboxs), self.batch_size)]
                    
                    all_bbs = []
                    all_fake_probs = []
                    bboxs_, classes_, confidence_ = [], [], []
                    
                    # Xử lý từng batch
                    for face_batch, bbox_batch in zip(face_batches, bbox_batches):
                        bbs, fake_probs, drawing_data = self.process_face_batch(face_batch, bbox_batch)
                        all_bbs.extend(bbs)
                        all_fake_probs.extend(fake_probs)
                        
                        batch_bboxs, batch_classes, batch_confidence = drawing_data
                        for i, (bbox, cls, conf) in enumerate(zip(batch_bboxs, batch_classes, batch_confidence)):
                            frame = self.draw_frame(frame, bbox, cls, conf)
                            bboxs_.append(bbox)
                            classes_.append(cls)
                            confidence_.append(conf)
                    
                    # Cập nhật các tracks
                    tracks = self.tracker.update_tracks(all_bbs, frame=org_frame)
                    
                    # Xử lý các track đã xác nhận
                    save_faces_futures = []
                    
                    for track, fake_prob in zip(tracks, all_fake_probs):
                        track_info[track.track_id].append(fake_prob)
                        if not track.is_confirmed():
                            continue
                        
                        if track.track_id not in arr_confirmed:
                            ltrb = track.to_ltrb()
                            
                            # Đảm bảo tọa độ nằm trong giới hạn khung hình
                            left, top, right, bottom = ltrb
                            left = max(0, left)
                            top = max(0, top)
                            right = min(org_frame.shape[1], right)
                            bottom = min(org_frame.shape[0], bottom)
                            
                            # Kiểm tra xem kích thước có hợp lệ không
                            if right - left > 10 and bottom - top > 10:
                                xmin, ymin, xmax, ymax = int(left), int(top), int(right), int(bottom)
                                face_cropped = org_frame[ymin:ymax, xmin:xmax]
                                
                                # Đảm bảo face_cropped có kích thước dương
                                if face_cropped.size > 0:
                                    try:
                                        dets = self.face_detector(face_cropped, 0)
                                        if len(dets) > 0:  # Đảm bảo tìm thấy khuôn mặt
                                            # Lưu mặt bất đồng bộ
                                            future = self.executor.submit(self.save_face, face_cropped.copy(), track.track_id)
                                            save_faces_futures.append(future)
                                    except Exception as e:
                                        print(f"Lỗi xử lý khuôn mặt: {e}")
                            else:
                                print(f"Bỏ qua ID: {track.track_id} do kích thước không đủ: {right-left}x{bottom-top}")
                                
                        arr_confirmed.append(track.track_id)
                    
                    # Chờ hoàn thành việc lưu ảnh
                    for future in save_faces_futures:
                        future.result()
            else:
                # Vẽ lại kết quả cho frame không xử lý
                frame = self.draw_frame(frame, bboxs_, classes_, confidence_)
                        
            self.video_writer.write(frame)
        
        # Tính toán kết quả cuối cùng
        track_info = {k: round(np.mean(v) * 100, 2) for k, v in track_info.items() if k in arr_confirmed and v}
        with open(f"{self.output_props_folder}/{self.base_file}.json", "w") as f:
            json.dump(track_info, f, default=float)

        self.cap.release()
        self.video_writer.release()
        
        # Giải phóng tài nguyên
        self.executor.shutdown()
        self.use_grad.__exit__(None, None, None)
        
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
                result = "CẢNH BÁO"
                break

        return video_file, prob_file, imgs_folder, result

# Khởi chạy hệ thống
if __name__ == "__main__":
    model, face_detector, face_predictor = get_model()
    # Sử dụng đường dẫn tương đối cho video mẫu
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    org_video = os.path.join(base_dir, "videos", "r3.mp4")
    process_video = RealTimeVisionSystem(model, face_detector, face_predictor, org_video, 0.6, 32, None, "fixed_num_frames", batch_size=8)
    video_file, prob_file, imgs_folder, result = process_video.run()
