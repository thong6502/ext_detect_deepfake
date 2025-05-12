<p align="center">
 <h1 align="center">Hệ thống phát hiện Video Deepfake</h1>
</p>

## Giới thiệu

Hệ thống nhận diện video deepfake tích hợp với tiện ích mở rộng trình duyệt, cho phép phát hiện và phân tích video giả mạo trên các nền tảng phổ biến như YouTube

## Demo

<p align="center">
  <img src="demo.gif" width=600><br/>
  <i>Camera app demo</i>
</p>

## Tổng quan

Dự án này gồm hai phần chính:

1. **Backend API Flask**: Phân tích video và phát hiện deepfake
2. **Tiện ích mở rộng Chrome**: Tích hợp với các nền tảng video phổ biến

Hệ thống sử dụng mô hình học sâu Xception được huấn luyện để phân biệt giữa video thật và deepfake, và cung cấp kết quả phân tích chi tiết.

## Tính năng chính

- Phát hiện deepfake trong video trực tuyến trên YouTube
- Phân tích từng khuôn mặt xuất hiện trong video
- Phân loại video theo độ tin cậy (Thật/Giả/Cảnh báo)
- Hiển thị xác suất phát hiện deepfake cho từng khuôn mặt
- Tìm và theo dõi khuôn mặt xuyên suốt video
- Giao diện web thân thiện hiển thị kết quả phân tích

## Cách hoạt động

1. Tiện ích mở rộng thêm nút "Check video" vào YouTube
2. Người dùng nhấn nút để gửi URL video đến backend
3. Backend tải video, trích xuất và căn chỉnh các khuôn mặt
4. Mô hình Xception phân tích từng khuôn mặt và đưa ra xác suất deepfake
5. DeepSort theo dõi khuôn mặt xuyên suốt video để phân tích nhất quán
6. Kết quả được hiển thị trên trang web với đánh dấu màu:
   - Xanh lá: Video thật
   - Cam: Cảnh báo có thể là deepfake
   - Đỏ: Video deepfake

## Công nghệ sử dụng

### Backend

- Flask: Web framework xử lý API
- PyTorch: Framework deep learning
- DeepSort: Thuật toán theo dõi đối tượng
- Dlib: Phát hiện khuôn mặt và landmark
- OpenCV: Xử lý hình ảnh và video
- Xception: Kiến trúc mạng neural CNN cho phân loại deepfake
- Ngrok: Phơi API ra internet
- yt-dlp: Tải video từ các nền tảng

### Tiện ích mở rộng Chrome

- Chrome Extension API
- JavaScript/HTML/CSS
- Tích hợp với YouTube

## Cài đặt

### Yêu cầu

- Python 3.10
- CUDA (khuyến nghị)
- Chrome browser

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Tải trọng số model

Để hệ thống hoạt động, bạn cần tải các file **Trọng số mô hình (best.pt)**: [Tải tại đây](https://drive.google.com/file/d/1EIQP6N-LH-3oNf0G9aVv91Nv38edra_z/view?usp=sharing):
- Đặt file `best.pt` vào thư mục `checkpoint/`

### Cài đặt tiện ích mở rộng Chrome

1. Mở Chrome và truy cập `chrome://extensions/`
2. Bật chế độ Developer mode (góc phải)
3. Chọn "Load unpacked" và chọn thư mục "Deepfake-extension"

### Cấu hình

1. Cập nhật file `ngrok.yaml` với YOUR_NGROK_AUTH_TOKEN của bạn
2. Kiểm tra đường dẫn mô hình trong `src/utils.py`

## Sử dụng

### Chạy backend

```bash
python backend.py
```

Backend sẽ khởi động và tạo một đường hầm Ngrok để công khai API.

### Sử dụng tiện ích

1. Truy cập video trên YouTube, Facebook hoặc TikTok
2. Nhấn nút "Check video" xuất hiện trên trang
3. Đợi trong khi backend phân tích video
4. Xem kết quả chi tiết:
   - Danh sách các khuôn mặt được phát hiện
   - Xác suất là deepfake cho từng khuôn mặt
   - Video đã phân tích với các đánh dấu

## Cấu trúc dự án

- `backend.py`: API Flask chính
- `src/`: Mã nguồn chính của hệ thống
  - `model.py`: Định nghĩa mô hình Xception
  - `utils.py`: Các hàm tiện ích và lớp xử lý video
  - `config.py`: Cấu hình ứng dụng
  - `libary/`: Thư viện phụ trợ
  - `downloader.py`: Tải video từ các nền tảng
- `Deepfake-extension/`: Mã nguồn tiện ích Chrome
  - `manifest.json`: Cấu hình tiện ích
  - `script.js`: Mã JavaScript chính
- `static/`: Lưu trữ kết quả phân tích
- `templates/`: Template HTML
- `checkpoint/`: Lưu trữ mô hình đã huấn luyện

## Phát triển

Dự án sử dụng mô hình Xception được huấn luyện trên tập dữ liệu Deepfake, cho phép phát hiện video giả mạo với độ chính xác cao. Hệ thống được tối ưu để chạy trên các GPU NVIDIA để xử lý video nhanh chóng.

### Ngưỡng phát hiện

Hệ thống sử dụng hai ngưỡng để phân loại video:

- Ngưỡng thấp (30%): Dưới ngưỡng này, video được phân loại là thật
- Ngưỡng cao (70%): Trên ngưỡng này, video được phân loại là giả
- Giữa hai ngưỡng: Video được đánh dấu cảnh báo

### Tối ưu hiệu suất

- Sử dụng phương pháp stride để trích xuất khung hình hiệu quả
- Căn chỉnh khuôn mặt trước khi phân tích
- Theo dõi khuôn mặt để giảm thiểu phân tích trùng lặp

## Thử nghiệm

<img src="image\metric.png" width="800">
