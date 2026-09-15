BẢN FIX ẨN THÔNG TIN MODEL PHỤ, GIỮ CROP TƯƠNG TÁC

Cách thay:
1. Copy app.py, index.html, requirements.txt trong thư mục này đè vào folder Food_detection hiện tại.
2. Đảm bảo model.pt vẫn nằm ở một trong hai vị trí:
   - Food_detection/model/model.pt
   - Food_detection/model.pt
3. Chạy:
   pip install -r requirements.txt
   python app.py
4. Mở:
   http://localhost:5000
5. Nếu trình duyệt còn giao diện cũ, bấm Ctrl + F5.

Ghi chú:
- Giao diện chỉ hiển thị là CNN.
- Thumbnail kết quả không vẽ khung phụ lên ảnh.
- Kết quả vẫn phân biệt Thịt kho và Thịt kho trứng, nhưng không hiển thị thông tin về cơ chế nội bộ.
- Vẫn giữ: kéo/resize khung crop, ảnh không bị xén, lưu khung đã chỉnh bằng localStorage.
- Khung crop được khóa tỷ lệ 1:1 trên giao diện.
- Backend tự cắt vuông ở chính giữa nếu nhận cấu hình khung chữ nhật cũ; phần dư bị bỏ, không padding và không kéo méo.
- Crop vuông được resize về 128x128 trước khi đưa vào model H5.
- Giữ nguyên toàn bộ tên class hiện tại, bao gồm “Canh rau muống”.

- Đã thêm preprocessing giảm chói sáng trước CNN:
  + Tự đo độ sáng và tỷ lệ vùng phản sáng trên từng crop.
  + Ảnh bình thường được giữ nguyên.
  + Ảnh quá sáng được gamma correction thích ứng, nén điểm trắng và CLAHE nhẹ trên kênh sáng LAB.
  + Ảnh gốc vẫn dùng cho thumbnail và phần kiểm tra trứng; preprocessing chỉ áp dụng cho model H5.
  + Response backend có trường lighting để kiểm thử; giao diện không hiển thị trường này.
