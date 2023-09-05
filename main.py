import time
from time import sleep
import cv2
import cvzone
from cvzone.PoseModule import PoseDetector
import matplotlib.pyplot as plt
import asyncio

video_path = 0
cap = cv2.VideoCapture(video_path)  # Mở camera

if not cap.isOpened():  # Kiểm tra camera có được mở hay không
    print("Không thể mở camera")
    exit()

# Khởi tạo PoseDetector
detector = PoseDetector()
fpsReader = cvzone.FPS()


# Tạo danh sách lưu số điểm ảnh và thời gian
motion_pixels_list = []
time_list = []
start_time = 0

# Giới hạn số lượng điểm dữ liệu bạn muốn hiển thị trên biểu đồ
max_data_points = 100
data_points_to_remove = 10

# Khởi tạo biến cảnh báo và thời gian bắt đầu cảnh báo
alert = False
fall_detect = False

async def process_frames():
    global alert
    global fall_detect
    while True:
        success, img_original = cap.read()

        if not success:
            print("Không thể đọc khung hình")
            break

        img_display = img_original.copy()
        img_display = detector.findPose(img_display)
        lmlist, bboxInfo = detector.findPosition(img_display, bboxWithHands=True)
    
        ret, frame1 = cap.read()  # Đọc khung hình hiện tại
        ret, frame2 = cap.read()  # Đọc khung hình tiếp theo

        if not ret:  # Kiểm tra khung hình rỗng
            print("Khung hình rỗng")
            break

        frameDiff = cv2.absdiff(frame1, frame2)  # Tính toán khung hình khác biệt giữa hai khung hình liên tiếp
        grayDiff = cv2.cvtColor(frameDiff, cv2.COLOR_BGR2GRAY)  # Chuyển khung hình khác biệt sang ảnh xám
        _, colorDiff = cv2.threshold(grayDiff, 50, 255, cv2.THRESH_BINARY)  # Chuyển ảnh xám thành ảnh nhị phân
        motion_pixels = cv2.countNonZero(colorDiff)  # Đếm số pixel có chuyển động

        # Tính thời gian hiện tại
        current_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    
        # Thêm số điểm ảnh và thời gian vào danh sách
        motion_pixels_list.append(motion_pixels)
        time_list.append(current_time)
    
        # Giới hạn số lượng điểm dữ liệu trên biểu đồ
        if len(motion_pixels_list) > max_data_points:
            motion_pixels_list.pop(0)
            time_list.pop(0)

        # Tạo khung hình lớn để hiển thị các giao diện
        big_frame = cv2.resize(img_display, (img_original.shape[1], img_original.shape[0]))

        if bboxInfo:
            x, y, w, h = bboxInfo["bbox"]
            length_horizontal = w
            length_vertical = h
            
            # Điều chỉnh khoảng cách giữa hai dòng chữ
            line_spacing = 10  # Điều chỉnh khoảng cách theo mong muốn

            # Vẽ thông tin về cạnh bên ngang và bên dọc lên khung hình lớn
            font_scale = 3  # Điều chỉnh kích thước chữ
            font_thickness = 3  # Điều chỉnh độ dày của chữ

            cv2.putText(big_frame, f"Be Rong: {length_horizontal}px", (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 0, 0), font_thickness)
            cv2.putText(big_frame, f"Chieu Cao: {length_vertical}px", (10, 60 + line_spacing),
                        cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), font_thickness)
            
            # Kiểm tra điều kiện phát hiện té ngã
            if (length_horizontal > 0): #Be rong mang gia tri duong
                if (motion_pixels > 22000 and (length_vertical < length_horizontal and length_horizontal - length_vertical <= 150)):
                    if not alert:
                        # Bắt đầu cảnh báo
                        alert = True
                else:
                    # Đặt lại biến cảnh báo và trạng thái phát hiện té ngã nếu chiều cao > chiều rộng
                    alert = False
                    fall_detect = False

                # Hiển thị thông báo nếu đang trong trạng thái cảnh báo
                if alert:
                    # In cảnh báo "Phát hiện Té ngã"
                    print("Fall Detect")
                    cv2.putText(big_frame, "Fall Detect", (x, y - 50),
                                cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), font_thickness)
                    fall_detect = True
            
            if (length_horizontal < 0): #Be rong mang gia tri am
                if (motion_pixels > 22000 and length_vertical + length_horizontal <= 380):
                    if not alert:
                        # Bắt đầu cảnh báo
                        alert = True
                else:
                    # Đặt lại biến cảnh báo và trạng thái phát hiện té ngã nếu chiều cao > chiều rộng
                    alert = False
                    fall_detect = False

                # Hiển thị thông báo nếu đang trong trạng thái cảnh báo
                if alert:
                    # In cảnh báo "Phát hiện Té ngã"
                    print("Fall Detect")
                    cv2.putText(big_frame, "Fall Detect", (x, y - 50),
                                cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), font_thickness)
                    fall_detect = True

        if not ret:  # Kiểm tra khung hình rỗng
            print("Khung hình rỗng")
            break            

        fps, frame1 = fpsReader.update(frame1, pos=(10, 30), color=(0, 255, 0), scale=2, thickness=3)    

        frame_ratio = frame1.shape[1] / frame1.shape[0]
        new_width = 520   
        new_height = int(new_width / frame_ratio)
        frame1 = cv2.resize(frame1, (new_width, new_height))

        # Chuyển đổi khung hình thành ảnh màu để hiển thị
        colorDiff_colored = cv2.cvtColor(colorDiff, cv2.COLOR_GRAY2BGR)

        # Tạo layout cho các giao diện
        frame1_resized = cv2.resize(frame1, (new_width, new_height))
        big_frame_resized = cv2.resize(big_frame, (new_width, new_height))
        colorDiff_resized = cv2.resize(colorDiff_colored, (new_width, new_height))

        # Kết hợp tất cả các giao diện cùng một hàng ngang
        combined_row = cv2.hconcat([frame1_resized, big_frame_resized, colorDiff_resized])
        
        cv2.imshow("Combined Interfaces", combined_row)  # Hiển thị giao diện kết hợp

        if cv2.waitKey(1) == 27:  # Thoát nếu nhấn phím ESC
            break
        # Vẽ biểu đồ động
        plt.clf()  # Xóa biểu đồ trước khi vẽ lại
        plt.plot(time_list, motion_pixels_list, 'b-')
        plt.xlabel('Thời gian (s)')
        plt.ylabel('Số điểm ảnh')
        plt.title('Biểu đồ động số điểm ảnh theo thời gian')

        # Tính toán giới hạn x cho biểu đồ
        if current_time > 10:
            plt.xlim(current_time - 10, current_time)
        
        # Đặt giới hạn trục tung từ 0 đến 40000
        plt.ylim(0, 60000)
        
        # Đặt giá trị trống cho trục ngang
        plt.gca().set_xticks([])

        plt.pause(0.01)
        
        await asyncio.sleep(0)       

# Khởi tạo event loop asyncio
loop = asyncio.get_event_loop()

# Bắt đầu chạy luồng xử lý chuyển động và cập nhật biểu đồ
loop.run_until_complete(process_frames())

cap.release()
cv2.destroyAllWindows()
