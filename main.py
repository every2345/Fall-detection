import cv2
import cvzone
from cvzone.PoseModule import PoseDetector
import matplotlib.pyplot as plt
import mediapipe
import asyncio
import os
from datetime import datetime

# Khởi tạo camera

#video_path = "D:/Work/NCKH/NCKH/falltest9.mp4"
video_path = 'http://100.89.222.35:8000/stream.mjpg'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Khởi tạo PoseDetector
detector = PoseDetector()

# Khởi tạo danh sách lưu số điểm ảnh và thời gian
motion_pixels_list = []
time_list = []
start_time = 0

# Giới hạn số lượng điểm dữ liệu bạn muốn hiển thị trên biểu đồ
max_data_points = 100

# Khởi tạo biến cảnh báo và thời gian bắt đầu cảnh báo
alert = False
fall_detect = False

# Khởi tạo thư mục lưu video ghi lại
output_folder = r"D:\Work\NCKH\Video Record Storage"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
# Khởi tạo biến cho quá trình ghi video
record_start_time = None
video_writer = None

# Tạo một biểu đồ ban đầu và hiển thị nó
plt.ion()  # Bật chế độ tương tác của Matplotlib
fig, ax = plt.subplots()

# Khởi tạo biến để lưu số điểm ảnh tối đa của camera
max_motion_pixels = 0
max_motion_pixels_printed = False  # Biến boolean để kiểm tra xem đã in ra hay chưa

async def process_frames():
    global alert #Biến cảnh báo 
    global fall_detect #Biến thông báo té ngã
    global max_motion_pixels
    global max_motion_pixels_printed
    global record_start_time
    global video_writer
    
    record_start_time = None
    
    while True:
        success, img_original = cap.read() #Đọc khung hình

        if not success:
            print("Không thể đọc khung hình")
            break
        
        if not max_motion_pixels_printed:
            # Tính số điểm ảnh tối đa trong một khung hình
            max_motion_pixels = img_original.shape[0] * img_original.shape[1]
            print("Số điểm ảnh tối đa của camera: ", max_motion_pixels)
            max_motion_pixels_printed = True

        img_display = img_original.copy() #Tạo bản sao từ khung hình gốc
        img_display = detector.findPose(img_display) #Khung hình bản sao sẽ được sử dụng để thực hiện việc đóng khung cơ thể con người và nhận diện chuyển động
        lmlist, bboxInfo = detector.findPosition(img_display, bboxWithHands=True) #Đưa việc đóng khung xương và nhận diện chuyển động vào khung hình bản sao
        
        # Thông báo khi có người xuất hiện trong khung ảnh
        if lmlist:
            print("Người được phát hiện trong khung hình!")

            # Bắt đầu quay video nếu chưa bắt đầu
            if record_start_time is None:
                record_start_time = datetime.now()
                video_name = f"record_{record_start_time.strftime('%Y%m%d%H%M%S')}.avi"
                video_path = os.path.join(output_folder, video_name)
                video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

        else:
            # Ngừng quay video nếu có người mà không được phát hiện
            if record_start_time is not None:
                elapsed_time = datetime.now() - record_start_time
                print(f"Ngừng quay video. Thời gian ghi: {elapsed_time}")
                record_start_time = None

                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
         
        big_frame = cv2.resize(img_display, (img_original.shape[1], img_original.shape[0]))

        # Ghi video nếu đã bắt đầu quay
        if video_writer is not None:
            video_writer.write(big_frame)
    
        ret, frame1 = cap.read() #Đọc khung hình 1
        ret, frame2 = cap.read() #Đọc khung hình 2

        if not ret:
            print("Khung hình rỗng")
            break

        #Chuyển ảnh màu thành ảnh nhị phân
        frameDiff = cv2.absdiff(frame1, frame2)
        grayDiff = cv2.cvtColor(frameDiff, cv2.COLOR_BGR2GRAY)
        _, colorDiff = cv2.threshold(grayDiff, 30, 255, cv2.THRESH_BINARY)

        motion_pixels = cv2.countNonZero(colorDiff) #Đếm số pixel có giá trị khác 0 (pixel trắng) trong ảnh nhị phân

        current_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() #Tính thời gian hiện tại bằng
        
        # Tính toán giá trị ngưỡng 
        threshold = motion_pixels / max_motion_pixels        
    
        motion_pixels_list.append(motion_pixels) #Thêm số pixel có sự khác biệt và thời gian tương ứng vào danh sách Motion_pixels_list
        time_list.append(current_time) #Thêm thời gian hiện tại vào danh sách current time 
    
        if len(motion_pixels_list) > max_data_points: 
            motion_pixels_list.pop(0)
            time_list.pop(0) #Sử dụng dữ liêu có trong time_list và motion_pixels_list để giới hạn khung hình biểu diễn trong đồ thị điểm ảnh

        big_frame = cv2.resize(img_display, (img_original.shape[1], img_original.shape[0])) #Gom tất các khung hình rời rạc lại thành một khung hình chung "Big Frame"


        #Xử lí té ngã
        if bboxInfo:
            x, y, w, h = bboxInfo["bbox"]
            length_horizontal = w
            length_vertical = h
            line_spacing = 10

            font_scale = 3
            font_thickness = 3

            cv2.putText(big_frame, f"Be Rong: {length_horizontal}px", (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 0, 0), font_thickness)
            cv2.putText(big_frame, f"Chieu Cao: {length_vertical}px", (10, 60 + line_spacing),
                        cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), font_thickness)
            
            if length_horizontal > 0:
                if threshold > 0.01 and (length_vertical < length_horizontal and length_horizontal - length_vertical <= 150):
                    if not alert:
                        alert = True
                else:
                    alert = False
                    fall_detect = False

                if alert:
                    print("Fall Detect")
                    cv2.putText(big_frame, "Fall Detect", (x, y - 50),
                                cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), font_thickness)
                    fall_detect = True
                    
            if length_horizontal < 0:
                if threshold > 0.01 and length_vertical + length_horizontal <= 380:
                    if not alert:
                        alert = True
                else:
                    alert = False
                    fall_detect = False

                if alert:
                    print("Fall Detect")
                    cv2.putText(big_frame, "Fall Detect", (x, y - 50),
                                cv2.FONT_HERSHEY_PLAIN, font_scale, (0, 0, 255), font_thickness)
                    fall_detect = True

        if not ret:
            print("Khung hình rỗng")
            break
        
        
        # Cập nhật biểu đồ trong thời gian thực
        ax.clear()
        ax.plot(time_list, motion_pixels_list, 'b-')
        ax.set_xlabel('Thời gian (s)')
        ax.set_ylabel('Số điểm ảnh')
        ax.set_title('Biểu đồ động số điểm ảnh theo thời gian')
        current_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if current_time > 10:
            ax.set_xlim(current_time - 10, current_time)
        ax.set_ylim(0, 307200)
        plt.gca().set_xticks([])
        plt.pause(0.01)

        #Chỉnh sửa độ phân giải
        frame_ratio = frame1.shape[1] / frame1.shape[0]
        new_width = 520   
        new_height = int(new_width / frame_ratio)
        frame1 = cv2.resize(frame1, (new_width, new_height))

        colorDiff_colored = cv2.cvtColor(colorDiff, cv2.COLOR_GRAY2BGR)

        frame1_resized = cv2.resize(frame1, (new_width, new_height))
        big_frame_resized = cv2.resize(big_frame, (new_width, new_height))
        colorDiff_resized = cv2.resize(colorDiff_colored, (new_width, new_height))

        combined_row = cv2.hconcat([frame1_resized, big_frame_resized, colorDiff_resized])
        
        await asyncio.sleep(0)       
        
        cv2.imshow("Combined Interfaces", combined_row)

        if cv2.waitKey(1) == 27:
            break


loop = asyncio.get_event_loop()

loop.run_until_complete(process_frames())

cap.release()
cv2.destroyAllWindows()
