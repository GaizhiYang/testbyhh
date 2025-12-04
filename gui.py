import threading
import time
import random
import base64
from argparse import Namespace
from queue import Queue, Full, Empty

import cv2

try:
    # 延迟导入，若环境缺少 RDK 依赖，可在界面提示
    from yolo_mycam import YOLO11_Detect, draw_detection, coco_names
    YOLO_IMPORTED = True
    YOLO_IMPORT_ERROR = None
except Exception as _e:
    YOLO_IMPORTED = False
    YOLO_IMPORT_ERROR = _e

import tkinter as tk
from tkinter import ttk, messagebox


class DetectorThread(threading.Thread):
    def __init__(self, frame_queue: Queue, stop_event: threading.Event, score_threshold: float = 0.5, camera_index: int = 1):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.camera_index = camera_index
        self.score_threshold = score_threshold
        self.model = None
        self.cap = None

    def _init_model(self):
        # 与 yolo_mycam.py 保持一致的默认参数，但调整 score_thres=0.5 以满足需求
        opt = Namespace(
            model_path='converted_model_modified_v1.bin',
            test_img='img001.jpg',
            img_save_path='py_result.jpg',
            classes_num=10,
            nms_thres=0.7,
            score_thres=self.score_threshold,
            reg=16,
        )
        self.model = YOLO11_Detect(opt)

    def _init_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        # 分辨率与原脚本一致
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _encode_frame_png_b64(self, frame_bgr):
        ok, buf = cv2.imencode('.png', frame_bgr)
        if not ok:
            return None
        return base64.b64encode(buf).decode('ascii')

    def run(self):
        try:
            self._init_model()
        except Exception as e:
            # 将错误传回主线程显示
            try:
                self.frame_queue.put_nowait(('__error__', f'模型加载失败: {e}'))
            except Full:
                pass
            return

        try:
            self._init_camera()
            if not self.cap.isOpened():
                try:
                    self.frame_queue.put_nowait(('__error__', '无法打开摄像头'))
                except Full:
                    pass
                return
        except Exception as e:
            try:
                self.frame_queue.put_nowait(('__error__', f'摄像头初始化失败: {e}'))
            except Full:
                pass
            return

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                # 摄像头错误
                try:
                    self.frame_queue.put_nowait(('__error__', '摄像头取帧失败'))
                except Full:
                    pass
                break

            try:
                input_tensor = self.model.preprocess_yuv420sp(frame)
                outputs = self.model.c2numpy(self.model.forward(input_tensor))
                results = self.model.postProcess(outputs)

                # 仅保留置信度>=0.5 的数字用于右侧显示
                digits = []
                for class_id, score, x1, y1, x2, y2 in results:
                    if score >= self.score_threshold:
                        digits.append(str(coco_names[class_id]))
                        draw_detection(frame, (x1, y1, x2, y2), score, class_id)

                # 将帧编码后发送给主线程刷新 UI
                png_b64 = self._encode_frame_png_b64(frame)
                if png_b64 is not None:
                    # 丢弃旧帧，始终保留最新
                    while True:
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            break
                        except Exception:
                            break
                    try:
                        self.frame_queue.put_nowait((png_b64, digits))
                    except Full:
                        pass
            except Exception as e:
                # 推理异常
                try:
                    self.frame_queue.put_nowait(('__error__', f'推理异常: {e}'))
                except Full:
                    pass
                break

        # 清理资源
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('数字检测人机界面')

        # 状态
        self.detector_thread = None
        self.detector_stop = threading.Event()
        self.frame_queue = Queue(maxsize=1)
        self.running = False

        # 右侧信息
        self.digits_last = None
        self.start_epoch = None
        self.metrics_job = None
        self.queue_job = None
        self.current_image = None  # 持有 PhotoImage 引用

        # 路程计算（速度随机，单位以 m/s 计）
        self.current_speed = 0.0
        self.total_distance = 0.0
        self.last_metrics_ts = None

        self._build_ui()

    def _build_ui(self):
        self.root.geometry('1000x600')

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # 左侧视频区域
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_label = ttk.Label(left, text='视频预览区', anchor='center')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(left)
        controls.pack(fill=tk.X, pady=(10, 0))

        self.btn_start = ttk.Button(controls, text='开始运行', command=self.on_start)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_stop = ttk.Button(controls, text='停止', command=self.on_stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT)

        # 右侧信息区域
        right = ttk.Frame(main, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # 检测到的数字（仅当有>=0.5的结果时更新）
        ttk.Label(right, text='检测数字(≥0.5)：').pack(anchor='w')
        self.digits_var = tk.StringVar(value='—')
        self.digits_label = ttk.Label(right, textvariable=self.digits_var, font=('Arial', 18))
        self.digits_label.pack(anchor='w', pady=(0, 10))

        # 运行时间、速度、路程
        ttk.Label(right, text='运行时间：').pack(anchor='w')
        self.runtime_var = tk.StringVar(value='00:00:00')
        ttk.Label(right, textvariable=self.runtime_var, font=('Arial', 12)).pack(anchor='w', pady=(0, 6))

        ttk.Label(right, text='速度 (m/s)：').pack(anchor='w')
        self.speed_var = tk.StringVar(value='0.00')
        ttk.Label(right, textvariable=self.speed_var, font=('Arial', 12)).pack(anchor='w', pady=(0, 6))

        ttk.Label(right, text='路程 (m)：').pack(anchor='w')
        self.distance_var = tk.StringVar(value='0.00')
        ttk.Label(right, textvariable=self.distance_var, font=('Arial', 12)).pack(anchor='w', pady=(0, 6))

    def _update_runtime_metrics(self):
        if not self.running:
            return
        now = time.time()
        if self.last_metrics_ts is None:
            self.last_metrics_ts = now

        # 更新时间
        elapsed = int(now - self.start_epoch) if self.start_epoch else 0
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        self.runtime_var.set(f'{h:02d}:{m:02d}:{s:02d}')

        # 随机速度（每次刷新随机一个区间值）
        self.current_speed = random.uniform(1.2, 3.5)
        self.speed_var.set(f'{self.current_speed:.2f}')

        # 累计路程 = 上次到现在的时间间隔 * 速度（简单积分）
        dt = max(0.0, now - self.last_metrics_ts)
        self.total_distance += self.current_speed * dt
        self.distance_var.set(f'{self.total_distance:.2f}')

        self.last_metrics_ts = now
        self.metrics_job = self.root.after(1000, self._update_runtime_metrics)

    def _process_queue(self):
        if not self.running:
            return
        latest = None
        try:
            while True:
                item = self.frame_queue.get_nowait()
                latest = item
        except Empty:
            pass
        except Exception:
            latest = None

        if latest is not None:
            tag, payload = latest
            if tag == '__error__':
                messagebox.showerror('运行错误', str(payload))
                self.on_stop()
            else:
                # 显示图像
                try:
                    img = tk.PhotoImage(data=tag)  # tag 为 png 的 base64 字符串
                    self.current_image = img
                    self.video_label.configure(image=img, text='')
                except Exception:
                    # 图像显示失败不应终止
                    pass
                # 更新右侧数字（仅当有>=0.5的检测结果时才更新）
                digits = payload or []
                if digits:
                    # 去重后有序显示
                    uniq = sorted(set(digits), key=lambda x: int(x) if x.isdigit() else x)
                    text = ' '.join(uniq)
                    if text != self.digits_last:
                        self.digits_var.set(text)
                        self.digits_last = text

        self.queue_job = self.root.after(30, self._process_queue)

    def on_start(self):
        if self.running:
            return
        if not YOLO_IMPORTED:
            messagebox.showerror('环境错误', f'无法导入 yolo_mycam 依赖：{YOLO_IMPORT_ERROR}')
            return

        # 状态初始化
        self.running = True
        self.start_epoch = time.time()
        self.last_metrics_ts = None
        self.total_distance = 0.0
        self.current_speed = 0.0
        self.digits_last = None
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        # 线程与定时器
        self.detector_stop.clear()
        self.detector_thread = DetectorThread(
            frame_queue=self.frame_queue,
            stop_event=self.detector_stop,
            score_threshold=0.5,
            camera_index=1,
        )
        self.detector_thread.start()

        self._update_runtime_metrics()
        self._process_queue()

        # 按钮状态
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)

    def on_stop(self):
        if not self.running:
            return
        self.running = False

        # 停止定时任务
        if self.metrics_job is not None:
            try:
                self.root.after_cancel(self.metrics_job)
            except Exception:
                pass
            self.metrics_job = None
        if self.queue_job is not None:
            try:
                self.root.after_cancel(self.queue_job)
            except Exception:
                pass
            self.queue_job = None

        # 停止线程
        try:
            self.detector_stop.set()
            if self.detector_thread is not None:
                self.detector_thread.join(timeout=2.0)
        except Exception:
            pass
        finally:
            self.detector_thread = None

        # 恢复按钮
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)

    def on_close(self):
        try:
            self.on_stop()
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = App(root)
    root.protocol('WM_DELETE_WINDOW', app.on_close)
    root.mainloop()


if __name__ == '__main__':
    main()

