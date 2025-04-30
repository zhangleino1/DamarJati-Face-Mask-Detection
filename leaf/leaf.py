from ultralytics import YOLO
import cv2

if __name__ == "__main__":
    # 直接指定图片路径
    img_path = "./chaye.jpg"

    # 加载模型
    model = YOLO("yolo11m.pt")
    # 运行检测，取第一个结果
    results = model(img_path)[0]
    # 在原图上绘制检测框
    annotated = results.plot()
    # 展示结果
    cv2.imshow("Tea Leaf Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()