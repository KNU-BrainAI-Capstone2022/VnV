try:
    import cv2
except ImportError:
    print("ERROR python-opencv must be installed")
    exit(1)

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera not found!")
    exit(1)

cap.set(3,640)
cap.set(4,360)
print(f"Resolution : {cap.get(3)}x{cap.get(4)}")
print(f"FPS : {cap.get(5)}")

cv2.namedWindow("C920", cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)
print("Running, ESC or Ctrl-c to exit...")

while True:
    ret, img = cap.read()
    if ret == False:
        print("Error reading image")
        break
    print(img.shape)
    # img = cv2.flip(img,1)
    #print(img.shape)
    cv2.imshow("C920", img)
    #print("Captrue Done")
    if cv2.waitKey(50) == 27: # 20fps
        cv2.imwrite("sample.jpg",img)
        break
cv2.destroyAllWindows()
