import cv2
import mediapipe
import numpy as np

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

with handsModule.Hands(static_image_mode=True) as hands:
    image = cv2.imread(r'C:\Users\Cullen\Downloads\IMG_2555.jpg')

    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    imageHeight, imageWidth, _ = image.shape

    if results.multi_hand_landmarks != None:
        for handLandmarks in results.multi_hand_landmarks:
            drawingModule.draw_landmarks(image, handLandmarks, handsModule.HAND_CONNECTIONS)
            for point in handsModule.HandLandmark:
                normalizedLandmark = handLandmarks.landmark[point]
                pixelCoord = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                          normalizedLandmark.y,
                                                                                          imageWidth, imageHeight)

                Ps, Pe = pixelCoord
                Ps1 = Ps - 20

                Pe1 = Ps1 + 40

                startPoint = (Ps1, Pe)
                endPoint = (Pe1, Pe)

                image = cv2.line(image, startPoint, endPoint, color=(0,255,0), thickness=2)

    cv2.imshow('Hand', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
