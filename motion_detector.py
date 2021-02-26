import cv2, time, pandas
from datetime import datetime

video = cv2.VideoCapture(0)


first_frame = None
status_list = [None, None]      # initiate to access
times = []
df = pandas.DataFrame(columns=['Start', 'End'])   # column names

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21),0)  # (object, (width, height) of Gaussian Kernal = parameters for bluriness, standard deviation)
    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]  # threshold methods
    # thresh_delta[0] = 30  -- threshold value we defined

    # smooth the threshold otherwise shadows are couted as an object
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
    # (object, dilate kernal, iterations to remove noises = bigger, better)

    # Draw contour lines around the object
    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # (copy of the frame that you want to find contour from,
    # draw external contour,
    # approximation method opencv apply)

    # filter out the contour lines
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:  # the larger the number, the bigger the object you draw contour around
            continue

        status = 1
        # (x,y) = upper left, (w, h) = lower right
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    status_list.append(status)
    status_list = status_list[-2:]
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow('Capturing', gray)
    cv2.imshow('Delta Frame', delta_frame)
    cv2.imshow('Threshold Frame', thresh_frame)
    cv2.imshow('Color Frame', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:  # in case camera close when an object is still there
            times.append(datetime.now())
        break

for i in range(0, len(times), 2):   # use range instead of the list directly due to the None in initiation, and DataFrame can only append a series
    if times[i] and times[i+1]:
        df = df.append({'Start': times[i], 'End': times[i+1]}, ignore_index = True)

df.to_csv(f'Times_EndAt_{times[-1].strftime("%Y-%m-%d_%H-%M-%S")}.csv')
video.release()
cv2.destroyAllWindows
