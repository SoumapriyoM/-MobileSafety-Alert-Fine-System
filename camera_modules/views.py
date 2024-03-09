from django.http import StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from . import camera
import cv2


def home(request):
    if request.method == "POST":
        return render(request, "index.html")
    else:
        return render(request, "index.html")


def __get_timing(fps):
    if fps > 1000 or fps <= 0:
        return 1
    return int(1000 / fps)


def __gen(camera: camera.webcam, timing=1):
    while True:
        frame = camera.get_frame()
        cv2.waitKey(timing)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


def video(request):
    mycam = camera.webcam()
    return StreamingHttpResponse(
        __gen(mycam, 4), content_type="multipart/x-mixed-replace; boundary=frame"
    )
    

def analyse(request):
    return render(request, "application.html")