
# Pizza sale counting

This project utilized YOLO11x to detect and count the number of pizzas sold in a period of time.
Model training process followed tutorial of Ultralytics. 

## 1. Brief

- About the deployment:
    - Sorry for this uncovenient set up, i will try to find another way for deployment.
    - The dockerfile is currently set up for CPU inference only, since i only have personal computer with AMD GPU not NVDIA one -> cannot verify whether it run successfully.
    - To save your time i uploaded the DEMO on the youtube link: https://youtu.be/kx3PM2kyerw
- About the system performance:
    - YOLO11x is good at detecting "pizza" in the scence but its performance is affected by motion blur.
    - During processing the system may miss counting because the pizza was covered by the staff while passing triggering line. Another case is that staff re-baked the pizza so
    the line got triggered more than once.

## 2. File requirement

- best.pt is larger than 100MB, therefore, i could not push it to github, please download manually: https://drive.google.com/file/d/132BtGroW1dugbQdbr6Kd3MsrjLBop3YT/view?usp=sharing
- The video being processed is ```SOHO_1462_CH03``` will also need to be downloaded manually: https://drive.google.com/file/d/11LSPgLVhZftVhe3ft1kRIawGMdTPReiK/view?usp=sharing
- After finishing, move the ```best.pt``` to folder ```pizza-counting/app/``` and move the video to ```pizza-counting/videos/```


## 3. Installation
- Clone repo to your machine.
    
    ``` cd pizza-counting ```

    ```docker build -t pizza-counting . ```

## 4. Usage
- Run docker with command
   ```docker run --rm -v "$(Get-Location)/wheels:/wheels" python:3.10 bash -c "pip download opencv-python==4.11.0.86 -d /wheels"```

    ```docker run --rm -v ${PWD}/videos:/app/videos -v ${PWD}/app/main.py:/app/main.py -e INPUT_VIDEO=/app/videos/1462_CH03_20250607192844_202844.mp4 -e OUTPUT_VIDEO=/app/videos/output107.avi pizza-counting```

- Change the range of video going to be processed by adjusting ```START_FRAME``` and ```END_FRAME``` in the ```main.py``` file.
- Change the video file name in the command above to match your video filename if needed.

- Output video is a sequence of annotated frames of the input video: where ```out``` is the number of pizzas sold.

![alt text](images/image.png)
