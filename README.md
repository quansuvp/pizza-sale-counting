
# Pizza sale counting

This project utilized YOLO11x to detect and count the number of pizzas sold in a period of time.

## 1. Installation
- Clone repo to your machine.
    
    ``` cd pizza-counting ```

    ```docker build -t pizza-counting . ```

## 2. Usage
- Run docker with command

    ```docker run --rm -v ${PWD}/videos:/app/videos -v ${PWD}/app/main.py:/app/main.py -e INPUT_VIDEO=/app/videos/input107.mp4 -e OUTPUT_VIDEO=/app/videos/output107.avi pizza-counting```

- Change the range of video going to be processed by adjusting ```START_FRAME``` and ```END_FRAME``` in the ```main.py``` file

![alt text](images/image.png)
