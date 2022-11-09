import random
import pyautogui
import time

while True:
    ans = random.randint(0,3)
    if ans == 0:
        sc = random.randint(-400,400)
        pyautogui.scroll(sc)
        pyautogui.click(x=851, y=498, clicks=1, button='left')  # 클릭 해제
    else:
        sc = random.randint(-400, 400)
        pyautogui.scroll(sc)
    sleep = random.randint(0,100)
    time.sleep(sleep)
