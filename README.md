# Hand Gesture Mouse Controller

This project allows you to control your computer’s mouse using **hand gestures** with the help of [OpenCV](https://opencv.org/), [MediaPipe](https://mediapipe.dev/), [NumPy](https://numpy.org/), and [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/).

It tracks your hand using a webcam and translates **finger movements** into **mouse actions** such as moving the cursor, left click, and right click.

---

## ✨ Features

- **Real-time Hand Tracking** using MediaPipe.
- **Mouse Cursor Movement** using your index finger.
- **Left Click** gesture (Thumb crosses index finger base).
- **Right Click** gesture (Middle finger tip moves below knuckle).
- **Smooth Movement** using a smoothing factor.
- Works with **any standard webcam**.

---

## 🎯 Gestures

| Gesture                          | Action       |
|----------------------------------|--------------|
| Move Index Finger                | Move Cursor  |
| Thumb crosses Index Finger base  | Left Click   |
| Middle Finger bends below knuckle| Right Click  |

---

## ⚠️ Notes

- Ensure you have a **working webcam**.
- Works best in **good lighting conditions**.
- Gesture detection accuracy may vary with camera quality.

---

## 📌 Future Improvements

- Add **drag and drop** gesture.
- Add **double click** gesture.
- Multi-hand support for advanced actions.

---



## 📂 Project Structure
```plaintext
wordcounter/
 ├──  README.md
 ├──  requirements.txt 
 └── workingonhandmouse.py
