import torch
import torch.nn as nn
import torch.nn.functional as F

import tkinter as tk
from PIL import ImageGrab
import cv2

# load model
def VGG_block(num_convs, in_channels, out_channels):
    blk = []
    for _ in range(num_convs):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    
    return nn.Sequential(*blk)

def VGG(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(VGG_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = VGG(conv_arch)
checkpoint = torch.load('VGG_checkpoint_epoch40.pt', map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['model_state_dict'])
net = net.float()

def draw(event):
    '''
    event to draw with mouse
    '''
    x, y = event.x, event.y
    canvas.create_oval(x-4, y-4, x+4, y+4, fill='black',outline='black')

def predict():
    '''
    event to predict the number
    '''
    # get the correct size of the image

    # note: below is the correct one, but doesn't work on my mac
    # x=root.winfo_rootx()+canvas.winfo_x()
    # y=root.winfo_rooty()+canvas.winfo_y()
    # x1=x+canvas.winfo_width()
    # y1=y+canvas.winfo_height()

    # note: below is the incorrect one but due to the bug in screenshot on mac, only this works for me
    x = 199 + 199
    y = 60 + 60
    x1 = x + 224 * 2
    y1 = y + 224 * 2
    
    # make and save the screenshot
    ImageGrab.grab().crop((x, y, x1, y1)).save('image.png')

    # load the saved image and convert to greyscale
    image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

    # process the image
    # 1. change to black background and white number due to the training data is like this
    image = 255 - image
    # 2. resize to (224, 224), which is the size of the input data for my VGG model
    image = cv2.resize(image, (224, 224))
    # 3. convert to tensor and reshape to (1, 1, 224, 224)
    image = torch.tensor(image, dtype=torch.float32).reshape(1, 1, 224, 224)

    # make prediction
    net.eval()
    result = net(image)
    prediction = int(torch.argmax(F.softmax(result, dim=1)))

    # show the prediction
    result_label.config(text=f'Prediction: {prediction}')

# create the GUI window
root = tk.Tk()
root.title('手写识别')
frame = tk.Frame(root)
frame.pack()

# create the canvas
canvas = tk.Canvas(frame, width=224, height=224, bg='white')

# create the 'clear' button
clear_button = tk.Button(
    frame, text='Clear', bg='red',
    command=lambda: canvas.delete('all')
    )

# create the 'predict' button
predict_button = tk.Button(
    frame, text='Predict', command=predict
    )

# create label to show prediction
result_label = tk.Label(frame, text='Prediction ')

# create label to show info.
info_author_label = tk.Label(frame, text='作者：何慕菱')
info_gui_label = tk.Label(frame, text='点击Predict按钮进行识别\n点击Clear按钮清空画布')

# use grid to locate the widgets
canvas.grid(row=0, rowspan=3, column=0, columnspan=3)

predict_button.grid(row=3, column=0)
clear_button.grid(row=3, column=2)

result_label.grid(row=2, rowspan=2, column=3, columnspan=2)
info_author_label.grid(row=0, column=3, columnspan=2)
info_gui_label.grid(row=1, column=3, columnspan=2)

# bind event
canvas.bind('<B1-Motion>', draw)

# begin main loop
root.mainloop()
