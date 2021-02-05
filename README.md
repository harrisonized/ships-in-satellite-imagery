# Ships in Satellite Imagery

The purpose of this project is to learn some basic image classification and object detection techniques. I felt that this particular project was a good starting point, because the data is clean and freely available [here](https://www.kaggle.com/rhammell/ships-in-satellite-imagery). This work closely follows byrachonok's notebook, found [here](https://www.kaggle.com/byrachonok/keras-for-search-ships-in-satellite-image/comments).



#### Getting Started

To set up a conda environment, assuming you have a GPU, follow [this guide](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/):

```bash
conda create -n tf-gpu tensorflow-gpu python=3.6
```

Pip or conda install the following libraries:

```
matplotlib==3.3.2
pandas==1.1.5
Pillow==8.1.0
imutils==0.5.4
tqdm==4.56.0
tensorflow==2.2.0
keras==2.4.3
opencv==4.5.1-dev
```

Strictly speaking, the versions don't matter all that much, but if you run into some dependency issues, check that your versions match up. The only requirement listed here that isn't required is `opencv`. I had a hard time building it on my machine, but that's because I wanted `ffmpeg` support. To build it, follow the instructions [here](https://stackoverflow.com/questions/50816241/compile-opencv-with-cmake-to-integrate-it-within-a-conda-env). Otherwise, you can safely skip this.



#### License

This work is distributed under the terms of the [Apache 2.0 License](https://github.com/harrisonized/ships-in-satellite-imagery/blob/master/LICENSE). Please remember to [give credit](https://github.com/harrisonized/ships-in-satellite-imagery/blob/master/CREDITS.md) if you reuse any components of this project.

