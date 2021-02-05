import imutils
import numpy as np


# Functions
# # flatten_image_array
# # flat_image_to_array
# # pyramid
# # sliding_window


def flatten_image_array(image_array: np.ndarray, n_colors=3, height=80, width=80):
    """Converts an image array in this format:
    array([[[ 92, 109,  90],
            [ 92, 108,  89],
            [ 91, 107,  88],
            ...,
            [ 96, 109,  93],
            [ 95, 109,  92],
            [ 93, 107,  91]],

           [[ 93, 108,  91],
            [ 91, 108,  88],
            [ 91, 108,  88],
            ...,
            [ 95, 109,  92],
            [ 95, 109,  92],
            [ 95, 109,  92]],

           ...,
           
           [[ 94, 106,  90],
            [ 91, 103,  87],
            [ 93, 105,  89],
            ...,
            [ 96, 111,  94],
            [ 94, 110,  93],
            [ 94, 110,  93]]], dtype=uint8)
           
    To this:
    [92, 92, 91, ..., 96, 95, 93,  # 0:80
     93, 91, 91, ..., 95, 95, 95,  # 80:160
     ...
     94, 91, 93, ..., 96, 94, 94,  # 6320:6400
     109, 108, 107, ..., 109, 109, 107,  # 6400:6480
     108, 108, 108, ..., 109, 109, 109,  # 6480:6560
     ...
     106, 103, 105, ..., 111, 110, 110,  # 12720:12800
     90, 89, 88, ..., 93, 92, 91,  # 12800:12880
     91, 88, 88, ..., 92, 92, 92,  # 12880:12960
     ...
     90, 87, 97..., 94, 93, 93]  # 19120:19200
     
    This was used in ships-in-satellite-imagery to store image data in json
    See: https://www.kaggle.com/rhammell/ships-in-satellite-imagery
    """
    return list(map(int, np.reshape(image_array.swapaxes(2, 1).swapaxes(1, 0), [n_colors*height*width])))
    
    
def flat_image_to_array(flat_image: list, n_colors=3, height=80, width=80):
    """The reverse operation of flatten_image_array().
    """
    return np.reshape(flat_image, [n_colors, height, width]).swapaxes(0, 1).swapaxes(1, 2).astype(np.uint8)


def pyramid(image, scale=1.5, min_height=30, min_width=30):
    """Generate progressively smaller images until min_height and min_width
    
    Example usage:
    image.shape = (1777, 2825, 4)
    
    resized_images = []
    for resized_image in pyramid(image):
        resized_images.append(resized_image)
    
    [x.shape for x in resized_images]
    [(1777, 2825, 4),
     (1184, 1883, 4),
     (789, 1255, 4),
     (525, 836, 4),
     (349, 557, 4),
     (232, 371, 4),
     (154, 247, 4),
     (102, 164, 4),
     (67, 109, 4),
     (44, 72, 4)]
    
    See: https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
    """
    yield image
    height, width, n_colors = image.shape
    
    while height/scale > min_height and width/scale > min_width:
        image = imutils.resize(image, width=int(width/scale))  # downscale
        yield image
        
        height, width, n_colors = image.shape

   
def sliding_window(image, step_size=10, window_height=80, window_width=80):
    """Slide a window across the image
    image should be a numpy array
    x, y coordinates on top left corner
    
    Example usage:
    image.shape = (1777, 2825, 4)
    windows = []
    for (x, y, window) in sliding_window(image, step_size=80):        
        windows.append([x, y])
        
    windows
    [[0, 0],
     [80, 0],
     [160, 0],
     ...
     [2560, 0],
     [2640, 0],
     [2720, 0],
     [0, 80],
     [80, 80],
     [160, 80],
     ...
     ...
     [2560, 1680],
     [2640, 1680],
     [2720, 1680]
    ]
    
    len(windows)
    770

    Note: This will not work for large arrays.
    coordinates = []
    windows = []
    for x, y, window in tqdm(sliding_window(scene_np, step_size=80)):
        coordinates.append((x, y))
        windows.append(window.tolist())

    coordinates = np.array(coordinates)
    windows = np.array(windows)

    Better to do this:
    predictions = []
    for x, y, window in tqdm(sliding_window(scene_np, step_size=40)):
        prediction = model.predict(np.array([windows]))
        prediction.append(prediction)
        
    See: https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
    """
    height, width, n_colors = image.shape
    print('num_it:', int((height-window_height)/step_size+1)*int((width-window_width)/step_size+1))
    
    for y in range(0, height-window_height, step_size):
        for x in range(0, width-window_width, step_size):
            window = image[y:y+window_height, x:x+window_width]
            yield (x, y, window)
