def downsize_clip(frames, fps):
    """
    Downsize a video clip to the specified frames per second (fps).
    
    Parameters:
    frames (list): A list of frames from the video clip.
    fps (int): The desired frames per second for the downsized clip.
    """
    step = int(30 / fps)  
    clip_frames = frames[::step]   
    return clip_frames  

    