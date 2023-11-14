CORRUPTIONS_BY_FREQUENCY = {
    'low': ['snow', 'frost', 'fog', 'brightness', 'contrast'],
    'medium': ['motion_blur', 'zoom_blur', 'defocus_blur', 'glass_blur', 'elastic_transform', 'jpeg_compression', 'pixelate'],
    'high': ['gaussian_noise', 'shot_noise', 'impulse_noise']
    }
CORRUPTIONS = {
    1: 'snow',
    2: 'frost',
    3: 'fog',
    4: 'brightness',
    5: 'contrast',
    6: 'motion_blur',
    7: 'zoom_blur',
    8: 'defocus_blur',
    9: 'glass_blur',
    10: 'elastic_transform',
    11: 'jpeg_compression',
    12: 'pixelate',
    13: 'gaussian_noise',
    14: 'shot_noise',
    15: 'impulse_noise'
}
SEVERITIES = [str(i+1) for i in range(5)]