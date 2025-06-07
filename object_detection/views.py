import os
import json
import cv2
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import ensure_csrf_cookie
import redis
import hashlib
from .utils import predict_objects_fast_rcnn, draw_boxes, load_categories, ROIPoolingLayer
from tensorflow.keras.layers import Lambda
import logging
import warnings

# Suppress TensorFlow deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.BASE_DIR, 'object_detection.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom Lambda function
def custom_lambda(x):
    logger.debug("Executing custom_lambda: reducing mean along last axis")
    return tf.reduce_mean(x, axis=-1)

# Connect to Redis
try:
    redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
    logger.info("Successfully connected to Redis at 127.0.0.1:6379")
except redis.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")
    raise redis.ConnectionError(f"Cannot connect to Redis at 127.0.0.1:6379: {str(e)}")

# Load the Fast R-CNN model
try:
    logger.debug("Loading Fast R-CNN model from media/fast_rcnn_model.h5")
    fast_rcnn_model = tf.keras.models.load_model(
        os.path.join(settings.MEDIA_ROOT, 'fast_rcnn_model.h5'),
        custom_objects={
            'ROIPoolingLayer': ROIPoolingLayer,
            'tf': tf,
            'Lambda': Lambda(custom_lambda)
        },
        compile=False
    )
    logger.info("Successfully loaded Fast R-CNN model")
except Exception as e:
    logger.error(f"Failed to load Fast R-CNN model: {str(e)}")
    raise Exception(f"Model loading failed: {str(e)}")

# Load categories from JSON
try:
    logger.debug("Loading categories")
    categories = load_categories()
    logger.info(f"Loaded {len(categories)} categories")
except Exception as e:
    logger.error(f"Failed to load categories: {str(e)}")
    raise Exception(f"Category loading failed: {str(e)}")

@ensure_csrf_cookie
def index(request):
    logger.debug("Rendering index page")
    return render(request, 'object_detection/index.html')

def process_image(request):
    logger.debug("Processing image request")
    
    if request.method != 'POST' or 'image' not in request.FILES:
        logger.error("Invalid request: No image uploaded or incorrect method")
        return JsonResponse({'success': False, 'error': 'No image uploaded'}, status=400)

    image_file = request.FILES['image']
    logger.debug(f"Received image file: {image_file.name}")

    # Generate cache key using image file hash
    try:
        image_data = image_file.read()
        cache_key = hashlib.md5(image_data).hexdigest()
        logger.debug(f"Generated cache key: {cache_key}")
    except Exception as e:
        logger.error(f"Failed to generate cache key: {str(e)}")
        return JsonResponse({'success': False, 'error': 'Failed to process image file'}, status=400)

    # Check Redis cache
    try:
        cached_result = redis_client.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for key {cache_key}")
            print(f"Cache hit for key {cache_key}")
            result = json.loads(cached_result)
            logger.debug(f"Returning cached result: {result}")
            print(f"Returning cached result: {result}")
            return JsonResponse({
                'success': True,
                'result_image': result['result_image'],
                'detections': result['detections'],
                'total_objects': len(result['detections'])
            })
        else:
            logger.info(f"Cache miss for key {cache_key}")
            print(f"Cache miss for key {cache_key}")
    except redis.RedisError as e:
        logger.error(f"Redis error during cache check: {str(e)}")
        logger.warning("Proceeding without cache due to Redis error")

    # Process image if not in cache
    try:
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
        filename = fs.save(image_file.name, image_file)
        image_path = fs.path(filename)
        logger.debug(f"Saved uploaded image to: {image_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded image: {str(e)}")
        return JsonResponse({'success': False, 'error': 'Failed to save image'}, status=500)

    # Read and preprocess image
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            fs.delete(filename)
            return JsonResponse({'success': False, 'error': 'Invalid image'}, status=400)
        logger.debug(f"Read image with shape: {img.shape}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logger.debug("Converted image to RGB")
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        fs.delete(filename)
        return JsonResponse({'success': False, 'error': 'Image processing error'}, status=500)

    # Make predictions
    try:
        logger.info("Starting object detection with Fast R-CNN")
        pred_boxes, pred_classes, pred_scores = predict_objects_fast_rcnn(
            fast_rcnn_model, img_rgb, confidence_threshold=0.2
        )
        logger.info(f"Detected {len(pred_boxes)} objects")
    except Exception as e:
        logger.error(f"Object detection failed: {str(e)}")
        fs.delete(filename)
        return JsonResponse({'success': False, 'error': 'Object detection error'}, status=500)

    # Prepare detection results
    try:
        pred_labels = [categories.get(cls, str(cls)) for cls in pred_classes]
        logger.debug(f"Generated {len(pred_labels)} prediction labels")
        img_with_pred = draw_boxes(img_rgb, pred_boxes, pred_labels, color=(0, 255, 0))
        logger.debug("Drew bounding boxes on image")
    except Exception as e:
        logger.error(f"Failed to prepare detection results: {str(e)}")
        fs.delete(filename)
        return JsonResponse({'success': False, 'error': 'Failed to process detection results'}, status=500)

    # Save result image
    try:
        result_filename = f'result_{filename}'
        result_path = os.path.join(settings.MEDIA_ROOT, 'results', result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(img_with_pred, cv2.COLOR_RGB2BGR))
        logger.debug(f"Saved result image to: {result_path}")
    except Exception as e:
        logger.error(f"Failed to save result image: {str(e)}")
        fs.delete(filename)
        return JsonResponse({'success': False, 'error': 'Failed to save result image'}, status=500)

    # Prepare detections for response
    try:
        detections = [
            {
                'class': categories.get(cls, str(cls)),
                'confidence': float(score),
                'bbox': box
            } for cls, score, box in zip(pred_classes, pred_scores, pred_boxes)
        ]
        logger.debug(f"Prepared {len(detections)} detections for response")
    except Exception as e:
        logger.error(f"Failed to prepare detections for response: {str(e)}")
        fs.delete(filename)
        return JsonResponse({'success': False, 'error': 'Failed to prepare response'}, status=500)

    # Cache result in Redis (5 minutes = 300 seconds)
    try:
        cache_data = {
            'result_image': f'/media/results/{result_filename}',
            'detections': detections
        }
        redis_client.setex(cache_key, 300, json.dumps(cache_data))
        logger.info(f"Cached result for key {cache_key} with TTL 300 seconds")
    except redis.RedisError as e:
        logger.error(f"Failed to cache result in Redis: {str(e)}")
        logger.warning("Result not cached, but continuing with response")

    # Clean up uploaded file
    try:
        fs.delete(filename)
        logger.debug(f"Deleted uploaded file: {filename}")
    except Exception as e:
        logger.warning(f"Failed to delete uploaded file {filename}: {str(e)}")

    # Return response
    logger.info("Returning successful response")
    return JsonResponse({
        'success': True,
        'result_image': f'/media/results/{result_filename}',
        'detections': detections,
        'total_objects': len(detections)
    })