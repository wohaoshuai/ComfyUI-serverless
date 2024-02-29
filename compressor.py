from PIL import Image
import zlib
import base64
import io

def encode_images(images, size=400, quality=80):
    encoded_images = []
    for i, image in enumerate(images):
        # Resize the image to 256x256
        resized_image = image.resize((size, size))
        
        # Compress the image with lossy quality
        compressed_image = io.BytesIO()
        resized_image.save(compressed_image, format='JPEG', quality=quality)
        compressed_image = compressed_image.getvalue()
        
        # Compress the image bytes using zlib
        # compressed_bytes = zlib.compress(compressed_image)
        
        # Encode the compressed bytes as base64
        encoded_bytes = base64.b64encode(compressed_image)
        
        # Append the encoded image to the list
        encoded_images.append(encoded_bytes)
    
    return encoded_images

def encode_webps(images, size=400, quality=80):
    encoded_images = []
    for i, image in enumerate(images):
        # Resize the image to the specified size (default: 400x400)
        resized_image = image.resize((size, size))
        
        # Compress the image with lossy quality in WEBP format
        compressed_image = io.BytesIO()
        resized_image.save(compressed_image, format='WEBP', quality=quality)  # Change format to 'WEBP'
        compressed_image = compressed_image.getvalue()
        
        # Optionally compress the image bytes using zlib
        # compressed_bytes = zlib.compress(compressed_image)
        
        # Encode the compressed bytes as base64
        encoded_bytes = base64.b64encode(compressed_image)
        
        # Append the encoded image to the list
        encoded_images.append(encoded_bytes)
    
    return encoded_images

def save_encoded_images_to_file(encoded_images, filename):
    with open(filename, "w") as f:
        for encoded_image in encoded_images:
            f.write(encoded_image.decode() + "\n")

def save_encoded_images_to_string(encoded_images):
    encoded_string = ""
    for encoded_image in encoded_images:
        encoded_string += encoded_image.decode() + "\n"
    return encoded_string

def read_encoded_images_from_file(filename):
    with open(filename, "r") as f:
        encoded_images = [line.strip() for line in f]
    return encoded_images

def decode_images(encoded_images):
    decoded_images = []
    for i, encoded_image in enumerate(encoded_images):
        # Decode the base64-encoded image
        compressed_bytes = base64.b64decode(encoded_image)
        
        # Decompress the image bytes using zlib
        decompressed_bytes = zlib.decompress(compressed_bytes)
        
        # Create a PIL Image from the decompressed bytes
        image = Image.open(io.BytesIO(decompressed_bytes))
        
        # Append the decoded image to the list
        decoded_images.append(image)
    
    return decoded_images
