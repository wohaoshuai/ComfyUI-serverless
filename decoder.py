from compressor import save_encoded_images_to_file, encode_images, decode_images, read_encoded_images_from_file


# Read the encoded images from the file
encoded_images = read_encoded_images_from_file("encoded_images.txt")

# Decode the images back into a list of PIL Images
decoded_images = decode_images(encoded_images)

# Save the decoded images to individual files
for i, image in enumerate(decoded_images):
    image.save(f"decode/test{i}.png")
