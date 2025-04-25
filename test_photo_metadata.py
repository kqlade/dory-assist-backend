from app.utils.photo_metadata import extract_photo_metadata

if __name__ == "__main__":
    result = extract_photo_metadata("heic_test.HEIC")
    print(result) 