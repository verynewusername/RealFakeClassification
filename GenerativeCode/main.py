# Author: Efe Gorkem Sirin
# Date: 2024-04-18
import requests
import base64
from PIL import Image
import io
import json
import os
import time

TXT2IMGURL = "http://127.0.0.1:7860/sdapi/v1/txt2img"

RSP_C = 4000
RV_C = 4000
RSP_G_C = 660
RV_G_C = 660

RSP_M = 1000
RV_M = 1000

RSP_F = 1000
RV_F = 1000


RV_G = {
    "prompt": "Close up face with glasses",
    "negative_prompt": "sketch, cartoon, drawing, anime",
    "width": 1024,
    "height": 1024,
    "seed": -1,
    # "sampler_name": "string",
    "hr_checkpoint_name": "realisticVisionV60B1_v60B1VAE.safetensors",
    "save_images": False,
    "send_images": True,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 7,
    "restore_faces": False,
    "sampler_index": "DPM++ SDE",
}

RSP_G = {
    "prompt": "Close up face with glasses",
    "negative_prompt": "(watermark:1.2), (text:1.2), (logo:1.2), (3d render:1.2), drawing, painting, crayon",
    "width": 1024,
    "height": 1024,
    "seed": -1,
    # "sampler_name": "realisticStockPhoto_v20.safetensors",
    "hr_checkpoint_name": "realisticStockPhoto_v20.safetensors",
    "save_images": False,
    "send_images": True,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 5,
    "restore_faces": False,
    "sampler_index": "DPM++ SDE",
}

RSP = {
    "prompt": "Close up face",
    "negative_prompt": "(watermark:1.2), (text:1.2), (logo:1.2), (3d render:1.2), drawing, painting, crayon",
    "width": 1024,
    "height": 1024,
    "seed": -1,
    # "sampler_name": "realisticStockPhoto_v20.safetensors",
    "hr_checkpoint_name": "realisticStockPhoto_v20.safetensors",
    "save_images": False,
    "send_images": True,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 5,
    "restore_faces": False,
    "sampler_index": "DPM++ SDE",
}

RV = {
    "prompt": "Close up face",
    "negative_prompt": "sketch, cartoon, drawing, anime",
    "width": 1024,
    "height": 1024,
    "seed": -1,
    # "sampler_name": "string",
    "hr_checkpoint_name": "realisticVisionV60B1_v60B1VAE.safetensors",
    "save_images": False,
    "send_images": True,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 7,
    "restore_faces": False,
    "sampler_index": "DPM++ SDE",
}


RSP_M_P = {
    "prompt": "Close up face, male",
    "negative_prompt": "(watermark:1.2), (text:1.2), (logo:1.2), (3d render:1.2), drawing, painting, crayon",
    "width": 1024,
    "height": 1024,
    "seed": -1,
    # "sampler_name": "realisticStockPhoto_v20.safetensors",
    "hr_checkpoint_name": "realisticStockPhoto_v20.safetensors",
    "save_images": False,
    "send_images": True,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 5,
    "restore_faces": False,
    "sampler_index": "DPM++ SDE",
}

RV_M_P = {
    "prompt": "Close up face, male",
    "negative_prompt": "sketch, cartoon, drawing, anime",
    "width": 1024,
    "height": 1024,
    "seed": -1,
    # "sampler_name": "string",
    "hr_checkpoint_name": "realisticVisionV60B1_v60B1VAE.safetensors",
    "save_images": False,
    "send_images": True,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 7,
    "restore_faces": False,
    "sampler_index": "DPM++ SDE",
}


RSP_F_P = {
    "prompt": "Close up face, female",
    "negative_prompt": "(watermark:1.2), (text:1.2), (logo:1.2), (3d render:1.2), drawing, painting, crayon",
    "width": 1024,
    "height": 1024,
    "seed": -1,
    # "sampler_name": "realisticStockPhoto_v20.safetensors",
    "hr_checkpoint_name": "realisticStockPhoto_v20.safetensors",
    "save_images": False,
    "send_images": True,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 5,
    "restore_faces": False,
    "sampler_index": "DPM++ SDE",
}

RV_F_P = {
    "prompt": "Close up face, female",
    "negative_prompt": "sketch, cartoon, drawing, anime",
    "width": 1024,
    "height": 1024,
    "seed": -1,
    # "sampler_name": "string",
    "hr_checkpoint_name": "realisticVisionV60B1_v60B1VAE.safetensors",
    "save_images": False,
    "send_images": True,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 7,
    "restore_faces": False,
    "sampler_index": "DPM++ SDE",
} 


# ===================================================================== #

def count_visible_entries(directory):
    visible_entries = [entry for entry in os.listdir(directory) if not entry.startswith('.')]
    return len(visible_entries)

def save_images_from_json(json_data, output_folder, seed):
    images = json_data.get("images", [])
    
    for idx, img_data in enumerate(images):
        # Decode base64 image data
        img_bytes = base64.b64decode(img_data)
        
        # Convert bytes to image
        img = Image.open(io.BytesIO(img_bytes))
        
        # Save the image
        img_path = f"{output_folder}/{seed}.png"
        img.save(img_path)

def main():
    # Get start time
    start_time = time.time()

    # Get the image counts in the directory
    rvCount = count_visible_entries("RV")
    rspCount = count_visible_entries("RSP")
    rspgCount = count_visible_entries("RSPG")
    rvgCount = count_visible_entries("RVG")

    rvmCount = count_visible_entries("RV_M")
    rspmCount = count_visible_entries("RSP_M")
    rspfCount = count_visible_entries("RSP_F")
    rvfCount = count_visible_entries("RV_F")

    while rvCount < RV_C or rspCount < RSP_C or rspgCount < RSP_G_C or rvgCount < RV_G_C or rvmCount < RV_M or rspmCount < RSP_M or rspfCount < RSP_F or rvfCount < RV_F:
        # Info print
        # print("RSP Count: ", rspCount, "RV Count: ", rvCount)
        print("RSP Count: ", rspCount, "RV Count: ", rvCount, "RSP_G Count: ", rspCount, "RV_G Count: ", rvCount)

        if rspCount < RSP_C:
            response = requests.post(url=TXT2IMGURL, json=RSP)

            # print(response.json())  # Assuming the response is JSON

            # From the response get the seed
            info_data = json.loads(response.json()["info"])
            seed = info_data["seed"]

            # Save the "images" to a file
            save_images_from_json(response.json(), "RSP", seed)
            
        elif rvCount < RV_C:
            response = requests.post(url=TXT2IMGURL, json=RV)

            # print(response.json())  # Assuming the response is JSON

            # From the response get the seed
            info_data = json.loads(response.json()["info"])
            seed = info_data["seed"]

            # Save the "images" to a file
            save_images_from_json(response.json(), "RV", seed)
        elif rspgCount < RSP_G_C:
            response = requests.post(url=TXT2IMGURL, json=RSP_G)

            # print(response.json())  # Assuming the response is JSON

            # From the response get the seed
            info_data = json.loads(response.json()["info"])
            seed = info_data["seed"]

            # Save the "images" to a file
            save_images_from_json(response.json(), "RSPG", seed)
        
        elif rvgCount < RV_G_C:
            response = requests.post(url=TXT2IMGURL, json=RV_G)

            # print(response.json())  # Assuming the response is JSON

            # From the response get the seed
            info_data = json.loads(response.json()["info"])
            seed = info_data["seed"]

            # Save the "images" to a file
            save_images_from_json(response.json(), "RVG", seed)

        elif rvmCount < RV_M:
            response = requests.post(url=TXT2IMGURL, json=RSP_M_P)

            # print(response.json())  # Assuming the response is JSON

            # From the response get the seed
            info_data = json.loads(response.json()["info"])
            seed = info_data["seed"]
            save_images_from_json(response.json(), "RV_M", seed)
        elif rspmCount < RSP_M:
            response = requests.post(url=TXT2IMGURL, json=RSP_M_P)

            # print(response.json())  # Assuming the response is JSON

            # From the response get the seed
            info_data = json.loads(response.json()["info"])
            seed = info_data["seed"]
            save_images_from_json(response.json(), "RSP_M", seed)
        elif rspfCount < RSP_F:
            response = requests.post(url=TXT2IMGURL, json=RSP_F_P)

            # print(response.json())  # Assuming the response is JSON

            # From the response get the seed
            info_data = json.loads(response.json()["info"])
            seed = info_data["seed"]
            save_images_from_json(response.json(), "RSP_F", seed)
        elif rvfCount < RV_F:
            response = requests.post(url=TXT2IMGURL, json=RV_F_P)

            # print(response.json())  # Assuming the response is JSON

            # From the response get the seed
            info_data = json.loads(response.json()["info"])
            seed = info_data["seed"]
            save_images_from_json(response.json(), "RV_F", seed)
        else:
            print("All images are generated.")
            break

        # print(response.json())
        rvCount = count_visible_entries("RV")
        rspCount = count_visible_entries("RSP")
        rspgCount = count_visible_entries("RSPG")
        rvgCount = count_visible_entries("RVG")
        
        rvmCount = count_visible_entries("RV_M")
        rspmCount = count_visible_entries("RSP_M")
        rspfCount = count_visible_entries("RSP_F")
        rvfCount = count_visible_entries("RV_F")

    # Get end time
    end_time = time.time()
    # Calculate the time difference
    time_diff = end_time - start_time
    # print d h m s
    print("Time difference: ", time.strftime("%H:%M:%S", time.gmtime(time_diff)))

if __name__ == "__main__":
    main()
