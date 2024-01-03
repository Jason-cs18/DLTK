# text-to-video with damo-vilab/text-to-video-ms-1.7b
# official docs: https://huggingface.co/docs/diffusers/api/pipelines/text_to_video#diffusers.TextToVideoSDPipeline.encode_prompt.negative_prompt 

import time

import cv2
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

def play_video(video_path: str) -> None:
    cap= cv2.VideoCapture(video_path)

    fps= int(cap.get(cv2.CAP_PROP_FPS))

    # print("This is the fps ", fps)

    if cap.isOpened() == False:
        print("Error File Not Found")

    while cap.isOpened():
        ret,frame= cap.read()

        if ret == True:

            time.sleep(1/fps)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()



def infer_pytorch(text_prompt: str, accelerate=False) -> str:
    model_id = "damo-vilab/text-to-video-ms-1.7b"
    print("*"*20)
    print(f"NVIDIA GPU: {torch.cuda.get_device_name()} with {torch.cuda.get_device_properties(0).total_memory / 10**9:02f} G")
    print(f"Model: {model_id}")
    print(f"Accelerate: {accelerate}")
    print(f"Text Prompt: {text_prompt}")
    print(f"Torch Version: {torch.__version__}")
    # print(f"#Params: {pipeline.model.num_parameters() / 10**9}G")
    print("*"*20)
    if accelerate: #TBD
        pass
    else:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        pipe = pipe.to("cuda")

    #prompt = "Spiderman is surfing"
    video_frames = pipe(text_prompt).frames
    # print(video_path)
    video_path = export_to_video(video_frames)
    return video_path

def main():
    text_prompt = "Superman is surfing"
    video_path = infer_pytorch(text_prompt)
    # video_path = "/tmp/tmpfu7h6_88.mp4"
    play_video(video_path)


if __name__ == "__main__":
    main()